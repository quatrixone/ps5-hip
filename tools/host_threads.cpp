#include "host_threads.h"
#include <atomic>
#include <cerrno>
#include <pthread.h>
#include <time.h>

namespace ps5hip_threads {
namespace {
thread_local unsigned scalar_tls = 17; // Trivial TLS; no __cxa_thread_atexit dependency.
struct Context;
struct Worker { Context *context; unsigned id; };
struct Context {
    pthread_mutex_t gate{}, work{};
    pthread_cond_t changed{};
    pthread_key_t key{};
    timespec deadline{};
    Worker worker[2]{};
    std::atomic<int> error{0};
    std::atomic<bool> synchronization_uncertain{false};
    std::atomic<unsigned> destroyed[2]{};
    unsigned arrived = 0, turn = 0, counter = 0, turns = 0, busy = 0;
    unsigned observed_tls[2]{};
    unsigned main_key_value = 0xabad;
    bool released = false, abort = false;
};
static_assert(std::atomic<unsigned>::is_always_lock_free && std::atomic<int>::is_always_lock_free);
Context state; // Retained even if join/close is uncertain; no library-calling destructor.
std::atomic_flag invoked = ATOMIC_FLAG_INIT;
void error(Context &c, int e) noexcept {
    int zero = 0; c.error.compare_exchange_strong(zero, e ? e : EINVAL);
}
void key_destructor(void *value) noexcept {
    auto *w = static_cast<Worker *>(value);
    if (w != &state.worker[0] && w != &state.worker[1]) { error(state, EINVAL); return; }
    auto &c = *w->context;
    if (scalar_tls != 0x100u + w->id) error(c, EINVAL);
    if (c.destroyed[w->id].fetch_add(1) != 0) error(c, EINVAL);
}
bool lock(Context &c, pthread_mutex_t &m) noexcept {
    const int e = pthread_mutex_timedlock(&m, &c.deadline);
    if (e) error(c, e);
    return !e;
}
bool unlock(Context &c, pthread_mutex_t &m) noexcept {
    const int e = pthread_mutex_unlock(&m);
    if (e) { c.synchronization_uncertain = true; error(c, e); }
    return !e;
}
void wake(Context &c) noexcept { const int e = pthread_cond_broadcast(&c.changed); if(e) error(c,e); }
bool wait(Context &c) noexcept {
    const int e = pthread_cond_timedwait(&c.changed, &c.gate, &c.deadline);
    if (e) { error(c, e); c.abort = true; wake(c); }
    return !e;
}
void *worker_main(void *value) noexcept {
    auto &w = *static_cast<Worker *>(value); auto &c = *w.context;
    if (scalar_tls != 17) error(c, EINVAL);
    scalar_tls = 0x100u + w.id;
    const int set = pthread_setspecific(c.key, &w);
    if (set) error(c, set);
    if (!set && pthread_getspecific(c.key) != &w) error(c, EINVAL);
    // Main still owns work until both workers have recorded this real contention.
    const int busy = pthread_mutex_trylock(&c.work);
    if (busy != EBUSY) { error(c, EINVAL); if (!busy) (void)unlock(c, c.work); }
    if (!lock(c, c.gate)) return nullptr;
    ++c.arrived; if (busy == EBUSY) ++c.busy; wake(c);
    while (!c.released && !c.abort) if (!wait(c)) break;
    for (unsigned i = 0; i < iterations && !c.abort; ++i) {
        while (c.turn != w.id && !c.abort) if (!wait(c)) break;
        if (c.abort) break;
        if (scalar_tls != 0x100u + w.id || (!set && pthread_getspecific(c.key) != &w)) error(c, EINVAL);
        if (!lock(c, c.work)) { c.abort = true; wake(c); break; }
        if (c.turns != 2 * i + w.id) error(c, EINVAL);
        ++c.counter; ++c.turns;
        if (!unlock(c, c.work)) { c.abort = true; wake(c); break; }
        c.turn = 1 - w.id; wake(c);
    }
    c.observed_tls[w.id] = scalar_tls;
    (void)unlock(c, c.gate);
    return nullptr; // Real pthread exit invokes the key destructor.
}
}
Report run() noexcept {
    Report r;
    if (invoked.test_and_set()) { r.error = EALREADY; return r; }
    auto &c = state;
    bool gate = false, work = false, cond = false, key = false, attr = false, work_locked = false;
    pthread_attr_t attributes{}; pthread_t threads[2]{};
    auto setup = [&](int e) { if(e) error(c,e); return !e; };
    do {
        if (scalar_tls != 17) { error(c, EINVAL); break; }
        scalar_tls = 29;
        if (!setup(clock_gettime(CLOCK_REALTIME, &c.deadline) ? errno : 0)) break;
        c.deadline.tv_sec += 3;
        if (!(gate = setup(pthread_mutex_init(&c.gate, nullptr)))) break;
        if (!(work = setup(pthread_mutex_init(&c.work, nullptr)))) break;
        if (!(cond = setup(pthread_cond_init(&c.changed, nullptr)))) break;
        if (!(key = setup(pthread_key_create(&c.key, key_destructor)))) break;
        if (!setup(pthread_setspecific(c.key, &c.main_key_value))) break;
        if (pthread_getspecific(c.key) != &c.main_key_value) { error(c, EINVAL); break; }
        if (!(attr = setup(pthread_attr_init(&attributes)))) break;
        if (!setup(pthread_attr_setstacksize(&attributes, worker_stack_bytes))) break;
        std::size_t configured = 0;
        if (!setup(pthread_attr_getstacksize(&attributes, &configured))) break;
        if (configured != worker_stack_bytes) { error(c, EINVAL); break; }
        if (!(work_locked = lock(c, c.work))) break;
        for (unsigned id = 0; id < 2; ++id) {
            c.worker[id] = {&c, id};
            if (!setup(pthread_create(&threads[id], &attributes, worker_main, &c.worker[id]))) break;
            ++r.created;
        }
        if (!lock(c, c.gate)) break;
        if (r.created != 2) c.abort = true;
        while (c.arrived != r.created && !c.abort) if (!wait(c)) break;
        c.released = true;
        if (work_locked) { work_locked = false; (void)unlock(c, c.work); }
        wake(c);
        (void)unlock(c, c.gate);
    } while (false);
    if (work_locked) { work_locked = false; (void)unlock(c, c.work); }
    // Every created worker has bounded waits, including a partial-create worker.
    // A failed join keeps its backing and synchronization objects alive.
    for (unsigned id = 0; id < r.created; ++id) {
        void *value = reinterpret_cast<void *>(1);
        const int e = pthread_join(threads[id], &value);
        if (e) error(c,e);
        else { ++r.joined; if(value) error(c,EINVAL); }
    }
    const bool synchronization_safe = r.created == r.joined && !c.synchronization_uncertain;
    bool closed = synchronization_safe;
    if (attr && pthread_attr_destroy(&attributes)) { error(c,EINVAL); closed = false; }
    if (synchronization_safe) {
        if (scalar_tls != 29 || (key && pthread_getspecific(c.key) != &c.main_key_value)) error(c,EINVAL);
        // Main's key value is intentionally cleared: only worker exit destructors count.
        if (key && pthread_setspecific(c.key, nullptr)) { error(c,EINVAL); closed = false; }
        if (key && pthread_key_delete(c.key)) { error(c,EINVAL); closed = false; }
        if (cond && pthread_cond_destroy(&c.changed)) { error(c,EINVAL); closed = false; }
        if (work && pthread_mutex_destroy(&c.work)) { error(c,EINVAL); closed = false; }
        if (gate && pthread_mutex_destroy(&c.gate)) { error(c,EINVAL); closed = false; }
    }
    if (r.joined == r.created) {
        r.busy=c.busy; r.counter=c.counter; r.turns=c.turns;
        for(unsigned i=0;i<2;++i) { r.destructors[i]=c.destroyed[i]; r.worker_tls[i]=c.observed_tls[i]; }
    }
    r.main_tls=scalar_tls; r.error=c.error; r.cleanup_complete=closed;
    r.passed = !r.error && closed && r.created==2 && r.joined==2 && r.busy==2 &&
        r.counter==2*iterations && r.turns==2*iterations && r.destructors[0]==1 && r.destructors[1]==1 &&
        r.main_tls==29 && r.worker_tls[0]==0x100 && r.worker_tls[1]==0x101;
    if (!r.passed && !r.error) r.error = EINVAL;
    return r;
}
}
