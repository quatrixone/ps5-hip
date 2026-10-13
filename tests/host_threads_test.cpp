#include "host_threads.h"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

namespace {
const char *mode="normal";
std::atomic<unsigned> creates{0}, joins{0}, sets{0}, waits{0}, destroys{0}, cond_destroys{0}, key_deletes{0};
pthread_t main_thread{};
std::atomic<bool> unlock_injected{false};
void (*original_destructor)(void *)=nullptr;
bool is(const char *s) {return std::strcmp(mode,s)==0;}
void mutated_destructor(void *p) {
    if(is("missing-destructor")) return;
    original_destructor(p);
    if(is("duplicate-destructor")) original_destructor(p);
}
}
extern "C" {
int ps5hip_test_pthread_create(pthread_t*t,const pthread_attr_t*a,void*(*f)(void*),void*v) {
    unsigned n=++creates;
    if((is("create-first")&&n==1)||(is("create-second")&&n==2)) return EAGAIN;
    return pthread_create(t,a,f,v);
}
int ps5hip_test_pthread_join(pthread_t t,void**v) {
    ++joins;
    // Complete the real join first so this subprocess does not leak a test worker.
    const int e=pthread_join(t,v);
    return is("join-error")?EINVAL:e;
}
int ps5hip_test_pthread_key_create(pthread_key_t*k,void(*f)(void*)) {
    if(is("key-create"))return EAGAIN;
    original_destructor=f;return pthread_key_create(k,mutated_destructor);
}
int ps5hip_test_pthread_setspecific(pthread_key_t k,const void*v) {
    unsigned n=++sets;
    if((is("main-key")&&n==1)||(is("worker-key")&&n==2))return ENOMEM;
    return pthread_setspecific(k,v);
}
int ps5hip_test_pthread_attr_setstacksize(pthread_attr_t*a,size_t n) {
    if(is("stack-attr"))return EINVAL;
    if(n!=ps5hip_threads::worker_stack_bytes)std::abort();
    return pthread_attr_setstacksize(a,n);
}
int ps5hip_test_pthread_cond_timedwait(pthread_cond_t*c,pthread_mutex_t*m,const timespec*t) {
    ++waits;
    if(is("wait-timeout"))return ETIMEDOUT;
    return pthread_cond_timedwait(c,m,t);
}
int ps5hip_test_pthread_mutex_trylock(pthread_mutex_t*m) {
    if(is("busy-missing"))return EINVAL;
    return pthread_mutex_trylock(m);
}
int ps5hip_test_pthread_mutex_unlock(pthread_mutex_t*m) {
    const bool main=pthread_equal(pthread_self(),main_thread);
    const bool inject=(is("main-unlock")&&main)||(is("worker-unlock")&&!main);
    const int e=pthread_mutex_unlock(m);
    // Release the actual mutex, then report uncertainty; do not strand the test.
    if(inject && !unlock_injected.exchange(true))return EPERM;
    return e;
}
int ps5hip_test_pthread_mutex_destroy(pthread_mutex_t*m) {
    ++destroys;
    const int e=pthread_mutex_destroy(m);
    return is("mutex-close")?EBUSY:e;
}
int ps5hip_test_pthread_cond_destroy(pthread_cond_t*c) {
    ++cond_destroys; const int e=pthread_cond_destroy(c);return is("cond-close")?EBUSY:e;
}
int ps5hip_test_pthread_key_delete(pthread_key_t k) {
    ++key_deletes; const int e=pthread_key_delete(k);return is("key-close")?EBUSY:e;
}
}
extern "C" void ps5hip_threads_test_configure(const char *value) {mode=value;main_thread=pthread_self();}
#ifndef PS5HIP_THREADS_TITLE_TEST
int main(int argc,char**argv) {
    if(argc!=2)return 2;ps5hip_threads_test_configure(argv[1]);
    const auto r=ps5hip_threads::run();
    std::printf("THREAD_MODEL mode=%s passed=%u clean=%u error=%d created=%u joined=%u busy=%u counter=%u turns=%u destructors=%u,%u tls=%u,%u,%u attempts=%u,%u waits=%u closes=%u,%u,%u actual_pthreads=true\n",mode,unsigned(r.passed),unsigned(r.cleanup_complete),r.error,r.created,r.joined,r.busy,r.counter,r.turns,r.destructors[0],r.destructors[1],r.main_tls,r.worker_tls[0],r.worker_tls[1],unsigned(creates),unsigned(joins),unsigned(waits),unsigned(destroys),unsigned(cond_destroys),unsigned(key_deletes));
    if(is("normal")||is("repeat")) {
        if(!r.passed||!r.cleanup_complete||r.error||r.counter!=256||r.turns!=256||r.destructors[0]!=1||r.destructors[1]!=1||r.busy!=2||r.main_tls!=29||r.worker_tls[0]!=256||r.worker_tls[1]!=257)return 3;
        if(is("repeat")){auto again=ps5hip_threads::run();if(again.passed||again.error!=EALREADY||creates!=2)return 4;}
    } else {
        if(r.passed||!r.error)return 5;
        const bool uncertain=is("join-error")||is("mutex-close")||is("cond-close")||is("key-close")||is("main-unlock")||is("worker-unlock");
        if(r.cleanup_complete==uncertain)return 6;
        if(!is("join-error")&&r.created!=r.joined)return 7;
        if(is("create-second")&&(r.created!=1||r.joined!=1||r.destructors[0]!=1||r.destructors[1]!=0))return 8;
        if(joins!=r.created)return 9;
        if(is("join-error")&&(destroys||cond_destroys||key_deletes||r.joined))return 10;
        if((is("main-unlock")||is("worker-unlock"))&&(!unlock_injected||destroys||cond_destroys||key_deletes||r.joined!=2))return 11;
    }
    return 0;
}

#endif
