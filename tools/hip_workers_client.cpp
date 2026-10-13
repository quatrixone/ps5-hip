#include "hip_workers_client.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <pthread.h>
#include <time.h>
extern "C" void threads_launch(unsigned*,unsigned,unsigned);
namespace ps5hip_workers {
namespace {
constexpr unsigned words=271,guard=7,poison=0xa5c39e71;
struct Worker {unsigned id=0,pending=0;pthread_t identity{};std::atomic<bool> identified{false};unsigned input[words]{},output[words]{};};
struct State {
    pthread_mutex_t gate{};pthread_cond_t changed{};timespec deadline{};
    std::atomic<int> error{0};std::atomic<unsigned> hip_error{0};
    std::atomic<bool> abort{false},uncertain{false};
    std::atomic<unsigned> dispatches{0},active{0},comparisons{0},templates{0},checks{0};
    unsigned phase=0;unsigned *device=nullptr;Worker workers[2]{},main{};
};
static_assert(std::atomic<unsigned>::is_always_lock_free&&std::atomic<int>::is_always_lock_free&&std::atomic<bool>::is_always_lock_free);
State state;Report last_report;std::atomic_flag invoked=ATOMIC_FLAG_INIT;
void fail(int e) noexcept {int zero=0;state.error.compare_exchange_strong(zero,e?e:EINVAL);state.abort=true;}
bool status(int e) noexcept {if(e)fail(e);return !e;}
bool hip(hipError_t e) noexcept {if(e==hipSuccess)return true;unsigned zero=0;state.hip_error.compare_exchange_strong(zero,unsigned(e));fail(EIO);return false;}
bool check(bool yes) noexcept {if(!yes)fail(EINVAL);return yes;}
bool phase_check(hipError_t e,hipError_t expected=hipSuccess) noexcept {
    if(e!=expected){unsigned zero=0;state.hip_error.compare_exchange_strong(zero,unsigned(e));fail(EINVAL);return false;}
    ++state.checks;return true;
}
bool lock() noexcept {const int e=pthread_mutex_timedlock(&state.gate,&state.deadline);if(e)state.uncertain=true;return status(e);}
bool unlock() noexcept {const int e=pthread_mutex_unlock(&state.gate);if(e){state.uncertain=true;fail(e);}return !e;}
bool broadcast() noexcept {const int e=pthread_cond_broadcast(&state.changed);if(e)state.uncertain=true;return status(e);}
bool publish(unsigned phase) noexcept {
    if(!lock())return false;
    state.phase=phase;
    const bool ok=broadcast();
    return unlock()&&ok;
}
bool await(unsigned phase) noexcept {
    if(!lock())return false;
    while(state.phase<phase&&!state.abort.load()){
        const int e=pthread_cond_timedwait(&state.changed,&state.gate,&state.deadline);
        if(e){fail(e);if(e!=ETIMEDOUT){state.uncertain=true;return false;}break;}
    }
    const bool ready=!state.abort.load()&&state.phase>=phase;
    return unlock()&&ready;
}
void wake_abort() noexcept {
    state.abort=true;
    if(state.uncertain.load())return;
    if(lock()){(void)broadcast();(void)unlock();}
}
bool push(Worker&w,unsigned grid) noexcept {
    const auto e=__hipPushCallConfiguration(dim3(grid),dim3(64),0,nullptr);
    if(e==hipSuccess)++w.pending;
    return phase_check(e);
}
bool pop(Worker&w,unsigned grid) noexcept {
    dim3 g,b;size_t shared=1;hipStream_t stream=reinterpret_cast<hipStream_t>(1);
    const auto e=__hipPopCallConfiguration(&g,&b,&shared,&stream);
    if(e==hipSuccess&&w.pending)--w.pending;
    return phase_check(e)&&check(g.x==grid&&g.y==1&&g.z==1&&b.x==64&&b.y==1&&b.z==1&&!shared&&!stream);
}
void drain(Worker&w) noexcept {
    // No coordination mutex is held here. Failed pop leaves runtime ownership
    // intact for its key destructor/retention rules; never reset a poisoned gate.
    while(w.pending){dim3 g,b;size_t shared;hipStream_t stream;
        if(__hipPopCallConfiguration(&g,&b,&shared,&stream)!=hipSuccess)break;
        --w.pending;
    }
}
bool launch(Worker&w,unsigned index,unsigned count,unsigned salt,const char*actor) noexcept {
    for(auto &v:w.input)v=poison;
    if(!hip(hipMemcpy(state.device,w.input,sizeof(w.input),hipMemcpyHostToDevice)))return false;
    threads_launch(state.device+guard,count,salt);
    if(!hip(hipGetLastError())||!hip(hipDeviceSynchronize())||
       !hip(hipMemcpy(w.output,state.device,sizeof(w.output),hipMemcpyDeviceToHost)))return false;
    for(unsigned i=0;i<words;++i){
        const auto expected=i>=guard&&i<guard+count?unsigned(uint64_t(i-guard)*17+salt):poison;
        if(!check(w.output[i]==expected&&w.input[i]==poison))return false;
    }
    ++state.dispatches;state.active+=count;state.comparisons+=words;state.templates+=words;
    std::printf("WORKERS_STEP index=%u actor=%s count=%u salt=%u exact=true guards=true template=true\n",index,actor,count,salt);
    return true;
}
bool worker_a(Worker&w) noexcept {
    if(!phase_check(hipMalloc(nullptr,4),hipErrorInvalidValue)||!publish(1)||!await(3))return false;
    if(!phase_check(hipPeekAtLastError(),hipErrorInvalidValue)||!phase_check(hipGetLastError(),hipErrorInvalidValue)||!phase_check(hipPeekAtLastError()))return false;
    if(!push(w,2)||!publish(4)||!await(5)||!pop(w,2)||!push(w,7)||!push(w,9)||!pop(w,9)||!pop(w,7)||!publish(6)||!await(7))return false;
    if(!launch(w,1,73,101,"A")||!publish(8)||!await(9))return false;
    return launch(w,3,73,65521,"A")&&publish(10);
}
bool worker_b(Worker&w) noexcept {
    if(!await(2)||!phase_check(hipPeekAtLastError()))return false;
    dim3 g,b;size_t shared;hipStream_t stream;
    if(!phase_check(__hipPopCallConfiguration(&g,&b,&shared,&stream),hipErrorInvalidValue)||!phase_check(hipGetLastError(),hipErrorInvalidValue)||!phase_check(hipPeekAtLastError())||!publish(3)||!await(4))return false;
    if(!push(w,5)||!publish(5)||!await(6)||!pop(w,5)||!publish(7)||!await(8))return false;
    return launch(w,2,257,1009,"B")&&publish(9);
}
void *worker_main(void*p) noexcept {
    auto&w=*static_cast<Worker*>(p);
    w.identity=pthread_self();w.identified.store(true,std::memory_order_release);
    try {if(!(w.id?worker_b(w):worker_a(w)))wake_abort();}
    catch(...){fail(EIO);wake_abort();}
    if(!state.uncertain.load())drain(w);return nullptr;
}
}
const Report &report() noexcept {return last_report;}
unsigned actor() noexcept {
    const auto self=pthread_self();
    if(state.main.identified.load(std::memory_order_acquire)&&pthread_equal(self,state.main.identity))return 0;
    for(unsigned i=0;i<2;++i)if(state.workers[i].identified.load(std::memory_order_acquire)&&pthread_equal(self,state.workers[i].identity))return i+1;
    return 3;
}
Report run() noexcept {
    Report r;
    if(invoked.test_and_set()){r.error=EALREADY;last_report=r;return r;}
    state.main.identity=pthread_self();state.main.identified.store(true,std::memory_order_release);
    bool gate=false,cond=false,attr=false;pthread_attr_t attributes{};pthread_t threads[2]{};
    do {
        const auto before=ps5hip_runtime_observation();
        if(!check(ps5hip_runtime_registration_ready()&&before.modules==1&&before.functions==1&&!before.completed&&!before.backend_attempted))break;
        if(!(gate=status(pthread_mutex_init(&state.gate,nullptr))))break;
        if(!(cond=status(pthread_cond_init(&state.changed,nullptr))))break;
        if(!(attr=status(pthread_attr_init(&attributes))))break;
        if(!status(pthread_attr_setstacksize(&attributes,worker_stack_bytes)))break;
        size_t stack=0;if(!status(pthread_attr_getstacksize(&attributes,&stack))||!check(stack==worker_stack_bytes))break;
        if(!hip(hipMalloc(reinterpret_cast<void**>(&state.device),words*sizeof(unsigned))))break;
        if(!launch(state.main,0,1,11,"main"))break;
        if(!status(clock_gettime(CLOCK_REALTIME,&state.deadline)?errno:0))break;
        state.deadline.tv_sec+=20;
        for(unsigned id=0;id<2;++id){state.workers[id].id=id;
            if(!status(pthread_create(&threads[id],&attributes,worker_main,&state.workers[id])))break;
            ++r.created;
        }
        if(r.created!=2){wake_abort();break;}
        if(!await(1))break;
        const auto warm=ps5hip_runtime_observation();
        if(!check(warm.completed==1&&warm.backend_attempted)||!phase_check(hipPeekAtLastError())||!publish(2))break;
    }while(false);
    if(state.error.load()&&r.created)wake_abort();
    for(unsigned id=0;id<r.created;++id){void*value=reinterpret_cast<void*>(1);
        const int e=pthread_join(threads[id],&value);
        if(e){fail(e);state.uncertain=true;}else{++r.joined;(void)check(!value);}
    }
    const bool joined=r.created==r.joined;
    if(joined&&!state.uncertain.load()&&!state.error.load()){
        if(check(r.created==2&&state.checks==expected_phase_checks))
            (void)launch(state.main,4,257,4294967280u,"main");
    }
    bool closed=joined&&!state.uncertain.load();
    if(closed&&state.device){
        if(ps5hip_runtime_terminal()||ps5hip_runtime_gpu_unresolved())closed=false;
        else if(hip(hipFree(state.device)))state.device=nullptr;else closed=false;
    }
    if(attr&&!status(pthread_attr_destroy(&attributes)))closed=false;
    if(joined&&!state.uncertain.load()){
        if(cond&&!status(pthread_cond_destroy(&state.changed)))closed=false;
        if(gate&&!status(pthread_mutex_destroy(&state.gate)))closed=false;
    }
    r.error=state.error;r.hip_error=state.hip_error;r.dispatches=state.dispatches;r.active_values=state.active;
    r.word_comparisons=state.comparisons;r.template_comparisons=state.templates;r.phase_checks=state.checks;
    r.cleanup_complete=closed;r.unsafe_teardown=!closed;
    r.passed=!r.error&&closed&&r.created==2&&r.joined==2&&r.dispatches==5&&r.active_values==661&&r.word_comparisons==1355&&r.template_comparisons==1355&&r.phase_checks==expected_phase_checks;
    if(!r.passed&&!r.error)r.error=EINVAL;
    last_report=r;return r;
}
}
int ps5hip_workers_client_main(int,char**) {
    const auto r=ps5hip_workers::run();
    if(r.passed){std::printf("WORKERS_CLIENT_PASS created=2 joined=2 dispatches=5 exact_values=661 word_comparisons=1355 template_comparisons=1355 phase_checks=17\n");return 0;}
    std::printf("WORKERS_CLIENT_FAIL error=%d hip_error=%u created=%u joined=%u cleanup_complete=%u unsafe=%u dispatches=%u\n",r.error,r.hip_error,r.created,r.joined,unsigned(r.cleanup_complete),unsigned(r.unsafe_teardown),r.dispatches);return 1;
}
