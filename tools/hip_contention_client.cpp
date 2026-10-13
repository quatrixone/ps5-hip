#include "hip_contention.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <pthread.h>
extern "C" void threads_launch(unsigned*,unsigned,unsigned);
namespace ps5hip_contention {
namespace {
constexpr unsigned count_words=271,guard=7,poison=0xa5c39e71;
struct Worker {unsigned id=0;pthread_t identity{};std::atomic<bool> identified{false};unsigned *device=nullptr;unsigned input[count_words]{},output[count_words]{};unsigned dispatches=0,values=0;};
Worker workers[2],main_actor;
Report last_report;
std::atomic<unsigned> hip_error{0};
std::atomic_flag invoked=ATOMIC_FLAG_INIT;
bool status(int e) noexcept {if(e)fail(e);return e==0;}
bool check(bool b) noexcept {if(!b)fail(EINVAL);return b;}
bool hip(hipError_t e) noexcept {if(e==hipSuccess)return true;unsigned zero=0;hip_error.compare_exchange_strong(zero,unsigned(e));fail(EIO);return false;}
bool launch(Worker&w,unsigned n,unsigned salt) noexcept {
    threads_launch(w.device+guard,n,salt);
    if(!hip(hipGetLastError())||!hip(hipDeviceSynchronize())||!hip(hipMemcpy(w.output,w.device,sizeof(w.output),hipMemcpyDeviceToHost)))return false;
    for(unsigned i=0;i<count_words;++i){
        const unsigned expected=i>=guard&&i<guard+n?unsigned(uint64_t(i-guard)*17+salt):poison;
        if(!check(w.output[i]==expected&&w.input[i]==poison))return false;
    }
    ++w.dispatches;w.values+=n;return true;
}
void *worker(void*p) noexcept {
    auto&w=*static_cast<Worker*>(p);w.identity=pthread_self();w.identified.store(true,std::memory_order_release);
    if(w.id==1||await_hold())(void)launch(w,w.id==1?73:257,w.id==1?101:1009);
    return nullptr;
}
}
unsigned actor() noexcept {
    const auto self=pthread_self();
    if(main_actor.identified.load(std::memory_order_acquire)&&pthread_equal(self,main_actor.identity))return 0;
    for(auto&w:workers)if(w.identified.load(std::memory_order_acquire)&&pthread_equal(self,w.identity))return w.id;
    return 3;
}
const Report &report() noexcept {return last_report;}
Report run() noexcept {
    Report r;
    if(invoked.test_and_set()){r.error=EALREADY;return last_report=r;}
    main_actor.identity=pthread_self();main_actor.identified.store(true,std::memory_order_release);
    pthread_attr_t attributes{};bool attr=false,uncertain=false;pthread_t threads[2]{};
    do {
        const auto before=ps5hip_runtime_observation();
        if(!check(ps5hip_runtime_registration_ready()&&before.modules==1&&before.functions==1&&!before.completed&&!before.backend_attempted))break;
        if(!(attr=status(pthread_attr_init(&attributes)))||!status(pthread_attr_setstacksize(&attributes,512*1024)))break;
        size_t stack=0;if(!status(pthread_attr_getstacksize(&attributes,&stack))||!check(stack==512*1024))break;
        for(unsigned i=0;i<2;++i){auto&w=workers[i];w.id=i+1;for(auto&v:w.input)v=poison;
            if(!hip(hipMalloc(reinterpret_cast<void**>(&w.device),sizeof(w.input)))||!hip(hipMemcpy(w.device,w.input,sizeof(w.input),hipMemcpyHostToDevice)))break;
        }
        if(error())break;
        main_actor.device=workers[0].device;for(auto&v:main_actor.input)v=poison;
        if(!launch(main_actor,1,11))break;
        // Establish independent poisoned buffers before either worker enters HIP.
        for(auto&w:workers)if(!hip(hipMemcpy(w.device,w.input,sizeof(w.input),hipMemcpyHostToDevice)))break;
        if(error())break;
        for(unsigned i=0;i<2;++i){if(!status(pthread_create(&threads[i],&attributes,worker,&workers[i])))break;++r.created;}
    }while(false);
    for(unsigned i=0;i<r.created;++i){void *result=reinterpret_cast<void*>(1);const int e=pthread_join(threads[i],&result);
        if(e){fail(e);uncertain=true;}else{++r.joined;(void)check(result==nullptr);}
    }
    // Unknown joins permit only atomic diagnostics. Worker storage and buffers
    // remain alive; the title must enter its no-runtime-query quarantine.
    bool closed=!uncertain&&r.created==r.joined;
    if(closed){
        r.dispatches=main_actor.dispatches;r.values=main_actor.values;
        for(auto&w:workers){r.dispatches+=w.dispatches;r.values+=w.values;}
        r.words=r.templates=r.dispatches*count_words;
        if(!error())(void)evidence(false);
        if(ps5hip_runtime_terminal()||ps5hip_runtime_gpu_unresolved())closed=false;
        else for(auto&w:workers)if(w.device){if(hip(hipFree(w.device)))w.device=nullptr;else closed=false;}
    }
    if(attr&&!status(pthread_attr_destroy(&attributes)))closed=false;
    r.error=error();r.hip_error=hip_error;r.cleanup_complete=closed;r.unsafe_teardown=!closed;
    r.passed=!r.error&&closed&&r.created==2&&r.joined==2&&r.dispatches==3&&r.values==331&&r.words==813&&r.templates==813;
    if(!r.passed&&!r.error)r.error=EINVAL;
    return last_report=r;
}
}
int ps5hip_contention_client_main(int,char**){
    const auto r=ps5hip_contention::run();
    if(r.passed){std::puts("CONTEND_CLIENT_PASS created=2 joined=2 dispatches=3 exact_values=331 word_comparisons=813 template_comparisons=813");return 0;}
    std::printf("CONTEND_CLIENT_FAIL error=%d hip_error=%u created=%u joined=%u cleanup_complete=%u unsafe=%u dispatches=%u\n",r.error,r.hip_error,r.created,r.joined,unsigned(r.cleanup_complete),unsigned(r.unsafe_teardown),r.dispatches);return 1;
}
