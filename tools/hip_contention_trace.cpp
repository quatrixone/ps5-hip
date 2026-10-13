#include "hip_contention.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <ps5platform/agc.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <type_traits>
extern "C" {
decltype(hipLaunchKernel) __real_hipLaunchKernel;
decltype(__hipPushCallConfiguration) __real___hipPushCallConfiguration;
decltype(pthread_mutex_lock) __real_pthread_mutex_lock;
decltype(sceAgcInit) __real_sceAgcInit;
decltype(sceAgcDriverSubmitDcb) __real_sceAgcDriverSubmitDcb;
decltype(sceAgcSuspendPoint) __real_sceAgcSuspendPoint;
}
namespace ps5hip_contention {
namespace {
struct Api {
    std::atomic<bool> active{false};
    std::atomic<unsigned> calls{0},locks{0};
    std::atomic<int> result{0},lock_result{0};
    std::atomic<uint64_t> entered{0},request{0},acquired{0},exited{0};
};
struct Agc {std::atomic<unsigned> count{0},order{0};std::atomic<int> result{0};};
Api a,b;Agc agc[7];std::atomic<unsigned> ordinal{0},collision{0};
std::atomic<int> first_error{0};std::atomic<bool> holding{false};
std::atomic<uint64_t> held_at{0},released_at{0};
std::atomic<pthread_mutex_t*> runtime_gate{nullptr};
static_assert(std::atomic<uint64_t>::is_always_lock_free&&std::atomic<pthread_mutex_t*>::is_always_lock_free&&std::atomic<unsigned>::is_always_lock_free);
uint64_t now() noexcept {
    timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t)||t.tv_sec<0){fail(EIO);return 0;}
    return uint64_t(t.tv_sec)*1000000000u+uint64_t(t.tv_nsec);
}
bool wait(bool for_hold) noexcept {
    const auto start=now();
    for(unsigned i=0;i<3000&&!error();++i){
        if(for_hold?holding.load():collision.load()==1)return true;
        const auto current=now();if(current<start||current-start>=2000000000u)break;
        if(usleep(1000)){fail(errno);break;}
    }
    fail(ETIMEDOUT);return false;
}
void begin(Api &row) noexcept {if(row.calls.fetch_add(1)!=0)fail(EINVAL);row.entered=now();row.active=true;}
void end(Api &row,int result) noexcept {row.result=result;row.exited=now();row.active=false;if(result)fail(EIO);}
__attribute__((noinline)) bool before_api(Api &row,unsigned who) noexcept {
    const bool selected=actor()==who;if(selected)begin(row);return selected;
}
__attribute__((noinline)) void after_api(Api &row,bool selected,int result) noexcept {
    if(selected)end(row,result);
}
void agc_result(unsigned operation,unsigned who,int result) noexcept {
    unsigned slot=7;
    if(who==0)slot=operation;
    else if(who<=2&&operation>0)slot=1+who*2+operation-1;
    if(slot>=7){fail(EINVAL);return;}
    auto &row=agc[slot];if(row.count.fetch_add(1)!=0)fail(EINVAL);
    row.result=result;row.order=ordinal.fetch_add(1);if(result)fail(EIO);
}
// Keep the optional hold outside the forwarding body, so optimized images have
// one auditable original-submit call site as well as one call per invocation.
__attribute__((noinline)) void before_submit(unsigned who) noexcept {
    if(who==1){
        if(!a.active||a.locks!=1||a.lock_result!=0||!runtime_gate)fail(EINVAL);
        held_at=now();holding=true;(void)wait(false);released_at=now();holding=false;
    }
}
struct LockAttempt {bool first_a,first_b;};
__attribute__((noinline)) LockAttempt before_lock(pthread_mutex_t*m) noexcept {
    const unsigned who=actor();
    const bool first_a=who==1&&a.active&&a.locks==0;
    const bool first_b=who==2&&b.active&&b.locks==0;
    if(first_a)runtime_gate=m;
    if(first_b){
        b.request=now();
        if(runtime_gate!=m||!holding||!a.active||a.locks!=1||a.lock_result!=0)fail(EINVAL);
        else if(collision.fetch_add(1)!=0)fail(EINVAL);
    }
    return {first_a,first_b};
}
__attribute__((noinline)) void after_lock(LockAttempt attempt,int result) noexcept {
    if(attempt.first_a){a.lock_result=result;++a.locks;a.acquired=now();}
    if(attempt.first_b){b.lock_result=result;++b.locks;b.acquired=now();}
    if((attempt.first_a||attempt.first_b)&&result)fail(result);
}
}
void fail(int e) noexcept {int zero=0;first_error.compare_exchange_strong(zero,e?e:EINVAL);}
int error() noexcept {return first_error.load();}
bool await_hold() noexcept {return wait(true);}
bool evidence(bool print) noexcept {
    bool valid=!error()&&collision==1&&a.calls==1&&b.calls==1&&a.locks==1&&b.locks==1&&
        !a.active&&!b.active&&a.result==0&&b.result==0&&a.lock_result==0&&b.lock_result==0&&
        a.entered>0&&a.entered<=held_at&&held_at<=b.entered&&b.entered<=b.request&&
        b.request<=released_at&&released_at<=b.acquired&&b.acquired<=b.exited&&released_at<=a.exited&&ordinal==7;
    for(unsigned i=0;i<7;++i)valid=valid&&agc[i].count==1&&agc[i].order==i&&agc[i].result==0;
    if(!valid){fail(EINVAL);return false;}
    if(print){
        std::puts("CONTEND_PROOF api_overlap=1 same_gate_request_while_owned=1 a_launch_calls=1 b_push_calls=1 gate_acquired_a=1 gate_acquired_b=1 kernel_blocked_wait=unproven");
        std::printf("CONTEND_TIME hold_ns=%llu request_ns=%llu release_ns=%llu acquired_ns=%llu lock_call_ns=%llu includes_scheduling=true\n",
            (unsigned long long)held_at.load(),(unsigned long long)b.request.load(),(unsigned long long)released_at.load(),(unsigned long long)b.acquired.load(),(unsigned long long)(b.acquired.load()-b.request.load()));
        for(unsigned i=0;i<7;++i)std::printf("CONTEND_AGC ordinal=%u operation=%s actor=%u result=0x00000000\n",i,i==0?"init":i%2?"submit":"kick",i<3?0:i<5?1:2);
    }
    return true;
}
}
extern "C" hipError_t __wrap_hipLaunchKernel(const void*f,dim3 grid,dim3 block,void**args,size_t shared,hipStream_t stream){
    using namespace ps5hip_contention;const bool selected=before_api(a,1);
    const auto result=__real_hipLaunchKernel(f,grid,block,args,shared,stream);after_api(a,selected,int(result));return result;
}
extern "C" hipError_t __wrap___hipPushCallConfiguration(dim3 grid,dim3 block,size_t shared,hipStream_t stream){
    using namespace ps5hip_contention;const bool selected=before_api(b,2);
    const auto result=__real___hipPushCallConfiguration(grid,block,shared,stream);after_api(b,selected,int(result));return result;
}
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t*m) noexcept(noexcept(pthread_mutex_lock(m))){
    using namespace ps5hip_contention;const auto attempt=before_lock(m);
    const int result=__real_pthread_mutex_lock(m);
    after_lock(attempt,result);
    return result;
}
extern "C" int32_t __wrap_sceAgcInit(uint32_t version){
    const auto who=ps5hip_contention::actor();const auto result=__real_sceAgcInit(version);ps5hip_contention::agc_result(0,who,result);return result;
}
extern "C" int32_t __wrap_sceAgcDriverSubmitDcb(ps5_agc_submit_description*p){
    using namespace ps5hip_contention;const auto who=actor();
    before_submit(who);
    const auto result=__real_sceAgcDriverSubmitDcb(p);agc_result(1,who,result);return result;
}
extern "C" int32_t __wrap_sceAgcSuspendPoint(){
    const auto who=ps5hip_contention::actor();const auto result=__real_sceAgcSuspendPoint();ps5hip_contention::agc_result(2,who,result);return result;
}
static_assert(std::is_same_v<decltype(&__wrap_hipLaunchKernel),decltype(&hipLaunchKernel)>);
static_assert(std::is_same_v<decltype(&__wrap___hipPushCallConfiguration),decltype(&__hipPushCallConfiguration)>);
static_assert(std::is_same_v<decltype(&__wrap_pthread_mutex_lock),decltype(&pthread_mutex_lock)>);
static_assert(std::is_same_v<decltype(&__wrap_sceAgcInit),decltype(&sceAgcInit)>);
static_assert(std::is_same_v<decltype(&__wrap_sceAgcDriverSubmitDcb),decltype(&sceAgcDriverSubmitDcb)>);
static_assert(std::is_same_v<decltype(&__wrap_sceAgcSuspendPoint),decltype(&sceAgcSuspendPoint)>);
