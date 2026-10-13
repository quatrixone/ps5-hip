// Actual client/interposers; synthetic HIP/AGC, no runtime or device claim.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include "../tools/hip_contention.h"
#include <ps5platform/agc.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>
extern "C" hipError_t __wrap_hipLaunchKernel(const void*,dim3,dim3,void**,size_t,hipStream_t);
extern "C" hipError_t __wrap___hipPushCallConfiguration(dim3,dim3,size_t,hipStream_t);
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t*);
extern "C" int32_t __wrap_sceAgcInit(uint32_t);
extern "C" int32_t __wrap_sceAgcDriverSubmitDcb(ps5_agc_submit_description*);
extern "C" int32_t __wrap_sceAgcSuspendPoint();
namespace {
pthread_mutex_t gate=PTHREAD_MUTEX_INITIALIZER;
thread_local hipError_t last=hipSuccess;
unsigned *buffers[2]{},completed=0;const char*mode="normal";
std::atomic<unsigned> creates{0},joins{0};
void check(bool b){if(!b)std::abort();}
struct Payload{unsigned*p,n,salt;}payload{};
}
extern "C" int __real_pthread_mutex_lock(pthread_mutex_t*m){return pthread_mutex_lock(m);}
extern "C" hipError_t hipMalloc(void**p,size_t n){check(n==271*4);*p=std::malloc(n);check(*p);for(auto&b:buffers)if(!b){b=static_cast<unsigned*>(*p);return hipSuccess;}std::abort();}
extern "C" hipError_t hipFree(void*p){for(auto&b:buffers)if(b==p){std::free(b);b=nullptr;return hipSuccess;}std::abort();}
extern "C" hipError_t hipMemcpy(void*d,const void*s,size_t n,hipMemcpyKind){std::memcpy(d,s,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return hipSuccess;}
extern "C" hipError_t hipGetLastError(){const auto r=last;last=hipSuccess;return r;}
extern "C" hipError_t __real___hipPushCallConfiguration(dim3,dim3,size_t,hipStream_t){check(__wrap_pthread_mutex_lock(&gate)==0);check(pthread_mutex_unlock(&gate)==0);return hipSuccess;}
extern "C" int32_t __real_sceAgcInit(uint32_t v){check(v==8);return 0;}
extern "C" int32_t __real_sceAgcDriverSubmitDcb(ps5_agc_submit_description*p){check(p==reinterpret_cast<ps5_agc_submit_description*>(&payload));for(unsigned i=0;i<payload.n;++i)payload.p[i]=i*17u+payload.salt;return 0;}
extern "C" int32_t __real_sceAgcSuspendPoint(){return 0;}
extern "C" hipError_t __real_hipLaunchKernel(const void*,dim3,dim3,void**args,size_t,hipStream_t){
 check(__wrap_pthread_mutex_lock(&gate)==0);
 if(!completed)check(__wrap_sceAgcInit(8)==0);
 payload=*static_cast<Payload*>(args[0]);check(__wrap_sceAgcDriverSubmitDcb(reinterpret_cast<ps5_agc_submit_description*>(&payload))==0);check(__wrap_sceAgcSuspendPoint()==0);++completed;
 check(pthread_mutex_unlock(&gate)==0);return hipSuccess;
}
extern "C" void threads_launch(unsigned*p,unsigned n,unsigned salt){
 if(!std::strcmp(mode,"skipped-push")&&ps5hip_contention::actor()==2)return;
 last=__wrap___hipPushCallConfiguration(dim3(5),dim3(64),0,nullptr);if(last)return;
 Payload args{p,n,salt};void*arguments[]={&args};last=__wrap_hipLaunchKernel(nullptr,dim3(5),dim3(64),arguments,0,nullptr);
}
extern "C" bool ps5hip_runtime_registration_ready()noexcept{return true;}
extern "C" bool ps5hip_runtime_terminal(){return false;}
extern "C" bool ps5hip_runtime_gpu_unresolved(){return false;}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation()noexcept{return {1,1,completed,completed!=0};}
extern "C" int client_pthread_create(pthread_t*t,const pthread_attr_t*a,void*(*f)(void*),void*p)noexcept{if(++creates==2&&!std::strcmp(mode,"create-second"))return EAGAIN;return pthread_create(t,a,f,p);}
extern "C" int client_pthread_join(pthread_t t,void**p){const auto n=++joins;const int e=pthread_join(t,p);return n==1&&!std::strcmp(mode,"join")?EINVAL:e;}
int main(int argc,char**argv){check(argc==2);mode=argv[1];const int result=ps5hip_contention_client_main(0,nullptr);const auto&r=ps5hip_contention::report();
 if(!std::strcmp(mode,"normal")){check(!result&&r.passed&&ps5hip_contention::evidence(true)&&!buffers[0]&&!buffers[1]);}
 else check(result&&!r.passed&&r.error);
 check(joins==r.created);for(auto*b:buffers)std::free(b);
 std::printf("CONTEND_TSAN_PASS mode=%s synthetic_hip=true\n",mode);
}
