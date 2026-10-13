#define main prior_thread_model_main
#define ps5hip_runtime_model_backend original_contention_backend
#include "hip_threads_model.cpp"
#undef ps5hip_runtime_model_backend
#undef main
#include "../tools/hip_contention.h"
#include <ps5platform/agc.h>
#include <unistd.h>
extern "C" int32_t __wrap_sceAgcInit(uint32_t);
extern "C" int32_t __wrap_sceAgcDriverSubmitDcb(ps5_agc_submit_description*);
extern "C" int32_t __wrap_sceAgcSuspendPoint();
extern "C" int __real_pthread_mutex_lock(pthread_mutex_t*);
namespace {
const char *scenario="normal";unsigned creates=0,joins=0;
std::atomic<unsigned> actor_locks[4]{};
Backend *transport=nullptr;uint32_t *packet=nullptr;uint32_t packet_words=0;
bool is(const char*s){return std::strcmp(scenario,s)==0;}
class TracedBackend final:public ps5hip::SubmissionBackend {
 Backend inner;
 public:
 TracedBackend(){transport=&inner;check(__wrap_sceAgcInit(8)==0);}
 void flush(void*p,size_t n)override{inner.flush(p,n);}
 int32_t submit(uint32_t*p,uint32_t n)override{packet=p;packet_words=n;return __wrap_sceAgcDriverSubmitDcb(reinterpret_cast<ps5_agc_submit_description*>(p));}
 int32_t kick()override{
  if(is("missing-agc")&&ps5hip_contention::actor()==1)return inner.kick();
  const auto result=__wrap_sceAgcSuspendPoint();
  if(is("duplicate-agc")&&ps5hip_contention::actor()==1)check(__wrap_sceAgcSuspendPoint()==0);
  return result;
 }
 uint64_t now_ns()override{return inner.now_ns();}
 void sleep_us(unsigned n)override{inner.sleep_us(n);}
};
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;return std::make_unique<TracedBackend>();}
extern "C" int32_t __real_sceAgcInit(uint32_t v){check(v==8);return 0;}
extern "C" int32_t __real_sceAgcDriverSubmitDcb(ps5_agc_submit_description*p){check(reinterpret_cast<uint32_t*>(p)==packet);return transport->submit(packet,packet_words);}
extern "C" int32_t __real_sceAgcSuspendPoint(){return transport->kick();}
extern "C" int model_lock(pthread_mutex_t*m)noexcept{
 const auto who=ps5hip_contention::actor();const auto n=++actor_locks[who];
 if((is("a-lock")&&who==1&&n==3)||(is("b-lock")&&who==2&&n==1))return EINVAL;
 return __real_pthread_mutex_lock(m);
}
extern "C" int trace_clock(clockid_t c,timespec*t)noexcept{
 if(is("clock")&&ps5hip_contention::actor()==1){errno=EIO;return -1;}return clock_gettime(c,t);
}
extern "C" int trace_sleep(useconds_t n){
 if(is("sleep")&&ps5hip_contention::actor()==1){errno=EIO;return -1;}return usleep(n);
}
extern "C" int client_pthread_create(pthread_t*t,const pthread_attr_t*a,void*(*f)(void*),void*p)noexcept{
 ++creates;if((is("create-first")&&creates==1)||(is("create-second")&&creates==2))return EAGAIN;return pthread_create(t,a,f,p);
}
extern "C" int client_pthread_join(pthread_t t,void**p){++joins;const auto e=pthread_join(t,p);return is("join")&&joins==1?EINVAL:e;}
extern "C" void client_threads_launch(unsigned*p,unsigned n,unsigned salt){
 if(is("skipped-push")&&ps5hip_contention::actor()==2)return;
 threads_launch(p,n,salt);if(is("guard")&&ps5hip_contention::actor()==2)p[-1]^=1;
}
int main(int argc,char**argv){
 check(argc==2);scenario=argv[1];mode=scenario;
 const int result=ps5hip_contention_client_main(0,nullptr);const auto&r=ps5hip_contention::report();
 check(joins==r.created);
 if(is("normal")){
  check(!result&&r.passed&&r.cleanup_complete&&!r.unsafe_teardown&&r.dispatches==3&&r.values==331&&r.words==813&&r.templates==813&&submits==3&&allocations.empty());
  check(ps5hip_contention::evidence(true));ps5hip_runtime_finish();check(ps5hip_runtime_ready());
 }else{
  expect_final=false;check(result==1&&!r.passed&&r.error);
  if(is("join"))check(r.unsafe_teardown&&allocations.size()==2);
  if(is("a-lock")||is("b-lock"))check(r.unsafe_teardown&&!r.cleanup_complete);
  // These transport injections occur in the main warmup, before either worker
  // exists. Retention must match the precise GPU ownership state, not any leak.
  if(is("submit")||is("kick")||is("timeout")||is("postflush")||is("cleanup")){
   const bool cleanup=is("cleanup");
   check(!r.created&&!r.joined&&!r.dispatches&&submits==1&&backends==1&&r.unsafe_teardown&&!r.cleanup_complete);
   check(ps5hip_runtime_terminal()&&ps5hip_runtime_gpu_unresolved()!=cleanup);
   check(allocations.size()==size_t(cleanup?3:6)&&releases==unsigned(cleanup?3:0));
   std::printf("CONTEND_WARMUP_OWNERS terminal=1 unresolved=%u retained=%zu releases=%u workers_created=0\n",unsigned(!cleanup),allocations.size(),releases);
  }
 }
 std::printf("CONTEND_MODEL_PASS mode=%s created=%u joined=%u submits=%u retained=%zu model_only=true\n",scenario,r.created,r.joined,submits,allocations.size());return 0;
}
