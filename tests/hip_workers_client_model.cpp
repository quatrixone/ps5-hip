// Reuse the actual packet/memory transport model, not a fake HIP implementation.
#define main prior_thread_model_main
#define ps5hip_runtime_model_backend original_worker_model_backend
#include "hip_threads_model.cpp"
#undef ps5hip_runtime_model_backend
#undef main
#include "../tools/hip_workers_client.h"
#include <string>
namespace {
const char*scenario="normal";unsigned creates=0,joins=0,mutex_destroyed=0,cond_destroyed=0;
std::atomic<unsigned> unlocks{0},waits{0},broadcasts{0};
std::atomic<unsigned> backend_actor{3},prebackend_checks{0};
bool is(const char*s){return std::strcmp(scenario,s)==0;}
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){
 backend_actor=ps5hip_workers::actor();
 check(backend_actor==(ps5hip_workers::worker_initializes?1u:0u));
 if(ps5hip_workers::worker_initializes)check(prebackend_checks==1);
 return original_worker_model_backend();
}
extern "C" ps5hip::HipRuntimeObservation client_ps5hip_runtime_observation()noexcept{
 auto observation=ps5hip_runtime_observation();
 if(ps5hip_workers::worker_initializes&&ps5hip_workers::actor()==1){
  ++prebackend_checks;check(backends==0&&submits==0&&!observation.backend_attempted&&!observation.completed);
  if(is("premature-backend"))observation.backend_attempted=true;
 }
 return observation;
}
extern "C" int client_pthread_create(pthread_t*t,const pthread_attr_t*a,void*(*f)(void*),void*p)noexcept{
 ++creates;if((is("create-first")&&creates==1)||(is("create-second")&&creates==2))return EAGAIN;
 return pthread_create(t,a,f,p);
}
extern "C" int client_pthread_join(pthread_t t,void**p){++joins;const int r=pthread_join(t,p);return is("join")&&joins==1?EINVAL:r;}
extern "C" int client_pthread_mutex_unlock(pthread_mutex_t*m)noexcept{const int r=pthread_mutex_unlock(m);return is("unlock")&&unlocks++==0?EINVAL:r;}
extern "C" int client_pthread_cond_timedwait(pthread_cond_t*c,pthread_mutex_t*m,const timespec*t){
 if(is("wait")&&waits++==0)return ETIMEDOUT;
 if(is("wait-unknown")&&waits++==0){check(pthread_mutex_unlock(m)==0);return EINVAL;}
 if(is("unlock")||is("wait-unknown")||is("broadcast")){timespec short_deadline{};check(clock_gettime(CLOCK_REALTIME,&short_deadline)==0);short_deadline.tv_sec+=1;return pthread_cond_timedwait(c,m,&short_deadline);}
 return pthread_cond_timedwait(c,m,t);
}
extern "C" int client_pthread_cond_broadcast(pthread_cond_t*c)noexcept{const int e=pthread_cond_broadcast(c);return is("broadcast")&&broadcasts++==0?EINVAL:e;}
extern "C" int client_pthread_mutex_destroy(pthread_mutex_t*m)noexcept{++mutex_destroyed;return pthread_mutex_destroy(m);}
extern "C" int client_pthread_cond_destroy(pthread_cond_t*c)noexcept{++cond_destroyed;const int e=pthread_cond_destroy(c);return is("destroy")?EINVAL:e;}
extern "C" int client_pthread_attr_setstacksize(pthread_attr_t*a,size_t n)noexcept{check(n==512*1024);return is("stack")?EINVAL:pthread_attr_setstacksize(a,n);}
extern "C" hipError_t client_hipMemcpy(void*dst,const void*src,size_t bytes,hipMemcpyKind kind){
 const auto e=hipMemcpy(dst,src,bytes,kind);
 if(e==hipSuccess&&is("template")&&kind==hipMemcpyHostToDevice)static_cast<unsigned*>(const_cast<void*>(src))[0]^=1;
 return e;
}
extern "C" hipError_t client_hipFree(void*p){return is("free")?hipErrorUnknown:hipFree(p);}
extern "C" void client_threads_launch(unsigned*p,unsigned n,unsigned salt){
 static unsigned index=0;const unsigned expected_actor[]={0,1,2,1,0};
 check(index<ps5hip_workers::expected_dispatches&&ps5hip_workers::actor()==expected_actor[index++ + (ps5hip_workers::worker_initializes?1u:0u)]);threads_launch(p,n,salt);
 if(hipPeekAtLastError()==hipSuccess&&is("guard"))p[-1]^=1;
}
int main(int argc,char**argv){
 check(argc==2);scenario=argv[1];mode=scenario;
 const int result=ps5hip_workers_client_main(0,nullptr);const auto&r=ps5hip_workers::report();
 check(r.created==joins); // A failed join is still attempted once; all real model workers joined.
 if(is("normal")){
  check(result==0&&r.passed&&r.cleanup_complete&&!r.unsafe_teardown&&r.created==2&&r.joined==2&&r.dispatches==ps5hip_workers::expected_dispatches&&r.active_values==ps5hip_workers::expected_values&&r.word_comparisons==ps5hip_workers::expected_words&&r.template_comparisons==ps5hip_workers::expected_words&&r.phase_checks==17);
  check(r.prebackend_verified==ps5hip_workers::worker_initializes&&prebackend_checks==unsigned(ps5hip_workers::worker_initializes));
  check(allocations.empty()&&backends==1&&submits==ps5hip_workers::expected_dispatches&&mutex_destroyed==1&&cond_destroyed==1);ps5hip_runtime_finish();check(ps5hip_runtime_ready());
 }else{
  expect_final=false;check(result==1&&!r.passed&&r.error);
  if(is("join")||is("unlock")||is("wait-unknown")||is("broadcast")){check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==1&&mutex_destroyed==0&&cond_destroyed==0);}
  else if(is("free"))check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==1&&submits==ps5hip_workers::expected_dispatches);
  else if(is("destroy"))check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.empty()&&submits==ps5hip_workers::expected_dispatches);
  else if(is("submit")||is("kick")||is("timeout")||is("throw")||is("unknown")||is("postflush")||is("cleanup")){
   const bool clean=is("cleanup");check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==size_t(clean?2:5)&&releases==unsigned(clean?3:0)&&submits==1&&backends==1);
  }else check(r.cleanup_complete&&!r.unsafe_teardown&&allocations.empty());
  if(is("create-first"))check(r.created==0&&r.joined==0&&r.dispatches==ps5hip_workers::warmup_dispatches);
  if(is("create-second"))check(r.created==1&&r.joined==1&&r.dispatches==ps5hip_workers::warmup_dispatches);
  if(is("join"))check(r.created==2&&r.joined==1&&r.dispatches==ps5hip_workers::expected_dispatches-1);
  if(is("premature-backend"))check(ps5hip_workers::worker_initializes&&r.created==2&&r.joined==2&&r.dispatches==0&&submits==0&&backends==0&&!r.prebackend_verified&&prebackend_checks==1);
  if(is("stack"))check(!r.created&&!r.dispatches&&!submits&&!backends);
 }
 std::printf("WORKERS_MODEL_PASS mode=%s created=%u joined=%u join_attempts=%u retained=%zu model_only=true\n",scenario,r.created,r.joined,joins,allocations.size());return 0;
}
