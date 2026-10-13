// Reuse the actual packet/memory transport model, not a fake HIP implementation.
#define main prior_thread_model_main
#include "hip_threads_model.cpp"
#undef main
#include "../tools/hip_workers_client.h"
#include <string>
namespace {
const char*scenario="normal";unsigned creates=0,joins=0,mutex_destroyed=0,cond_destroyed=0;
std::atomic<unsigned> unlocks{0},waits{0},broadcasts{0};
bool is(const char*s){return std::strcmp(scenario,s)==0;}
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
 check(index<5&&ps5hip_workers::actor()==expected_actor[index++]);threads_launch(p,n,salt);
 if(hipPeekAtLastError()==hipSuccess&&is("guard"))p[-1]^=1;
}
int main(int argc,char**argv){
 check(argc==2);scenario=argv[1];mode=scenario;
 const int result=ps5hip_workers_client_main(0,nullptr);const auto&r=ps5hip_workers::report();
 check(r.created==joins); // A failed join is still attempted once; all real model workers joined.
 if(is("normal")){
  check(result==0&&r.passed&&r.cleanup_complete&&!r.unsafe_teardown&&r.created==2&&r.joined==2&&r.dispatches==5&&r.active_values==661&&r.word_comparisons==1355&&r.template_comparisons==1355&&r.phase_checks==17);
  check(allocations.empty()&&backends==1&&submits==5&&mutex_destroyed==1&&cond_destroyed==1);ps5hip_runtime_finish();check(ps5hip_runtime_ready());
 }else{
  expect_final=false;check(result==1&&!r.passed&&r.error);
  if(is("join")||is("unlock")||is("wait-unknown")||is("broadcast")){check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==1&&mutex_destroyed==0&&cond_destroyed==0);}
  else if(is("free"))check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==1&&submits==5);
  else if(is("destroy"))check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.empty()&&submits==5);
  else if(is("submit")||is("kick")||is("timeout")||is("throw")||is("unknown")||is("postflush")||is("cleanup")){
   const bool clean=is("cleanup");check(r.unsafe_teardown&&!r.cleanup_complete&&allocations.size()==size_t(clean?2:5)&&releases==unsigned(clean?3:0)&&submits==1&&backends==1);
  }else check(r.cleanup_complete&&!r.unsafe_teardown&&allocations.empty());
  if(is("create-first"))check(r.created==0&&r.joined==0&&r.dispatches==1);
  if(is("create-second"))check(r.created==1&&r.joined==1&&r.dispatches==1);
  if(is("join"))check(r.created==2&&r.joined==1&&r.dispatches==4);
  if(is("stack"))check(!r.created&&!r.dispatches&&!submits&&!backends);
 }
 std::printf("WORKERS_MODEL_PASS mode=%s created=%u joined=%u join_attempts=%u retained=%zu model_only=true\n",scenario,r.created,r.joined,joins,allocations.size());return 0;
}
