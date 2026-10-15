// Actual public HIP/ROCclr lifetime with modeled memory; no GPU execution.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {~Final(){
 checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
 checked(observed.allocations==6&&observed.frees==6&&observed.launches==0&&backend_calls==0,__LINE__);
 checked(observed.queues==4&&observed.retired_queues==4&&allocations.empty(),__LINE__);
 std::puts("PUBLIC_ALLOCATION_LIFETIME_FINAL ownership=clean gpu_execution=false");
}};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
namespace allocation_lifetime_fixture {inline void* free_source(void*);}
static pthread_t freeing_thread{};static bool freeing_thread_created=false;
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){
 const bool target=f==allocation_lifetime_fixture::free_source;
 if(target&&!std::strcmp(mode,"thread-create-fail")){
  return EAGAIN;
 }
 const int result=__real_pthread_create(p,a,f,d);
 if(target&&result==0){freeing_thread=*p;freeing_thread_created=true;}
 return result;
}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){
 if(freeing_thread_created&&pthread_equal(p,freeing_thread)){
  if(!std::strcmp(mode,"thread-join-fail"))return EINVAL;
  freeing_thread_created=false;
 }
 return __real_pthread_join(p,v);
}
static void ok(hipError_t s,int l){if(s!=hipSuccess){std::fprintf(stderr,"LIFETIME_ERROR line=%d status=%d\n",l,int(s));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_allocation_lifetime_fixture.h"
int main(int argc,char**argv){
 checked(argc==1||(argc==2&&(!std::strcmp(argv[1],"thread-create-fail")||!std::strcmp(argv[1],"thread-join-fail"))),__LINE__);
 mode=argc==2?argv[1]:"normal";resume=true;HIP_OK(hipInit(0));
 allocation_lifetime_fixture::run();completed=true;
 std::puts("PUBLIC_QUEUED_FREE_PASS gated_free_observed=true output_exact=true exec_alive_at_free=true gpu_execution=false");
 std::puts("PUBLIC_ALLOCATION_MIGRATION_PASS old_freed=true source_graph_alive=true replay_exact=true gpu_execution=false");
 return 0;
}
