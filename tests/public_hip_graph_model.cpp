// Public HIP host-node graph replay, updates and lifetime; no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
static bool combined=false;

struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==0&&observed.frees==0&&observed.launches==0&&backend_calls==0,__LINE__);
  checked(observed.queues==(combined?5:3)&&observed.retired_queues==(combined?5:3)&&allocations.empty(),__LINE__);
  std::printf("PUBLIC_GRAPH_FINAL queues=%u retired=%u allocations=0 frees=0 ownership=clean gpu_execution=false\n",combined?5:3,combined?5:3);
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_GRAPH_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_graph_fixture.h"
int main(int argc,char**argv){
 checked(argc==2,__LINE__);combined=std::strcmp(argv[1],"combined")==0;
 if(combined){
  run_graph_fixture("explicit",3,false);run_graph_fixture("recover",5,false);
  checked(observed.queues==5&&observed.retired_queues==4,__LINE__);
  std::puts("PUBLIC_GRAPH_COMBINED_PASS graphs=2 replays=32 callbacks=64 invalidations=6 gpu_execution=false");
 }else run_graph_fixture(argv[1],3,true);
 completed=true;
}
