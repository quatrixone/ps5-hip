// Real public HIP/ROCclr path; only providers and GPU transport are modeled.
#define PS5HIP_MODEL_PAIRED_SALT
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {
 ~Final(){
  std::fprintf(stderr,"PUBLIC_GRAPH_GPU_COUNTS queues=%u retired=%u allocations=%u frees=%u\n",unsigned(observed.queues),unsigned(observed.retired_queues),unsigned(observed.allocations),unsigned(observed.frees));
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==2&&observed.frees==2&&observed.launches==4,__LINE__);
  const unsigned queues=3;
  checked(observed.queues==queues&&observed.retired_queues==queues&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_COPY_GPU_FINAL queues_retired=true allocations=2 frees=2 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_GRAPH_GPU_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_copy_gpu_fixture.h"
int main(int argc,char**argv){
 checked(argc==3,__LINE__);mode=argv[2];resume=true;
 copy_gpu_fixture::run(argv[1],std::strcmp(mode,"explicit")!=0,[](hipGraphExec_t exec,const hipGraphNode_t* nodes){
  if(!std::strcmp(mode,"copy-h2d"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[0],0));
  if(!std::strcmp(mode,"copy-d2d"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[2],0));
  if(!std::strcmp(mode,"copy-d2h"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[3],0));
 });
 checked(submits==4&&backend_calls==1,__LINE__);completed=true;
 std::puts("PUBLIC_COPY_GPU_PASS rounds=2 launches=4 values=292 guards=32 recovery_cases=27 setter_rejections=4 update_rejections=2 eager_equal=true disabled_replays=6 callbacks=8 gpu_execution=false");
 return 0;
}
