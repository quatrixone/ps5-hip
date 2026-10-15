// Real public HIP/ROCclr path; only providers and GPU transport are modeled.
#define PS5HIP_MODEL_CHANGING_SALT
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
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==2&&observed.frees==2&&observed.launches==2,__LINE__);
  checked(observed.queues==3&&observed.retired_queues==3&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_STREAM_GPU_FINAL queues=3 retired=3 allocations=2 frees=2 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_STREAM_GPU_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_stream_gpu_fixture.h"
int main(int argc,char**argv){
 checked(argc==3,__LINE__);mode=argv[2];
 checked(!std::strcmp(mode,"normal")||!std::strcmp(mode,"bypass")||!std::strcmp(mode,"noop")||!std::strcmp(mode,"corrupt"),__LINE__);
 // Producer host callback supplies the controlled stall; transport need not stall.
 resume=true;
 stream_gpu_fixture::run(argv[1],!std::strcmp(mode,"bypass"));
 checked(submits==2&&backend_calls==1,__LINE__);completed=true;
 std::puts("PUBLIC_STREAM_GPU_PASS rounds=2 launches=2 values=146 guards=16 source_preserved=true event_reused=true gpu_execution=false");
}
