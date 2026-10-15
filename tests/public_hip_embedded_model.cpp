// Genuine compiler registration/stubs and CLR; only providers/GPU transport are modeled.
#define PS5HIP_MODEL_CHANGING_SALT
#ifndef PS5HIP_EMBEDDED_MODULES
#define PS5HIP_EMBEDDED_MODULES 1
#endif
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
#if defined(PS5HIP_EMBEDDED_MISSING_NAME)
  checked(observed.allocations==0&&observed.frees==0&&observed.launches==0,__LINE__);
  checked(observed.queues==observed.retired_queues&&allocations.empty(),__LINE__);
#else
  checked(observed.allocations==1&&observed.frees==1&&observed.launches==PS5HIP_EMBEDDED_MODULES,__LINE__);
  checked(observed.queues==1&&observed.retired_queues==1&&allocations.empty(),__LINE__);
#endif
  std::puts("PUBLIC_EMBEDDED_FINAL ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
extern "C" void launch_public_embedded(unsigned*,unsigned,unsigned);
#if defined(PS5HIP_EMBEDDED_MISSING_NAME)
extern "C" const void* thread_words;
#endif
#if PS5HIP_EMBEDDED_MODULES == 2
extern "C" void launch_public_embedded_b(unsigned*,unsigned,unsigned);
#endif
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_EMBEDDED_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
int main(int argc,char**argv){
 checked(argc==2,__LINE__);mode=argv[1];resume=true;
 HIP_OK(hipInit(0));
#if defined(PS5HIP_EMBEDDED_MISSING_NAME)
 for(unsigned retry=0;retry<2;++retry){
  hipFuncAttributes attributes{};
  checked(hipFuncGetAttributes(&attributes,&thread_words)==hipErrorInvalidDeviceFunction,__LINE__);
  launch_public_embedded(nullptr,73,19);
  checked(hipGetLastError()==hipErrorInvalidDeviceFunction,__LINE__);
 }
 checked(backend_calls==0&&submits==0,__LINE__);completed=true;
 std::puts("PUBLIC_EMBEDDED_MISSING_NAME_PASS attribute_and_launch_rejected=true retries=2 submits=0 gpu_execution=false");
 return 0;
#endif
 constexpr unsigned words=81;unsigned input[words],output[words];
 for(unsigned i=0;i<words;++i){input[i]=0xa5000000u+i;output[i]=0xdeadbeef;}
 unsigned* buffer=nullptr;HIP_OK(hipMalloc(&buffer,sizeof(input)));
 for(unsigned pass=0;pass<PS5HIP_EMBEDDED_MODULES;++pass){
  HIP_OK(hipMemcpy(buffer,input,sizeof(input),hipMemcpyHostToDevice));
  const unsigned salt=19+pass*24;
#if PS5HIP_EMBEDDED_MODULES == 2
  if(pass)launch_public_embedded_b(buffer,73,salt);else
#endif
  launch_public_embedded(buffer,73,salt);
  HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(output,buffer,sizeof(output),hipMemcpyDeviceToHost));
  for(unsigned i=0;i<words;++i)if(output[i]!=(i<73?i*17+salt:input[i])){
   std::fprintf(stderr,"PUBLIC_EMBEDDED_OUTPUT_MISMATCH index=%u\n",i);std::abort();
  }
 }
 checked(backend_calls==1&&submits==PS5HIP_EMBEDDED_MODULES,__LINE__);
 HIP_OK(hipFree(buffer));completed=true;
 std::printf("PUBLIC_EMBEDDED_PASS compiler_stubs=true modules=%u values=%u guards=%u gpu_execution=false\n",unsigned(PS5HIP_EMBEDDED_MODULES),73u*PS5HIP_EMBEDDED_MODULES,8u*PS5HIP_EMBEDDED_MODULES);
}
