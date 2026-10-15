// Public HIP buffer fills with exact byte and guard checks; no GPU submission.
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
  checked(observed.allocations==1&&observed.frees==1&&observed.launches==0&&backend_calls==0,__LINE__);
  checked(observed.queues==2&&observed.retired_queues==2&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_FILL_FINAL queues=2 retired=2 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_FILL_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
int main(int argc,char**argv){
 checked(argc==1||(argc==2&&std::strcmp(argv[1],"rejected-command")==0),__LINE__);
 HIP_OK(hipInit(0));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void* buffer=nullptr;HIP_OK(hipMalloc(&buffer,128));
 unsigned char expected[128],actual[128];std::memset(expected,0x39,sizeof(expected));
 if(argc==2){
  HIP_OK(hipMemcpy(buffer,expected,sizeof(expected),hipMemcpyHostToDevice));
  checked(hipMemset(static_cast<char*>(buffer)+3,0xa5,9)==hipErrorUnknown,__LINE__);
  HIP_OK(hipMemcpy(actual,buffer,sizeof(actual),hipMemcpyDeviceToHost));
  checked(std::memcmp(actual,expected,sizeof(actual))==0,__LINE__);
  HIP_OK(hipFree(buffer));HIP_OK(hipStreamDestroy(stream));completed=true;
  std::puts("PUBLIC_FILL_REJECTION completion_error=true unchanged=true gpu_execution=false");return 0;
 }
 HIP_OK(hipMemset(buffer,0x39,sizeof(expected)));HIP_OK(hipDeviceSynchronize());
 HIP_OK(hipMemset(static_cast<char*>(buffer)+3,0x1a5,9));std::memset(expected+3,0xa5,9);
 HIP_OK(hipMemsetD16(static_cast<char*>(buffer)+16,0xb37d,7));
 for(size_t i=16;i<30;i+=2){expected[i]=0x7d;expected[i+1]=0xb3;}
 HIP_OK(hipMemsetD32Async(static_cast<char*>(buffer)+40,0x87654321,11,stream));
 for(size_t i=40;i<84;i+=4){expected[i]=0x21;expected[i+1]=0x43;expected[i+2]=0x65;expected[i+3]=0x87;}
 HIP_OK(hipMemsetAsync(static_cast<char*>(buffer)+121,0xe4,5,stream));std::memset(expected+121,0xe4,5);
 HIP_OK(hipStreamSynchronize(stream));HIP_OK(hipDeviceSynchronize());
 checked(hipMemset(static_cast<char*>(buffer)+127,0,2)==hipErrorInvalidValue,__LINE__);
 checked(hipMemsetD16(static_cast<char*>(buffer)+1,0,1)==hipErrorInvalidValue,__LINE__);
 checked(hipMemsetD32(static_cast<char*>(buffer)+2,0,1)==hipErrorInvalidValue,__LINE__);
 checked(hipMemsetD16(buffer,0,SIZE_MAX/2+1)==hipErrorInvalidValue,__LINE__);
 checked(hipMemsetD32Async(buffer,0,SIZE_MAX/4+1,stream)==hipErrorInvalidValue,__LINE__);
 checked(hipMemset2D(buffer,16,0,8,2)==hipErrorNotSupported,__LINE__);
 checked(hipMemset3D({buffer,16,8,2},0,{8,2,2})==hipErrorNotSupported,__LINE__);
 checked(hipMemsetAsync(buffer,0,1,reinterpret_cast<hipStream_t>(0x12345))==hipErrorInvalidHandle,__LINE__);
 HIP_OK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
 checked(hipMemsetAsync(buffer,0,1,stream)==hipErrorNotSupported,__LINE__);
 hipGraph_t graph=nullptr;HIP_OK(hipStreamEndCapture(stream,&graph));HIP_OK(hipGraphDestroy(graph));
 HIP_OK(hipGraphCreate(&graph,0));hipGraphNode_t node=nullptr;hipMemsetParams params{};
 params.dst=buffer;params.value=0;params.elementSize=1;params.width=1;params.height=1;
 checked(hipGraphAddMemsetNode(&node,graph,nullptr,0,&params)==hipErrorNotSupported,__LINE__);
 HIP_OK(hipGraphDestroy(graph));
 HIP_OK(hipMemcpy(actual,buffer,sizeof(actual),hipMemcpyDeviceToHost));
 for(size_t i=0;i<sizeof(actual);++i)if(actual[i]!=expected[i]){
  std::fprintf(stderr,"PUBLIC_FILL_MISMATCH index=%zu actual=%u expected=%u\n",i,actual[i],expected[i]);std::abort();
 }
 HIP_OK(hipFree(buffer));HIP_OK(hipStreamDestroy(stream));completed=true;
 std::puts("PUBLIC_FILL_PASS bytes=128 patterns=1,2,4 guards=true gpu_execution=false");
}
