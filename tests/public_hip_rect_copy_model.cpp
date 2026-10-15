// Rectangular D2D qualification with exact padding and failure propagation.
// Logical public HIP stream/event ordering; no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
static bool copy_case=false;
static std::mutex ordering_mutex;
static std::condition_variable ordering_changed;
static bool producer_entered=false,release_producer=false,producer_finished=false;
static bool consumer_finished=false,consumer_early=false;
struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==(copy_case?3:0)&&observed.frees==(copy_case?3:0)&&observed.launches==0&&backend_calls==0,__LINE__);
  checked(observed.queues==(copy_case?3:2)&&observed.retired_queues==(copy_case?3:2)&&allocations.empty(),__LINE__);
  std::puts(copy_case?"PUBLIC_STREAM_COPY_FINAL queues=3 retired=3 allocations=3 frees=3 ownership=clean gpu_execution=false":"PUBLIC_STREAM_FINAL queues=2 retired=2 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_STREAM_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
int main(){
 copy_case=true;HIP_OK(hipInit(0));
 hipStream_t a=nullptr,b=nullptr;HIP_OK(hipStreamCreateWithFlags(&a,hipStreamNonBlocking));HIP_OK(hipStreamCreateWithFlags(&b,hipStreamNonBlocking));
 void *src=nullptr,*dst=nullptr,*spare=nullptr;
 HIP_OK(hipMalloc(&src,512));HIP_OK(hipMalloc(&dst,512));HIP_OK(hipMalloc(&spare,512));
 auto* source=static_cast<unsigned char*>(src);auto* target=static_cast<unsigned char*>(dst);
 unsigned char seed[512],expected[512],actual[512];unsigned compared=0;
 for(unsigned round=0;round<2;++round){
  for(unsigned i=0;i<512;++i){seed[i]=static_cast<unsigned char>(i*13+round*37+1);expected[i]=0xa5;}
  HIP_OK(hipMemcpy(src,seed,512,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dst,expected,512,hipMemcpyHostToDevice));
  for(unsigned y=0;y<4;++y)for(unsigned x=0;x<8;++x)expected[5+y*24+x]=seed[3+y*16+x];
  HIP_OK(hipMemcpy2D(target+5,24,source+3,16,8,4,hipMemcpyDeviceToDevice));
  HIP_OK(hipMemcpy(actual,dst,512,hipMemcpyDeviceToHost));checked(!std::memcmp(actual,expected,512),__LINE__);compared+=512;
  std::memset(expected,0x59,512);HIP_OK(hipMemcpy(dst,expected,512,hipMemcpyHostToDevice));
  hipMemcpy3DParms desc{};desc.srcPtr=make_hipPitchedPtr(src,16,12,5);desc.dstPtr=make_hipPitchedPtr(dst,24,20,5);
  desc.srcPos=make_hipPos(2,1,0);desc.dstPos=make_hipPos(4,1,0);desc.extent=make_hipExtent(7,3,3);desc.kind=hipMemcpyDeviceToDevice;
  for(unsigned z=0;z<3;++z)for(unsigned y=0;y<3;++y)for(unsigned x=0;x<7;++x)expected[4+24+z*120+y*24+x]=seed[2+16+z*80+y*16+x];
  HIP_OK(hipMemcpy3DAsync(&desc,a));HIP_OK(hipStreamSynchronize(a));
  HIP_OK(hipMemcpy(actual,dst,512,hipMemcpyDeviceToHost));checked(!std::memcmp(actual,expected,512),__LINE__);compared+=512;
  desc.srcPos.x=10;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);desc.srcPos.x=2;
  desc.dstPos.x=20;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);desc.dstPos.x=4;
  desc.srcPos.y=4;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);desc.srcPos.y=1;
  desc.dstPos.y=4;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);desc.dstPos.y=1;
  checked(hipMemcpy2D(target+500,24,source,16,8,4,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
  checked(hipMemcpy2D(target,24,source+500,16,8,4,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
  checked(hipMemcpy2D(target,4,source,16,8,4,hipMemcpyDeviceToDevice)==hipErrorInvalidPitchValue,__LINE__);
  desc.srcPos.z=SIZE_MAX;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);
  desc.srcPos.z=0;desc.srcPtr.ysize=SIZE_MAX;checked(hipMemcpy3D(&desc)==hipErrorInvalidValue,__LINE__);
  checked(hipMemcpy2D(actual,16,source,16,8,4,hipMemcpyDeviceToHost)==hipErrorNotSupported,__LINE__);
  checked(hipMemcpy2D(target+1,16,target,16,8,2,hipMemcpyDeviceToDevice)==hipErrorUnknown,__LINE__);
  HIP_OK(hipMemcpy(actual,dst,512,hipMemcpyDeviceToHost));checked(!std::memcmp(actual,expected,512),__LINE__);compared+=512;
  HIP_OK(hipMemcpy(actual,src,512,hipMemcpyDeviceToHost));checked(!std::memcmp(actual,seed,512),__LINE__);compared+=512;
 }
 auto invalid=reinterpret_cast<hipStream_t>(uintptr_t(0x1234));
 checked(hipMemcpy2DAsync(dst,16,src,16,8,4,hipMemcpyDeviceToDevice,invalid)==hipErrorInvalidHandle,__LINE__);
 checked(hipMemcpy3DAsync(nullptr,invalid)==hipErrorInvalidHandle,__LINE__);
 checked(hipMemcpyParam2DAsync(nullptr,invalid)==hipErrorInvalidHandle,__LINE__);
 hipMemcpy3DParms capture{};capture.srcPtr=make_hipPitchedPtr(src,16,16,4);capture.dstPtr=make_hipPitchedPtr(dst,16,16,4);
 capture.extent=make_hipExtent(8,4,1);capture.kind=hipMemcpyDeviceToDevice;
 checked(hipMemcpyParam2D(nullptr)==hipErrorInvalidValue,__LINE__);
 checked(hipMemcpyParam2DAsync(nullptr,a)==hipErrorInvalidValue,__LINE__);
 HIP_MEMCPY3D driver{};driver.srcMemoryType=hipMemoryTypeDevice;driver.dstMemoryType=hipMemoryTypeDevice;
 driver.srcDevice=src;driver.dstDevice=dst;driver.srcPitch=16;driver.dstPitch=16;
 driver.srcHeight=4;driver.dstHeight=4;driver.WidthInBytes=8;driver.Height=4;driver.Depth=1;
 HIP_OK(hipStreamBeginCapture(a,hipStreamCaptureModeGlobal));
 checked(hipDrvMemcpy3DAsync(&driver,a)==hipErrorNotSupported,__LINE__);
 checked(hipMemcpy2DAsync(dst,16,src,16,8,4,hipMemcpyDeviceToDevice,a)==hipErrorNotSupported,__LINE__);
 checked(hipMemcpy3DAsync(&capture,a)==hipErrorNotSupported,__LINE__);
 hipGraph_t graph=nullptr;HIP_OK(hipStreamEndCapture(a,&graph));HIP_OK(hipGraphDestroy(graph));
 HIP_OK(hipGraphCreate(&graph,0));hipGraphNode_t node=nullptr;
 checked(hipGraphAddMemcpyNode(&node,graph,nullptr,0,&capture)==hipErrorNotSupported,__LINE__);HIP_OK(hipGraphDestroy(graph));
 checked(compared==4096,__LINE__);
 HIP_OK(hipFree(spare));HIP_OK(hipFree(dst));HIP_OK(hipFree(src));HIP_OK(hipStreamDestroy(b));HIP_OK(hipStreamDestroy(a));completed=true;
 std::puts("PUBLIC_RECT_COPY_PASS compared_bytes=4096 rounds=2 padding=true rejected=31 synchronous_failure=true gpu_execution=false");
}
