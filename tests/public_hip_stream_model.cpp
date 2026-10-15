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
int main(int argc,char**argv){
 checked(argc==2,__LINE__);
 const bool default_case=std::strncmp(argv[1],"default-",8)==0;
 const bool reverse=std::strcmp(argv[1],"default-reverse-blocking")==0||std::strcmp(argv[1],"default-reverse-nonblocking")==0;
 const bool nonblocking=std::strcmp(argv[1],"default-nonblocking")==0||std::strcmp(argv[1],"default-reverse-nonblocking")==0;
 checked(std::strcmp(argv[1],"normal")==0||std::strcmp(argv[1],"bypass")==0||
         std::strcmp(argv[1],"copy")==0||std::strcmp(argv[1],"copy-bypass")==0||
         std::strcmp(argv[1],"default-blocking")==0||std::strcmp(argv[1],"default-nonblocking")==0||
         std::strcmp(argv[1],"default-reverse-blocking")==0||std::strcmp(argv[1],"default-reverse-nonblocking")==0,__LINE__);
 copy_case=std::strncmp(argv[1],"copy",4)==0;
 const bool bypass=std::strcmp(argv[1],"bypass")==0||std::strcmp(argv[1],"copy-bypass")==0||(default_case&&nonblocking);
 HIP_OK(hipInit(0));hipStream_t producer=nullptr,consumer=nullptr;
 if(!default_case||!reverse)HIP_OK(hipStreamCreateWithFlags(&producer,default_case&&!nonblocking?hipStreamDefault:hipStreamNonBlocking));
 if(!default_case||reverse)HIP_OK(hipStreamCreateWithFlags(&consumer,default_case&&!nonblocking?hipStreamDefault:hipStreamNonBlocking));
 void *input=nullptr,*middle=nullptr,*output=nullptr;
 unsigned char seed[128],actual[128];
 for(unsigned i=0;i<128;++i)seed[i]=static_cast<unsigned char>(i*13+7);
 if(copy_case){
  HIP_OK(hipMalloc(&input,128));HIP_OK(hipMalloc(&middle,128));HIP_OK(hipMalloc(&output,128));
  HIP_OK(hipMemcpy(input,seed,128,hipMemcpyHostToDevice));
  HIP_OK(hipMemset(middle,0,128));HIP_OK(hipMemset(output,0xa5,128));
 }
 hipEvent_t event=nullptr;HIP_OK(hipEventCreateWithFlags(&event,hipEventDisableTiming));
 HIP_OK(hipLaunchHostFunc(producer,[](void*){
  std::unique_lock<std::mutex> lock(ordering_mutex);producer_entered=true;ordering_changed.notify_all();
  checked(ordering_changed.wait_for(lock,std::chrono::seconds(5),[]{return release_producer;}),__LINE__);
  producer_finished=true;ordering_changed.notify_all();
 },nullptr));
 if(copy_case)HIP_OK(hipMemcpyAsync(middle,input,128,hipMemcpyDeviceToDevice,producer));
 HIP_OK(hipEventRecord(event,producer));
 {
  std::unique_lock<std::mutex> lock(ordering_mutex);
  checked(ordering_changed.wait_for(lock,std::chrono::seconds(5),[]{return producer_entered;}),__LINE__);
 }
 checked(hipEventQuery(event)==hipErrorNotReady,__LINE__);
 if(!bypass&&!default_case)HIP_OK(hipStreamWaitEvent(consumer,event,0));
 if(copy_case)HIP_OK(hipMemcpyAsync(output,middle,128,hipMemcpyDeviceToDevice,consumer));
 HIP_OK(hipLaunchHostFunc(consumer,[](void*){
  std::lock_guard<std::mutex> lock(ordering_mutex);consumer_early=!producer_finished;
  consumer_finished=true;ordering_changed.notify_all();
 },nullptr));
 {
  std::unique_lock<std::mutex> lock(ordering_mutex);
  // Both cases use the same bounded wait; bypass wakes early on consumer progress.
  const bool progressed=ordering_changed.wait_for(lock,std::chrono::seconds(1),[]{return consumer_finished;});
  checked(progressed==bypass&&!producer_finished,__LINE__);
  if(bypass)checked(consumer_early,__LINE__);
  release_producer=true;ordering_changed.notify_all();
 }
 HIP_OK(hipStreamSynchronize(consumer));
 {std::lock_guard<std::mutex> lock(ordering_mutex);checked(consumer_finished&&consumer_early==bypass,__LINE__);}
 HIP_OK(hipEventSynchronize(event));HIP_OK(hipEventQuery(event));
 if(copy_case){
  HIP_OK(hipStreamSynchronize(producer));
  HIP_OK(hipMemcpy(actual,output,128,hipMemcpyDeviceToHost));
  for(unsigned i=0;i<128;++i)checked(actual[i]==(bypass?0:seed[i]),__LINE__);
  HIP_OK(hipFree(output));HIP_OK(hipFree(middle));HIP_OK(hipFree(input));
 }
 HIP_OK(hipStreamSynchronize(producer));
 HIP_OK(hipStreamQuery(producer));HIP_OK(hipStreamQuery(consumer));
 HIP_OK(hipEventDestroy(event));
 if(consumer)HIP_OK(hipStreamDestroy(consumer));
 if(producer)HIP_OK(hipStreamDestroy(producer));
 completed=true;
 if(default_case){std::printf("PUBLIC_DEFAULT_STREAM_PASS direction=%s nonblocking=%s implicit_order=%s gpu_execution=false\n",reverse?"default-to-explicit":"explicit-to-default",nonblocking?"true":"false",nonblocking?"false":"true");return 0;}
 if(copy_case)std::puts(bypass?"PUBLIC_STREAM_COPY_BYPASS stale_bytes=128 gpu_execution=false":"PUBLIC_STREAM_COPY_PASS exact_bytes=128 event_ordered=true gpu_execution=false");
 else std::puts(bypass?"PUBLIC_STREAM_BYPASS early_consumer_detected=true gpu_execution=false":"PUBLIC_STREAM_PASS event_ordered=true gpu_execution=false");
}
