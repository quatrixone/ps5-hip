// Shared public HIP GPU/event oracle. Caller supplies checked and HIP_OK.
#pragma once
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace stream_gpu_fixture {
struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered=false, released=false, consumed=false;
};
inline void hold(void* data) {
  auto& gate=*static_cast<Gate*>(data);
  std::unique_lock<std::mutex> lock(gate.mutex);
  gate.entered=true;gate.changed.notify_all();
  checked(gate.changed.wait_for(lock,std::chrono::seconds(5),[&]{return gate.released;}),__LINE__);
}
inline void consumed(void* data) {
  auto& gate=*static_cast<Gate*>(data);
  std::lock_guard<std::mutex> lock(gate.mutex);
  gate.consumed=true;gate.changed.notify_all();
}
inline void run(const char* path,bool bypass=false) {
  HIP_OK(hipInit(0));
  hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,path));
  hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
  hipStream_t producer=nullptr,consumer=nullptr;
  HIP_OK(hipStreamCreateWithFlags(&producer,hipStreamNonBlocking));
  HIP_OK(hipStreamCreateWithFlags(&consumer,hipStreamNonBlocking));
  hipEvent_t event=nullptr;HIP_OK(hipEventCreateWithFlags(&event,hipEventDisableTiming));
  constexpr unsigned words=81,active=73;
  // These remain live when the native failure hook quarantines without unwinding.
  static Gate gate;
  static std::array<uint32_t,words> input{},output{},source{};
  void *buffer=nullptr,*snapshot=nullptr;
  HIP_OK(hipMalloc(&buffer,sizeof(input)));HIP_OK(hipMalloc(&snapshot,sizeof(input)));
  for(unsigned round=0;round<2;++round) {
    for(unsigned i=0;i<words;++i){input[i]=0xa5000000u+i*7u+round*113u;output[i]=source[i]=0xdeadbeefu;}
    HIP_OK(hipMemcpy(buffer,input.data(),sizeof(input),hipMemcpyHostToDevice));
    HIP_OK(hipMemset(snapshot,0x5a,sizeof(input)));
    {std::lock_guard<std::mutex> lock(gate.mutex);gate.entered=gate.released=gate.consumed=false;}
    HIP_OK(hipLaunchHostFunc(producer,hold,&gate));
    unsigned count=active,salt=19+round*24;void* args[]={&buffer,&count,&salt};
    HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,producer,args,nullptr));
    HIP_OK(hipEventRecord(event,producer));
    {std::unique_lock<std::mutex> lock(gate.mutex);
     checked(gate.changed.wait_for(lock,std::chrono::seconds(5),[&]{return gate.entered;}),__LINE__);}
    checked(hipEventQuery(event)==hipErrorNotReady,__LINE__);
    if(!bypass)HIP_OK(hipStreamWaitEvent(consumer,event,0));
    HIP_OK(hipMemcpyAsync(snapshot,buffer,sizeof(input),hipMemcpyDeviceToDevice,consumer));
    HIP_OK(hipLaunchHostFunc(consumer,consumed,&gate));
    {std::unique_lock<std::mutex> lock(gate.mutex);
     const bool progressed=gate.changed.wait_for(lock,bypass?std::chrono::milliseconds(5000):std::chrono::milliseconds(100),[&]{return gate.consumed;});
     checked(progressed==bypass&&!gate.released,__LINE__);
     gate.released=true;gate.changed.notify_all();}
    HIP_OK(hipStreamSynchronize(consumer));HIP_OK(hipStreamSynchronize(producer));
    HIP_OK(hipEventSynchronize(event));HIP_OK(hipEventQuery(event));
    HIP_OK(hipStreamQuery(producer));HIP_OK(hipStreamQuery(consumer));
    {std::lock_guard<std::mutex> lock(gate.mutex);checked(gate.consumed,__LINE__);}
    HIP_OK(hipMemcpy(output.data(),snapshot,sizeof(output),hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(source.data(),buffer,sizeof(source),hipMemcpyDeviceToHost));
    for(unsigned i=0;i<words;++i){
      const uint32_t expected=i<active?i*17u+salt:input[i];
      if(output[i]!=expected||source[i]!=expected||input[i]!=0xa5000000u+i*7u+round*113u){
        std::fprintf(stderr,"PUBLIC_STREAM_GPU_OUTPUT_MISMATCH round=%u index=%u\n",round,i);
        checked(false,__LINE__);
      }
    }
  }
  HIP_OK(hipEventDestroy(event));
  HIP_OK(hipStreamDestroy(consumer));HIP_OK(hipStreamDestroy(producer));
  HIP_OK(hipFree(snapshot));HIP_OK(hipFree(buffer));HIP_OK(hipModuleUnload(module));
}
}
