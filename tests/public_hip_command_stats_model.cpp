// Genuine Command lifetimes; modeled platform only, no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include <thread>
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {~Final(){
 const auto stats=amd::command_stats();
 checked(completed&&stats.created==stats.destroyed&&stats.live==0,__LINE__);
 checked(observed.gate_destroyed&&observed.queues==observed.retired_queues&&!observed.terminal&&!observed.uncertain,__LINE__);
 checked(allocations.empty()&&backend_calls==0,__LINE__);
 std::puts("PUBLIC_COMMAND_STATS_FINAL live=0 balanced=true ownership=clean gpu_execution=false");
}};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
struct Probe final:amd::Command {
 bool retained=false;
 explicit Probe(bool fail=false):Command(CL_COMMAND_USER){if(fail)throw std::runtime_error("derived construction");}
 Probe(amd::HostQueue& queue,bool fail=false):Command(queue,CL_COMMAND_USER){if(fail)throw std::runtime_error("derived construction");}
 void submit(amd::device::VirtualDevice&)override{checked(false,__LINE__);}
 bool terminate()override{return retained?false:Command::terminate();}
};
static void balanced_since(const amd::CommandStats& before,uint64_t count){
 const auto after=amd::command_stats();
 checked(after.created==before.created+count&&after.destroyed==before.destroyed+count&&after.live==before.live,__LINE__);
}
int main(){
 mode="normal";resume=true;checked(hipInit(0)==hipSuccess,__LINE__);
 auto before=amd::command_stats();
 auto* held=new Probe;held->retain();checked(held->release()==1,__LINE__);
 auto retained=amd::command_stats();
 checked(retained.created==before.created+1&&retained.destroyed==before.destroyed&&retained.live==before.live+1,__LINE__);
 checked(held->release()==0,__LINE__);balanced_since(before,1);
 before=amd::command_stats();held=new Probe;held->retained=true;
 checked(held->release()==0&&held->referenceCount()==0,__LINE__);
 retained=amd::command_stats();checked(retained.live==before.live+1&&retained.destroyed==before.destroyed,__LINE__);
 delete held;balanced_since(before,1);
 before=amd::command_stats();bool caught=false;
 try{Probe failing(true);}catch(const std::runtime_error&){caught=true;}
 checked(caught,__LINE__);balanced_since(before,1);
 hipStream_t stream=nullptr;checked(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking)==hipSuccess,__LINE__);
 auto& queue=*reinterpret_cast<hip::Stream*>(stream);before=amd::command_stats();
 {Probe queued(queue);}balanced_since(before,1);before=amd::command_stats();caught=false;
 try{Probe failing(queue,true);}catch(const std::runtime_error&){caught=true;}
 checked(caught,__LINE__);balanced_since(before,1);
 checked(hipStreamDestroy(stream)==hipSuccess,__LINE__);
 before=amd::command_stats();
 std::mutex gate_lock;std::condition_variable gate;unsigned ready=0,finished=0;bool proceed=false;
 constexpr unsigned threads=4,rounds=64;std::vector<std::thread> workers;
 for(unsigned i=0;i<threads;++i)workers.emplace_back([&]{
  for(unsigned round=0;round<rounds;++round){
   Probe command;
   if(round==0){std::unique_lock<std::mutex> lock(gate_lock);++ready;gate.notify_all();
    checked(gate.wait_for(lock,std::chrono::seconds(5),[&]{return proceed;}),__LINE__);}
   const auto stats=amd::command_stats();checked(stats.created>=stats.destroyed&&stats.live==stats.created-stats.destroyed&&stats.peak>=stats.live,__LINE__);
  }
  std::lock_guard<std::mutex> lock(gate_lock);++finished;gate.notify_all();
 });
 {std::unique_lock<std::mutex> lock(gate_lock);
  checked(gate.wait_for(lock,std::chrono::seconds(5),[&]{return ready==threads;}),__LINE__);
  const auto overlap=amd::command_stats();
  checked(overlap.created==before.created+threads&&overlap.destroyed==before.destroyed&&overlap.live==before.live+threads&&overlap.peak>=overlap.live,__LINE__);
  proceed=true;gate.notify_all();
  checked(gate.wait_for(lock,std::chrono::seconds(5),[&]{return finished==threads;}),__LINE__);
 }
 for(auto& worker:workers)worker.join();balanced_since(before,threads*rounds);
 completed=true;
 std::puts("PUBLIC_COMMAND_STATS_PASS constructors=2 derived_failures=2 retained_reference=true retained_zero_reference=true concurrent_threads=4 concurrent_objects=256 coherent=true gpu_execution=false");
}
