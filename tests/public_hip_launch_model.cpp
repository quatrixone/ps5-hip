// Genuine public HIP launch path; providers and packet transport are modeled.
#define main adapter_unused_main
#define backend nonnull_backend
#include "clr_adapter_model.cpp"
#undef main
#undef backend
#undef check
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
static unsigned nullable_submits=0;
// Only the added count-zero dispatch has a special transport oracle. All real
// writes and negative output controls still use the existing non-null model.
class NullableTransport final:public ps5hip::SubmissionBackend {
 Transport nonnull;
 bool zero_seen=false;
public:
 void flush(void* p,size_t n)override{nonnull.flush(p,n);}
 int32_t submit(uint32_t* w,uint32_t count)override{
  if(zero_seen)return nonnull.submit(w,count);
  checked(submits==0&&nullable_submits==0&&observed.allocations==0&&observed.frees==0,__LINE__);
  std::map<unsigned,std::vector<unsigned>> registers;
  for(size_t at=0;at<count;){auto h=w[at];size_t n=((h>>16)&0x3fff)+1;
   checked(h>>30==3&&n<count-at,__LINE__);
   if((h&0xffff)==0x7600)checked(registers.emplace(w[at+1],std::vector<unsigned>(w+at+2,w+at+n+1)).second,__LINE__);
   at+=n+1;
  }
  const auto& user=registers.at(0x240);const auto& program=registers.at(0x20c);
  auto arg=uint64_t(user.at(4))|(uint64_t(user.at(5))<<32);live(arg,16);
  uint64_t output=1;unsigned n=1,salt=0;
  std::memcpy(&output,reinterpret_cast<void*>(arg),8);
  std::memcpy(&n,reinterpret_cast<void*>(arg+8),4);
  std::memcpy(&salt,reinterpret_cast<void*>(arg+12),4);
  checked(output==0&&n==0&&salt==19,__LINE__);
  const auto entry=(uint64_t(program.at(0))<<8)|(uint64_t(program.at(1))<<40);live(entry,4);
  checked(count>=8,__LINE__);const auto* end=w+count-8;
  checked(end[0]==0xc0064900,__LINE__);
  auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);
  // thread_words guards every store with index<count; count zero has no stores.
  // No live-range lookup, dereference or substitute allocation for output zero.
  *reinterpret_cast<volatile unsigned*>(marker)=1;
  zero_seen=true;++nullable_submits;return 0;
 }
 int32_t kick()override{return nonnull.kick();}
 uint64_t now_ns()override{return nonnull.now_ns();}
 void sleep_us(unsigned us)override{nonnull.sleep_us(us);}
};
static std::unique_ptr<ps5hip::SubmissionBackend> backend(){++backend_calls;return std::make_unique<NullableTransport>();}

struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==1&&observed.frees==1&&observed.launches==2&&
          submits+nullable_submits==2,__LINE__);
  checked(observed.queues==2&&observed.retired_queues==2&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_LAUNCH_FINAL ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_LAUNCH_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
int main(int argc,char**argv){
 checked(argc==3,__LINE__);mode=argv[2];
 HIP_OK(hipInit(0));
 checked(hipLaunchKernel(reinterpret_cast<void*>(1),dim3(1),dim3(64),nullptr,0,reinterpret_cast<hipStream_t>(0x1234))==hipErrorInvalidValue,__LINE__);
 hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,argv[1]));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 void* absent=nullptr;unsigned zero_count=0,zero_salt=19;
 void* zero_args[]={&absent,&zero_count,&zero_salt};
 HIP_OK(hipModuleLaunchKernel(function,1,1,1,64,1,1,0,nullptr,zero_args,nullptr));
 HIP_OK(hipDeviceSynchronize());
 checked(nullable_submits==1&&submits==0&&backend_calls==1&&observed.launches==1&&
         observed.allocations==0&&observed.frees==0&&allocations.empty(),__LINE__);
 {std::lock_guard<std::mutex> lock(mutex);checked(!entered&&!resume,__LINE__);}
 std::puts("PUBLIC_NULL_LAUNCH_PASS packed_null=true count=0 user_allocations=0 retained=0 gpu_execution=false");
 constexpr unsigned words=81;unsigned input[words],output[words];
 for(unsigned i=0;i<words;++i){input[i]=0xa5000000u+i;output[i]=0xdeadbeef;}
 void* buffer=nullptr;HIP_OK(hipMalloc(&buffer,sizeof(input)));HIP_OK(hipMemcpy(buffer,input,sizeof(input),hipMemcpyHostToDevice));
 unsigned count=73,salt=19;void* args[]={&buffer,&count,&salt};
 // Every rejected request must leave the device usable and submit no work.
 const unsigned shapes[][6]={{2,1,1,32,1,1},{2,1,1,128,1,1},
   {2,1,1,8,8,1},{2,2,1,64,1,1},{2,1,2,64,1,1},
   {2,1,1,64,2,1},{2,1,1,64,1,2}};
 for(const auto& shape:shapes)
  checked(hipModuleLaunchKernel(function,shape[0],shape[1],shape[2],shape[3],shape[4],shape[5],0,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
 checked(hipModuleLaunchKernel(function,2,1,1,64,1,1,4,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
 checked(hipExtModuleLaunchKernel(function,73,1,1,64,1,1,0,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
 for(size_t shared:{size_t(1),size_t(1)<<32}){
  checked(hipExtModuleLaunchKernel(function,128,1,1,64,1,1,shared,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
  checked(hipHccModuleLaunchKernel(function,128,1,1,64,1,1,shared,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
 }
 void* extra[]={HIP_LAUNCH_PARAM_END};
 checked(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,nullptr,extra)==hipErrorNotSupported,__LINE__);
 void* null_args[]={&buffer,nullptr,&salt};
 checked(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,null_args,nullptr)==hipErrorInvalidValue,__LINE__);
 hipStream_t capturing=nullptr;HIP_OK(hipStreamCreate(&capturing));
 HIP_OK(hipStreamBeginCapture(capturing,hipStreamCaptureModeThreadLocal));
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,capturing,args,nullptr));
 checked(hipExtModuleLaunchKernel(function,128,1,1,64,1,1,0,capturing,args,nullptr)==hipErrorNotSupported,__LINE__);
 checked(hipHccModuleLaunchKernel(function,128,1,1,64,1,1,0,capturing,args,nullptr)==hipErrorNotSupported,__LINE__);
 hipGraph_t graph=nullptr;HIP_OK(hipStreamEndCapture(capturing,&graph));
 size_t nodes=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&nodes));checked(nodes==1,__LINE__);
 hipKernelNodeParams params{};params.func=reinterpret_cast<void*>(function);
 params.gridDim=dim3(2,1,1);params.blockDim=dim3(64,1,1);params.kernelParams=args;
 hipGraphNode_t node=nullptr;
 HIP_OK(hipGraphAddKernelNode(&node,graph,nullptr,0,&params));
 HIP_OK(hipGraphDestroy(graph));HIP_OK(hipStreamDestroy(capturing));
 checked(nullable_submits==1&&submits==0&&backend_calls==1&&!observed.terminal&&!observed.uncertain,__LINE__);
 const bool event_case=std::strcmp(mode,"events")==0;
 hipEvent_t completion=nullptr;
 if(event_case)HIP_OK(hipEventCreateWithFlags(&completion,hipEventDisableTiming));
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,args,nullptr));
 if(event_case)HIP_OK(hipEventRecord(completion,nullptr));
 {
  std::unique_lock<std::mutex> lock(mutex);
  checked(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),__LINE__);
  if(event_case)checked(hipEventQuery(completion)==hipErrorNotReady,__LINE__);
  salt=999;count=0;resume=true;changed.notify_all();
 }
 if(event_case){
  HIP_OK(hipEventSynchronize(completion));HIP_OK(hipEventQuery(completion));
  HIP_OK(hipEventDestroy(completion));
 }
 HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(output,buffer,sizeof(output),hipMemcpyDeviceToHost));
 for(unsigned i=0;i<words;++i)if(output[i]!=(i<73?i*17+19:input[i])){
  std::fprintf(stderr,"PUBLIC_LAUNCH_OUTPUT_MISMATCH index=%u\n",i);std::abort();
 }
 checked(backend_calls==1&&submits==1&&nullable_submits==1,__LINE__);
 HIP_OK(hipFree(buffer));HIP_OK(hipModuleUnload(module));completed=true;
 if(event_case)std::puts("PUBLIC_EVENT_PASS pending_then_complete=true values=73 guards=8 captured_args=true gpu_execution=false");
 else std::puts("PUBLIC_LAUNCH_PASS values=73 guards=8 captured_args=true gpu_execution=false");
}
