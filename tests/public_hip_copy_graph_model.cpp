// Public HIP copy graphs with exact byte and guard checks; no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false,parallel_case=false;
struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==2&&observed.frees==2&&observed.launches==0&&backend_calls==0,__LINE__);
  std::fprintf(stderr,"COPY_GRAPH_QUEUES %u %u\n",unsigned(observed.queues),unsigned(observed.retired_queues));
  checked(observed.queues==(parallel_case?5:3)&&observed.retired_queues==observed.queues&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_COPY_GRAPH_FINAL queues_retired=true ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_COPY_GRAPH_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
static void fork_join(hipStream_t stream,void* a,void* b){
 struct State {std::mutex mutex;std::condition_variable cv;bool entered=false,release=false;unsigned joined=0;} state;
 hipGraph_t g=nullptr;HIP_OK(hipGraphCreate(&g,0));hipGraphNode_t gate,copy,other,join;
 hipHostNodeParams host{[](void* p){
  auto& x=*static_cast<State*>(p);std::unique_lock<std::mutex> lock(x.mutex);
  x.entered=true;x.cv.notify_all();
  checked(x.cv.wait_for(lock,std::chrono::seconds(5),[&]{return x.release;}),__LINE__);
 },&state};
 HIP_OK(hipGraphAddHostNode(&gate,g,nullptr,0,&host));
 HIP_OK(hipGraphAddMemcpyNode1D(&copy,g,&gate,1,b,a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphAddEmptyNode(&other,g,nullptr,0));
 hipGraphNode_t deps[]={copy,other};
 host.fn=[](void* p){auto& x=*static_cast<State*>(p);std::lock_guard<std::mutex> lock(x.mutex);checked(x.release,__LINE__);++x.joined;x.cv.notify_all();};
 HIP_OK(hipGraphAddHostNode(&join,g,deps,2,&host));
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,g,nullptr,nullptr,0));
 for(unsigned enabled=0;enabled<2;++enabled){
  HIP_OK(hipGraphNodeSetEnabled(exec,copy,enabled));
  HIP_OK(hipMemset(a,0x37,128));HIP_OK(hipMemset(b,0xa5,128));
  {std::lock_guard<std::mutex> lock(state.mutex);state.entered=false;state.release=false;state.joined=0;}
  HIP_OK(hipGraphLaunch(exec,stream));
  {std::unique_lock<std::mutex> lock(state.mutex);
   checked(state.cv.wait_for(lock,std::chrono::seconds(5),[&]{return state.entered;}),__LINE__);
   checked(!state.cv.wait_for(lock,std::chrono::milliseconds(100),[&]{return state.joined!=0;}),__LINE__);
   state.release=true;state.cv.notify_all();
  }
  HIP_OK(hipStreamSynchronize(stream));checked(state.joined==1,__LINE__);
  unsigned char bytes[128];HIP_OK(hipMemcpy(bytes,b,128,hipMemcpyDeviceToHost));
  for(auto value:bytes)checked(value==(enabled?0x37:0xa5),__LINE__);
 }
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(g));
 std::puts("PUBLIC_COPY_GRAPH_JOIN_PASS rounds=2 dependency_preserved=true gpu_execution=false");
}
static void capture_errors(hipStream_t stream,void* a,void* b){
 unsigned char input[128]{};
 unsigned cases=0;
 for(auto capture_mode:{hipStreamCaptureModeGlobal,hipStreamCaptureModeThreadLocal,hipStreamCaptureModeRelaxed}){
  for(unsigned api=0;api<4;++api){
   for(unsigned fault=0;fault<(api==0?3u:2u);++fault){
    HIP_OK(hipStreamBeginCapture(stream,capture_mode));
    HIP_OK(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream));
    const size_t bytes=fault==1?129:128;
    hipError_t status;
    switch(api){
     case 0:status=hipMemcpyAsync(fault==0?nullptr:a,input,bytes,fault==2?static_cast<hipMemcpyKind>(99):hipMemcpyHostToDevice,stream);break;
     case 1:status=hipMemcpyHtoDAsync(fault==0?nullptr:a,input,bytes,stream);break;
     case 2:status=hipMemcpyDtoDAsync(fault==0?nullptr:b,a,bytes,stream);break;
     default:status=hipMemcpyDtoHAsync(fault==0?nullptr:input,a,bytes,stream);break;
    }
    checked(status==(fault==2?hipErrorInvalidMemcpyDirection:hipErrorInvalidValue),__LINE__);
    hipStreamCaptureStatus capture_status=hipStreamCaptureStatusNone;
    HIP_OK(hipStreamIsCapturing(stream,&capture_status));
    checked(capture_status==hipStreamCaptureStatusInvalidated,__LINE__);
    checked(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream)==hipErrorStreamCaptureInvalidated,__LINE__);
    checked(hipStreamBeginCapture(stream,capture_mode)==hipErrorIllegalState,__LINE__);
    hipGraph_t failed=reinterpret_cast<hipGraph_t>(uintptr_t(1));
    checked(hipStreamEndCapture(stream,&failed)==hipErrorStreamCaptureInvalidated&&failed==nullptr,__LINE__);
    HIP_OK(hipStreamIsCapturing(stream,&capture_status));checked(capture_status==hipStreamCaptureStatusNone,__LINE__);
    // A clean subsequent capture must contain only its new copy.
    HIP_OK(hipStreamBeginCapture(stream,capture_mode));
    HIP_OK(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream));
    hipGraph_t recovered=nullptr;HIP_OK(hipStreamEndCapture(stream,&recovered));
    size_t count=0;HIP_OK(hipGraphGetNodes(recovered,nullptr,&count));checked(count==1,__LINE__);
    HIP_OK(hipGraphDestroy(recovered));++cases;
   }
  }
 }
 checked(cases==27,__LINE__);
 std::puts("PUBLIC_COPY_CAPTURE_ERRORS_PASS cases=27 modes=3 aliases=4 partial_graph_rejected=true stream_reused=true gpu_execution=false");
}

int main(int argc,char**argv){
 mode="normal";resume=true;parallel_case=argc==2&&!std::strcmp(argv[1],"parallel");
 checked(argc==2,__LINE__);const bool capture=!std::strcmp(argv[1],"capture");
 HIP_OK(hipInit(0));hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));
 capture_errors(stream,a,b);
 unsigned char input[128],output[128];std::memset(input,0x31,128);std::memset(output,0xa5,128);
 HIP_OK(hipMemset(b,0xa5,128));
 hipGraph_t graph=nullptr;hipGraphNode_t nodes[3]{};
 if(capture){
  HIP_OK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
  HIP_OK(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream));
  HIP_OK(hipMemcpyAsync(b,a,128,hipMemcpyDeviceToDevice,stream));
  HIP_OK(hipMemcpyAsync(output,b,128,hipMemcpyDeviceToHost,stream));
  HIP_OK(hipStreamEndCapture(stream,&graph));
  size_t n=1;HIP_OK(hipGraphGetRootNodes(graph,&nodes[0],&n));checked(n==1,__LINE__);
  for(unsigned i=1;i<3;++i){n=1;HIP_OK(hipGraphNodeGetDependentNodes(nodes[i-1],&nodes[i],&n));checked(n==1,__LINE__);}
 }else{
  HIP_OK(hipGraphCreate(&graph,0));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[0],graph,nullptr,0,a,input,128,hipMemcpyHostToDevice));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[1],graph,&nodes[0],1,b,a,128,hipMemcpyDeviceToDevice));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],graph,&nodes[1],1,output,b,128,hipMemcpyDeviceToHost));
 }
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(unsigned disabled=0;disabled<3;++disabled){
  HIP_OK(hipGraphNodeSetEnabled(exec,nodes[disabled],0));
  for(unsigned round=0;round<2;++round){
   HIP_OK(hipMemset(a,0x17,128));HIP_OK(hipMemset(b,0xa5,128));std::memset(output,0xdd,128);
   HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
   for(auto v:output)checked(v==(disabled==0?0x17:disabled==1?0xa5:0xdd),__LINE__);
  }
  HIP_OK(hipGraphNodeSetEnabled(exec,nodes[disabled],1));
 }
 // Source edits must not mutate the already-instantiated executable.
 unsigned char replacement[128],target[128];
 for(unsigned i=0;i<128;++i)replacement[i]=static_cast<unsigned char>(i*7+19);
 auto* ap=static_cast<unsigned char*>(a);auto* bp=static_cast<unsigned char*>(b);
 HIP_OK(hipGraphMemcpyNodeSetParams1D(nodes[0],ap+16,replacement+16,64,hipMemcpyHostToDevice));
 HIP_OK(hipGraphMemcpyNodeSetParams1D(nodes[1],bp+16,ap+16,64,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphMemcpyNodeSetParams1D(nodes[2],target+16,bp+16,64,hipMemcpyDeviceToHost));
 std::memset(output,0xdd,128);std::memset(target,0xcc,128);
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(input,output,128)==0,__LINE__);
 for(auto v:target)checked(v==0xcc,__LINE__);
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[0],ap+16,replacement+16,64,hipMemcpyHostToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],bp+16,ap+16,64,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[2],target+16,bp+16,64,hipMemcpyDeviceToHost));
 // Rejected updates must leave the last valid parameters intact.
 checked(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],nullptr,ap+16,64,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],bp+16,ap+16,129,hipMemcpyDeviceToDevice)!=hipSuccess,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],bp+16,ap+16,64,hipMemcpyHostToDevice)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipMemset(a,0xa5,128));HIP_OK(hipMemset(b,0xa5,128));
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 unsigned char device_bytes[128];
 for(void* buffer:{a,b}){
  HIP_OK(hipMemcpy(device_bytes,buffer,128,hipMemcpyDeviceToHost));
  for(unsigned i=0;i<128;++i)checked(device_bytes[i]==(i>=16&&i<80?replacement[i]:0xa5),__LINE__);
 }
 for(unsigned i=0;i<128;++i)checked(target[i]==(i>=16&&i<80?replacement[i]:0xcc),__LINE__);
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[0],a,input,128,hipMemcpyHostToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],b,a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[2],output,b,128,hipMemcpyDeviceToHost));
 std::puts("PUBLIC_COPY_GRAPH_UPDATE_PASS source_isolated=true offsets=true guards=true rejected_update_preserved=true gpu_execution=false");
 HIP_OK(hipGraphDestroy(graph));
 for(unsigned round=0;round<4;++round){
  for(unsigned i=0;i<128;++i)input[i]=static_cast<unsigned char>(i*13+round*17+3);
  std::memset(output,0xa5,128);
  HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
  checked(std::memcmp(input,output,128)==0,__LINE__);
 }
 HIP_OK(hipGraphExecDestroy(exec));if(parallel_case)fork_join(stream,a,b);HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(b));HIP_OK(hipFree(a));completed=true;
 std::puts("PUBLIC_COPY_GRAPH_PASS rounds=4 bytes=512 disabled_replays=6 source_destroyed=true gpu_execution=false");
 return 0;
}
