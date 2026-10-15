// Public HIP copy graphs with exact byte and guard checks; no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_graph_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false,parallel_case=false;
struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==2&&observed.frees==2&&observed.launches==0&&backend_calls==0,__LINE__);
  std::fprintf(stderr,"COPY_GRAPH_QUEUES %u %u\n",unsigned(observed.queues),unsigned(observed.retired_queues));
  checked(observed.queues==(parallel_case?8:6)&&observed.retired_queues==observed.queues&&allocations.empty(),__LINE__);
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
#include "public_hip_nested_update_fixture.h"
static void graph_updates(hipGraphExec_t exec,hipGraph_t original,hipStream_t stream,
                          void* a,void* b,unsigned char* input,unsigned char* output){
 unsigned char alternate[128],target[128];std::memset(alternate,0x79,128);std::memset(target,0xdd,128);
 hipGraph_t replacement=nullptr;HIP_OK(hipGraphCreate(&replacement,0));hipGraphNode_t nodes[3]{};
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[0],replacement,nullptr,0,a,alternate,128,hipMemcpyHostToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[1],replacement,&nodes[0],1,b,a,128,hipMemcpyDeviceToDevice));
 hipMemcpy3DParms params{};params.srcPtr=make_hipPitchedPtr(b,128,128,1);
 params.dstPtr=make_hipPitchedPtr(target,128,128,1);params.extent=make_hipExtent(128,1,1);params.kind=hipMemcpyDeviceToHost;
 // The public 3D constructor is unsupported; retain its actual internal representation.
 auto* three_d=new hip::GraphMemcpyNode(&params);
 reinterpret_cast<hip::Graph*>(replacement)->AddNode(three_d);
 reinterpret_cast<hip::GraphNode*>(nodes[1])->AddEdge(three_d);
 nodes[2]=reinterpret_cast<hipGraphNode_t>(three_d);
 hipGraphNode_t error=nullptr;hipGraphExecUpdateResult result=hipGraphExecUpdateSuccess;
 checked(hipGraphExecUpdate(exec,replacement,&error,&result)==hipErrorGraphExecUpdateFailure,__LINE__);
 checked(result==hipGraphExecUpdateErrorNodeTypeChanged&&error==nodes[2],__LINE__);
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(input,output,128)==0,__LINE__);
 for(auto v:target)checked(v==0xdd,__LINE__);
 // Replace only the incompatible representation, then prove compatible update/replay.
 HIP_OK(hipGraphDestroyNode(nodes[2]));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],replacement,&nodes[1],1,target,b,128,hipMemcpyDeviceToHost));
 HIP_OK(hipGraphExecUpdate(exec,replacement,&error,&result));checked(result==hipGraphExecUpdateSuccess,__LINE__);
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(alternate,target,128)==0,__LINE__);
 auto fake_exec=reinterpret_cast<hipGraphExec_t>(uintptr_t(1));
 auto fake_graph=reinterpret_cast<hipGraph_t>(uintptr_t(1));
 checked(hipGraphExecUpdate(fake_exec,replacement,&error,&result)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecUpdate(exec,fake_graph,&error,&result)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipGraphExecUpdate(exec,original,&error,&result));
 HIP_OK(hipGraphDestroy(replacement));
 checked(hipGraphExecUpdate(exec,replacement,&error,&result)==hipErrorInvalidValue,__LINE__);
 std::puts("PUBLIC_GRAPH_EXEC_UPDATE_PASS late_type_rejection_preserved=true compatible_replay=true invalid_handles=true gpu_execution=false");
}
static void setter_errors(hipGraphExec_t exec,hipGraphNode_t copy,void* a,void* b){
 hipGraph_t graph=nullptr;HIP_OK(hipGraphCreate(&graph,0));
 hipGraphNode_t wrong[2]{};
 HIP_OK(hipGraphAddEmptyNode(&wrong[0],graph,nullptr,0));
 hipHostNodeParams host{[](void*){},nullptr};
 HIP_OK(hipGraphAddHostNode(&wrong[1],graph,&wrong[0],1,&host));
 hipGraphExec_t other=nullptr;HIP_OK(hipGraphInstantiate(&other,graph,nullptr,nullptr,0));
 for(auto node:wrong){
  checked(hipGraphMemcpyNodeSetParams1D(node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
  checked(hipGraphExecMemcpyNodeSetParams1D(other,node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 }
 // Public 3D-node creation is unsupported; exercise its registered internal type.
 hipMemcpy3DParms params{};
 hip::GraphMemcpyNode three_d(&params);
 hip::GraphMemcpyNodeFromSymbol from_symbol(b,a,128,0,hipMemcpyDeviceToDevice);
 hip::GraphMemcpyNodeToSymbol to_symbol(a,b,128,0,hipMemcpyDeviceToDevice);
 for(hip::GraphNode* internal:{static_cast<hip::GraphNode*>(&three_d),static_cast<hip::GraphNode*>(&from_symbol),static_cast<hip::GraphNode*>(&to_symbol)}){
  auto node=reinterpret_cast<hipGraphNode_t>(internal);
  checked(hipGraphMemcpyNodeSetParams1D(node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
  checked(hipGraphExecMemcpyNodeSetParams1D(exec,node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 }
 auto fake_node=reinterpret_cast<hipGraphNode_t>(uintptr_t(1));
 checked(hipGraphMemcpyNodeSetParams1D(fake_node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(exec,fake_node,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 auto fake_exec=reinterpret_cast<hipGraphExec_t>(uintptr_t(1));
 checked(hipGraphExecMemcpyNodeSetParams1D(fake_exec,copy,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(other,copy,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipGraphExecDestroy(other));
 checked(hipGraphExecMemcpyNodeSetParams1D(other,copy,b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipGraphDestroy(graph));
 checked(hipGraphMemcpyNodeSetParams1D(wrong[1],b,a,128,hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 std::puts("PUBLIC_COPY_SETTER_ERRORS_PASS public_wrong_types=2 internal_subtypes=3 foreign_and_stale_handles=true gpu_execution=false");
}
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
#include "public_hip_copy_recovery_fixture.h"
int main(int argc,char**argv){
 mode="normal";resume=true;parallel_case=argc==2&&!std::strcmp(argv[1],"parallel");
 checked(argc==2,__LINE__);const bool capture=!std::strcmp(argv[1],"capture");
 HIP_OK(hipInit(0));hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));
 copy_recovery::capture_errors(stream,a,b);
 std::puts("PUBLIC_COPY_CAPTURE_ERRORS_PASS cases=27 modes=3 aliases=4 partial_graph_rejected=true stream_reused=true gpu_execution=false");
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
 setter_errors(exec,nodes[1],a,b);
 graph_updates(exec,graph,stream,a,b,input,output);
 nested_updates(stream,a,b);
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
