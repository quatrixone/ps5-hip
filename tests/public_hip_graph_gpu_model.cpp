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
  std::fprintf(stderr,"PUBLIC_GRAPH_GPU_COUNTS queues=%u retired=%u allocations=%u frees=%u\n",unsigned(observed.queues),unsigned(observed.retired_queues),unsigned(observed.allocations),unsigned(observed.frees));
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==1&&observed.frees==1&&observed.launches==2,__LINE__);
  checked(observed.queues==7&&observed.retired_queues==7&&allocations.empty(),__LINE__);
  std::puts("PUBLIC_GRAPH_GPU_FINAL queues_retired=true allocations=1 frees=1 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_GRAPH_GPU_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
static int fail_argument = -1;
static unsigned argument_calls = 0;
static int fail_command = -1;
static unsigned command_calls = 0;
static unsigned callbacks = 0;
extern "C" void* ps5hip_graph_argument_alloc(size_t bytes){
 const auto call=argument_calls++;
 if(fail_argument>=0&&call==unsigned(fail_argument))return nullptr;
 return std::malloc(bytes);
}
extern "C" hipError_t ps5hip_graph_command_status(){return fail_command>=0&&command_calls++==unsigned(fail_command)?hipErrorOutOfMemory:hipSuccess;}
static void inject(int at){fail_argument=at;argument_calls=0;}
int main(int argc,char**argv){
 checked(argc==3,__LINE__);mode=argv[2];resume=true;
 HIP_OK(hipInit(0));
 hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,argv[1]));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void* buffer=nullptr;HIP_OK(hipMalloc(&buffer,81*sizeof(unsigned)));
 unsigned count=73,salt=19;void* args[]={&buffer,&count,&salt};
 hipKernelNodeParams params{};params.func=reinterpret_cast<void*>(function);
 params.gridDim=dim3(2);params.blockDim=dim3(64);params.kernelParams=args;
 hipGraph_t graph=nullptr;HIP_OK(hipGraphCreate(&graph,0));
 hipGraphNode_t node=nullptr;
 // Overflow must be rejected before narrowing; no graph mutation or dispatch.
 params.gridDim.x=0x40000002u;
 checked(hipGraphAddKernelNode(&node,graph,nullptr,0,&params)==hipErrorInvalidConfiguration,__LINE__);
 params.gridDim.x=2;
 for(int at=0;at<4;++at){
  inject(at);node=reinterpret_cast<hipGraphNode_t>(0x1234);
  checked(hipGraphAddKernelNode(&node,graph,nullptr,0,&params)==hipErrorOutOfMemory&&node==nullptr,__LINE__);
  size_t nodes=99;HIP_OK(hipGraphGetNodes(graph,nullptr,&nodes));checked(nodes==0,__LINE__);
 }
 inject(-1);
 HIP_OK(hipGraphAddKernelNode(&node,graph,nullptr,0,&params));
 hipGraphNode_t host=nullptr;hipHostNodeParams host_params{[](void*){++callbacks;},nullptr};
 HIP_OK(hipGraphAddHostNode(&host,graph,&node,1,&host_params));
 for(int at=0;at<4;++at){
  inject(at);hipGraph_t clone=nullptr;
  checked(hipGraphClone(&clone,graph)==hipErrorOutOfMemory&&clone==nullptr,__LINE__);
  inject(at);hipGraphExec_t failed=nullptr;
  checked(hipGraphInstantiate(&failed,graph,nullptr,nullptr,0)==hipErrorOutOfMemory&&failed==nullptr,__LINE__);
  inject(at);hipGraphNode_t child=nullptr;hipGraph_t parent=nullptr;
  HIP_OK(hipGraphCreate(&parent,0));
  checked(hipGraphAddChildGraphNode(&child,parent,nullptr,0,graph)==hipErrorOutOfMemory&&child==nullptr,__LINE__);
  HIP_OK(hipGraphDestroy(parent));
 }
 inject(-1);
 // A partially constructed parallel batch, directly and inside a child graph,
 // must release all earlier commands without submitting either kernel.
 for(bool child:{false,true}){
  hipGraph_t parallel=nullptr,parent=nullptr;hipGraphExec_t batch=nullptr;
  HIP_OK(hipGraphCreate(&parallel,0));hipGraphNode_t roots[2]{};
  HIP_OK(hipGraphAddKernelNode(&roots[0],parallel,nullptr,0,&params));
  HIP_OK(hipGraphAddKernelNode(&roots[1],parallel,nullptr,0,&params));
  unsigned completed_batch=0;hipGraphNode_t join=nullptr;
  hipHostNodeParams joined{[](void* p){++*static_cast<unsigned*>(p);},&completed_batch};
  HIP_OK(hipGraphAddHostNode(&join,parallel,roots,2,&joined));
  if(child){
   HIP_OK(hipGraphCreate(&parent,0));hipGraphNode_t nested=nullptr;
   HIP_OK(hipGraphAddChildGraphNode(&nested,parent,nullptr,0,parallel));
  }
  HIP_OK(hipGraphInstantiate(&batch,child?parent:parallel,nullptr,nullptr,0));
  for(unsigned attempt=0;attempt<2;++attempt){
   fail_command=1;command_calls=0;
   checked(hipGraphLaunch(batch,stream)==hipErrorOutOfMemory,__LINE__);
   HIP_OK(hipStreamSynchronize(stream));checked(submits==0&&completed_batch==0,__LINE__);
  }
  fail_command=-1;
  if(!child){
   HIP_OK(hipGraphNodeSetEnabled(batch,roots[0],0));
   HIP_OK(hipGraphNodeSetEnabled(batch,roots[1],0));
   HIP_OK(hipGraphLaunch(batch,stream));HIP_OK(hipStreamSynchronize(stream));
   checked(submits==0&&completed_batch==1,__LINE__);
  }
  HIP_OK(hipGraphExecDestroy(batch));
  if(parent)HIP_OK(hipGraphDestroy(parent));
  HIP_OK(hipGraphDestroy(parallel));
 }
 count=0;salt=999; // graph must own values, not this caller's argument storage.
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 HIP_OK(hipGraphNodeSetEnabled(exec,node,0));
 for(unsigned i=0;i<2;++i){HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));}
 checked(submits==0&&callbacks==2,__LINE__);callbacks=0;
 HIP_OK(hipGraphNodeSetEnabled(exec,node,1));
 for(int at=0;at<4;++at){
  count=73;salt=43;inject(at);
  checked(hipGraphExecKernelNodeSetParams(exec,node,&params)==hipErrorOutOfMemory,__LINE__);
  inject(at);checked(hipGraphKernelNodeSetParams(node,&params)==hipErrorOutOfMemory,__LINE__);
 }
 inject(-1);count=0;salt=999;
 for(unsigned round=0;round<2;++round){
  unsigned input[81],output[81];
  for(unsigned i=0;i<81;++i){input[i]=0xa5000000u+i*7+round*113;output[i]=0xdeadbeef;}
  HIP_OK(hipMemcpy(buffer,input,sizeof(input),hipMemcpyHostToDevice));
  if(round==1){
   count=73;salt=43;HIP_OK(hipGraphExecKernelNodeSetParams(exec,node,&params));
   count=0;salt=999;
   HIP_OK(hipGraphDestroy(graph));graph=nullptr;
  }
  if(round==1){
   fail_command=0;command_calls=0;
   checked(hipGraphLaunch(exec,stream)==hipErrorOutOfMemory,__LINE__);
   HIP_OK(hipStreamSynchronize(stream));checked(callbacks==1&&submits==1,__LINE__);
   fail_command=-1;
  }
  HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
  checked(callbacks==round+1,__LINE__);
  HIP_OK(hipMemcpy(output,buffer,sizeof(output),hipMemcpyDeviceToHost));
  for(unsigned i=0;i<81;++i)if(output[i]!=(i<73?i*17+19+round*24:input[i])){
   std::fprintf(stderr,"PUBLIC_GRAPH_GPU_OUTPUT_MISMATCH round=%u index=%u\n",round,i);checked(false,__LINE__);
  }
 }
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(buffer));HIP_OK(hipModuleUnload(module));
 checked(submits==2&&backend_calls==1,__LINE__);completed=true;
 std::puts("PUBLIC_GRAPH_GPU_PASS rounds=2 values=146 guards=16 owned_args=true updated=true source_destroyed=true gpu_execution=false");
}
