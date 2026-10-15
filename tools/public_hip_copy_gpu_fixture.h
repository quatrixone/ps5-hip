// Shared host/native copy/kernel graph oracle. Caller provides checked and HIP_OK.
#pragma once
#include "public_hip_copy_recovery_fixture.h"
namespace copy_gpu_fixture {
inline void run(const char* path,bool capture,
                void(*before_replay)(hipGraphExec_t,const hipGraphNode_t*)=nullptr){
 HIP_OK(hipInit(0));hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,path));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,81*sizeof(unsigned)));HIP_OK(hipMalloc(&b,81*sizeof(unsigned)));
 copy_recovery::capture_errors(stream,a,b,81*sizeof(unsigned));
 static unsigned input[81],output[81],eager[81],callbacks;callbacks=0;
 for(unsigned i=0;i<81;++i)input[i]=0x12340000u+i;
 unsigned count=73,salt=19;void* args[]={&a,&count,&salt};
 hipKernelNodeParams params{};params.func=reinterpret_cast<void*>(function);
 params.gridDim=dim3(2);params.blockDim=dim3(64);params.kernelParams=args;
 hipHostNodeParams host{[](void* p){++*static_cast<unsigned*>(p);},&callbacks};
 hipGraph_t graph=nullptr;hipGraphNode_t nodes[5]{};
 if(capture){
  HIP_OK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
  HIP_OK(hipMemcpyAsync(a,input,sizeof(input),hipMemcpyHostToDevice,stream));
  HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,stream,args,nullptr));
  HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,stream));
  HIP_OK(hipMemcpyAsync(output,b,sizeof(output),hipMemcpyDeviceToHost,stream));
  HIP_OK(hipLaunchHostFunc(stream,host.fn,host.userData));
  HIP_OK(hipStreamEndCapture(stream,&graph));
  size_t total=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&total));checked(total==5,__LINE__);
  size_t n=1;HIP_OK(hipGraphGetRootNodes(graph,&nodes[0],&n));checked(n==1,__LINE__);
  for(unsigned i=1;i<5;++i){n=1;HIP_OK(hipGraphNodeGetDependentNodes(nodes[i-1],&nodes[i],&n));checked(n==1,__LINE__);}
 }else{
  HIP_OK(hipGraphCreate(&graph,0));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[0],graph,nullptr,0,a,input,sizeof(input),hipMemcpyHostToDevice));
  HIP_OK(hipGraphAddKernelNode(&nodes[1],graph,&nodes[0],1,&params));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],graph,&nodes[1],1,b,a,sizeof(input),hipMemcpyDeviceToDevice));
  HIP_OK(hipGraphAddMemcpyNode1D(&nodes[3],graph,&nodes[2],1,output,b,sizeof(output),hipMemcpyDeviceToHost));
  HIP_OK(hipGraphAddHostNode(&nodes[4],graph,&nodes[3],1,&host));
 }
 count=0;salt=999;
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 checked(hipGraphMemcpyNodeSetParams1D(nodes[4],b,a,sizeof(input),hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[4],b,a,sizeof(input),hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 auto fake_node=reinterpret_cast<hipGraphNode_t>(uintptr_t(1));
 auto fake_exec=reinterpret_cast<hipGraphExec_t>(uintptr_t(1));
 checked(hipGraphMemcpyNodeSetParams1D(fake_node,b,a,sizeof(input),hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecMemcpyNodeSetParams1D(fake_exec,nodes[2],b,a,sizeof(input),hipMemcpyDeviceToDevice)==hipErrorInvalidValue,__LINE__);
 hipGraphNode_t error=nullptr;hipGraphExecUpdateResult update=hipGraphExecUpdateSuccess;
 checked(hipGraphExecUpdate(fake_exec,graph,&error,&update)==hipErrorInvalidValue,__LINE__);
 checked(hipGraphExecUpdate(exec,reinterpret_cast<hipGraph_t>(uintptr_t(1)),&error,&update)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipGraphExecUpdate(exec,graph,&error,&update));checked(update==hipGraphExecUpdateSuccess,__LINE__);
 HIP_OK(hipGraphNodeSetEnabled(exec,nodes[1],0));
 for(unsigned disabled:{0u,2u,3u}){
  HIP_OK(hipGraphNodeSetEnabled(exec,nodes[disabled],0));
  for(unsigned round=0;round<2;++round){
   HIP_OK(hipMemset(a,0x17,sizeof(input)));HIP_OK(hipMemset(b,0xa5,sizeof(input)));
   for(auto& value:output)value=0xddddddddu;
   HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
   checked(callbacks==(disabled==0?0:disabled==2?2:4)+round+1,__LINE__);
   for(auto value:output)checked(value==(disabled==0?0x17171717u:disabled==2?0xa5a5a5a5u:0xddddddddu),__LINE__);
  }
  HIP_OK(hipGraphNodeSetEnabled(exec,nodes[disabled],1));
 }
 HIP_OK(hipGraphNodeSetEnabled(exec,nodes[1],1));
 for(unsigned round=0;round<2;++round){
  for(unsigned i=0;i<81;++i){input[i]=0xa5000000u+i*7+round*113;output[i]=eager[i]=0xdeadbeefu;}
  count=73;salt=19+24*round;
  if(round==1){HIP_OK(hipGraphExecKernelNodeSetParams(exec,nodes[1],&params));HIP_OK(hipGraphDestroy(graph));graph=nullptr;}
  // Eager control uses the same data and public operations as the graph.
  HIP_OK(hipMemcpyAsync(a,input,sizeof(input),hipMemcpyHostToDevice,stream));
  HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,stream,args,nullptr));
  HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,stream));
  HIP_OK(hipMemcpyAsync(eager,b,sizeof(eager),hipMemcpyDeviceToHost,stream));
  HIP_OK(hipStreamSynchronize(stream));
  // Erase eager device results so every graph copy edge is observable.
  HIP_OK(hipMemset(a,0x6b,sizeof(input)));HIP_OK(hipMemset(b,0x91,sizeof(input)));
  for(auto& value:output)value=0xdeadbeefu;
  count=0;salt=999;
  if(before_replay)before_replay(exec,nodes);
  HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
  checked(callbacks==7+round,__LINE__);
  for(unsigned i=0;i<81;++i){
   const unsigned expected=i<73?i*17+19+24*round:input[i];
   if(output[i]!=expected||eager[i]!=expected){
    std::fprintf(stderr,"PUBLIC_COPY_GPU_OUTPUT_MISMATCH round=%u index=%u\n",round,i);checked(false,__LINE__);
   }
  }
 }
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(b));HIP_OK(hipFree(a));HIP_OK(hipModuleUnload(module));
}
}
