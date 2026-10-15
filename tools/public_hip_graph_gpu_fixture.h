// Shared host/native GPU graph oracle. Caller provides checked and HIP_OK.
#pragma once
namespace graph_gpu_fixture {
inline void replay(hipGraph_t& graph,hipGraphExec_t exec,hipGraphNode_t node,
                   hipStream_t stream,void* buffer,hipKernelNodeParams& params,
                   unsigned& count,unsigned& salt,unsigned& callbacks,
                   void(*before_launch)(unsigned,hipStream_t)=nullptr){
 for(unsigned round=0;round<2;++round){
  static unsigned input[81],output[81];
  for(unsigned i=0;i<81;++i){input[i]=0xa5000000u+i*7+round*113;output[i]=0xdeadbeef;}
  HIP_OK(hipMemcpy(buffer,input,sizeof(input),hipMemcpyHostToDevice));
  if(round==1){
   count=73;salt=43;HIP_OK(hipGraphExecKernelNodeSetParams(exec,node,&params));
   count=0;salt=999;
   HIP_OK(hipGraphDestroy(graph));graph=nullptr;
  }
  if(before_launch)before_launch(round,stream);
  HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
  checked(callbacks==round+1,__LINE__);
  HIP_OK(hipMemcpy(output,buffer,sizeof(output),hipMemcpyDeviceToHost));
  for(unsigned i=0;i<81;++i)if(output[i]!=(i<73?i*17+19+round*24:input[i])){
   std::fprintf(stderr,"PUBLIC_GRAPH_GPU_OUTPUT_MISMATCH round=%u index=%u\n",round,i);checked(false,__LINE__);
  }
 }
}
inline void run(const char* path,bool capture=false){
 HIP_OK(hipInit(0));
 hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,path));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void* buffer=nullptr;HIP_OK(hipMalloc(&buffer,81*sizeof(unsigned)));
 unsigned count=73,salt=19;void* args[]={&buffer,&count,&salt};
 hipKernelNodeParams params{};params.func=reinterpret_cast<void*>(function);
 params.gridDim=dim3(2);params.blockDim=dim3(64);params.kernelParams=args;
 hipGraph_t graph=nullptr;hipGraphNode_t node=nullptr,host=nullptr;
 static unsigned callbacks=0;callbacks=0;
 hipHostNodeParams host_params{[](void* p){++*static_cast<unsigned*>(p);},&callbacks};
 if(capture){
  HIP_OK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
  HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,stream,args,nullptr));
  HIP_OK(hipLaunchHostFunc(stream,host_params.fn,host_params.userData));
  HIP_OK(hipStreamEndCapture(stream,&graph));
  size_t nodes=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&nodes));checked(nodes==2,__LINE__);
  size_t roots=1;HIP_OK(hipGraphGetRootNodes(graph,&node,&roots));checked(roots==1,__LINE__);
  size_t children=1;HIP_OK(hipGraphNodeGetDependentNodes(node,&host,&children));checked(children==1,__LINE__);
 }else{
  HIP_OK(hipGraphCreate(&graph,0));
  params.gridDim.x=0x40000002u;
  checked(hipGraphAddKernelNode(&node,graph,nullptr,0,&params)==hipErrorInvalidConfiguration,__LINE__);
  params.gridDim.x=2;
  HIP_OK(hipGraphAddKernelNode(&node,graph,nullptr,0,&params));
  HIP_OK(hipGraphAddHostNode(&host,graph,&node,1,&host_params));
 }
 count=0;salt=999;
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 checked(hipGraphExecKernelNodeSetParams(exec,host,&params)==hipErrorInvalidValue,__LINE__);
 HIP_OK(hipGraphNodeSetEnabled(exec,node,0));
 for(unsigned i=0;i<2;++i){HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));}
 checked(callbacks==2,__LINE__);callbacks=0;
 HIP_OK(hipGraphNodeSetEnabled(exec,node,1));
 replay(graph,exec,node,stream,buffer,params,count,salt,callbacks);
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(buffer));HIP_OK(hipModuleUnload(module));
}
}
