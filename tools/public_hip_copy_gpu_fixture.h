// Shared host/native copy/kernel graph oracle. Caller provides checked and HIP_OK.
#pragma once
#include "ps5hip/native_memory.h"
#include "public_hip_copy_recovery_fixture.h"
namespace copy_gpu_fixture {
inline void run(const char* path,bool capture,
                void(*before_replay)(hipGraphExec_t,const hipGraphNode_t*)=nullptr, bool joined=false){
 HIP_OK(hipInit(0));hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,path));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 hipStream_t peer=stream;hipEvent_t ready=nullptr,done=nullptr;
 if(joined){
  HIP_OK(hipStreamCreateWithFlags(&peer,hipStreamNonBlocking));
  HIP_OK(hipEventCreateWithFlags(&ready,hipEventDisableTiming));
  HIP_OK(hipEventCreateWithFlags(&done,hipEventDisableTiming));
 }
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
  if(joined){HIP_OK(hipEventRecord(ready,stream));HIP_OK(hipStreamWaitEvent(peer,ready,0));}
  HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,peer));
  HIP_OK(hipMemcpyAsync(output,b,sizeof(output),hipMemcpyDeviceToHost,peer));
  if(joined){HIP_OK(hipEventRecord(done,peer));HIP_OK(hipStreamWaitEvent(stream,done,0));}
  HIP_OK(hipLaunchHostFunc(stream,host.fn,host.userData));
  HIP_OK(hipStreamEndCapture(stream,&graph));
  if(joined){
   for(auto captured:{stream,peer}){hipStreamCaptureStatus state;HIP_OK(hipStreamIsCapturing(captured,&state));checked(state==hipStreamCaptureStatusNone,__LINE__);}
  }
  size_t total=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&total));checked(total==5,__LINE__);
  size_t n=1;HIP_OK(hipGraphGetRootNodes(graph,&nodes[0],&n));checked(n==1,__LINE__);
  for(unsigned i=1;i<5;++i){
   hipGraphNode_t next[2]{};n=2;HIP_OK(hipGraphNodeGetDependentNodes(nodes[i-1],next,&n));
   checked(n==(joined&&i==2?2u:1u),__LINE__);
   nodes[i]=next[0];
   if(joined&&i==2){hipGraphNodeType type;HIP_OK(hipGraphNodeGetType(next[0],&type));if(type==hipGraphNodeTypeHost)nodes[i]=next[1];}
   hipGraphNode_t dependencies[2]{};n=2;HIP_OK(hipGraphNodeGetDependencies(nodes[i],dependencies,&n));
   checked(n==(joined&&i==4?2u:1u),__LINE__);
   if(joined&&i==4)checked((dependencies[0]==nodes[1]&&dependencies[1]==nodes[3])||(dependencies[1]==nodes[1]&&dependencies[0]==nodes[3]),__LINE__);
   else checked(dependencies[0]==nodes[i-1],__LINE__);
  }
  n=0;HIP_OK(hipGraphNodeGetDependentNodes(nodes[4],nullptr,&n));checked(n==0,__LINE__);
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
  if(joined){HIP_OK(hipEventRecord(ready,stream));HIP_OK(hipStreamWaitEvent(peer,ready,0));}
  HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,peer));
  HIP_OK(hipMemcpyAsync(eager,b,sizeof(eager),hipMemcpyDeviceToHost,peer));
  if(joined){HIP_OK(hipEventRecord(done,peer));HIP_OK(hipStreamWaitEvent(stream,done,0));}
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
 HIP_OK(hipGraphExecDestroy(exec));
 if(joined){HIP_OK(hipEventDestroy(done));HIP_OK(hipEventDestroy(ready));HIP_OK(hipStreamDestroy(peer));}
 HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(b));HIP_OK(hipFree(a));HIP_OK(hipModuleUnload(module));
}
struct ReplayBacking {
 unsigned samples=0;
 uint64_t live_bytes=0,outstanding=0,peak_bytes=0,final_live_bytes=0,eager_allocations=0,eager_releases=0,graph_allocations=0,graph_releases=0;
};
inline ReplayBacking run_replay(const char* path,void(*before_replay)(hipGraphExec_t,const hipGraphNode_t*)=nullptr,
                               void(*before_sample)()=nullptr){
 const auto entry_backing=ps5hip::native_memory_stats();
 ReplayBacking backing;
 hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,path));hipFunction_t function=nullptr;
 HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 hipStream_t origin=nullptr,peer=nullptr;HIP_OK(hipStreamCreateWithFlags(&origin,hipStreamNonBlocking));
 HIP_OK(hipStreamCreateWithFlags(&peer,hipStreamNonBlocking));
 hipEvent_t ready=nullptr,done=nullptr;HIP_OK(hipEventCreateWithFlags(&ready,hipEventDisableTiming));HIP_OK(hipEventCreateWithFlags(&done,hipEventDisableTiming));
 static unsigned input[162],output[162],eager[162],callbacks;callbacks=0;
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,sizeof(input)));HIP_OK(hipMalloc(&b,sizeof(input)));
 void* regions[]={a,static_cast<unsigned*>(a)+81};unsigned count=73,salts[]={19,56};
 void* args0[]={&regions[0],&count,&salts[0]};void* args1[]={&regions[1],&count,&salts[1]};
 hipKernelNodeParams params[2]{};
 for(unsigned k=0;k<2;++k){params[k].func=reinterpret_cast<void*>(function);params[k].gridDim=dim3(2);params[k].blockDim=dim3(64);params[k].kernelParams=k?args1:args0;}
 HIP_OK(hipStreamBeginCapture(origin,hipStreamCaptureModeThreadLocal));
 HIP_OK(hipMemcpyAsync(a,input,sizeof(input),hipMemcpyHostToDevice,origin));
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,origin,args0,nullptr));
 HIP_OK(hipEventRecord(ready,origin));HIP_OK(hipStreamWaitEvent(peer,ready,0));
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,peer,args1,nullptr));
 HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,peer));
 HIP_OK(hipMemcpyAsync(output,b,sizeof(output),hipMemcpyDeviceToHost,peer));
 HIP_OK(hipEventRecord(done,peer));HIP_OK(hipStreamWaitEvent(origin,done,0));
 HIP_OK(hipLaunchHostFunc(origin,[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
 hipGraph_t graph=nullptr;HIP_OK(hipStreamEndCapture(origin,&graph));
 for(auto stream:{origin,peer}){hipStreamCaptureStatus state;HIP_OK(hipStreamIsCapturing(stream,&state));checked(state==hipStreamCaptureStatusNone,__LINE__);}
 size_t total=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&total));checked(total==6,__LINE__);
 hipGraphNode_t root=nullptr,kernels[2]{};size_t n=0;HIP_OK(hipGraphGetRootNodes(graph,nullptr,&n));checked(n==1,__LINE__);HIP_OK(hipGraphGetRootNodes(graph,&root,&n));
 HIP_OK(hipGraphNodeGetDependentNodes(root,nullptr,&n));checked(n==1,__LINE__);HIP_OK(hipGraphNodeGetDependentNodes(root,&kernels[0],&n));
 hipGraphNode_t next[2]{};n=0;HIP_OK(hipGraphNodeGetDependentNodes(kernels[0],nullptr,&n));checked(n==2,__LINE__);HIP_OK(hipGraphNodeGetDependentNodes(kernels[0],next,&n));
 for(auto node:next){hipGraphNodeType type;HIP_OK(hipGraphNodeGetType(node,&type));if(type==hipGraphNodeTypeKernel)kernels[1]=node;}
 checked(kernels[1]!=nullptr,__LINE__);
 hipGraphNode_t chain[]={root,kernels[0],kernels[1],nullptr,nullptr,nullptr};
 for(unsigned i=3;i<6;++i){n=0;HIP_OK(hipGraphNodeGetDependentNodes(chain[i-1],nullptr,&n));checked(n==1,__LINE__);HIP_OK(hipGraphNodeGetDependentNodes(chain[i-1],&chain[i],&n));}
 const hipGraphNodeType types[]={hipGraphNodeTypeMemcpy,hipGraphNodeTypeKernel,hipGraphNodeTypeKernel,hipGraphNodeTypeMemcpy,hipGraphNodeTypeMemcpy,hipGraphNodeTypeHost};
 for(unsigned i=0;i<6;++i){
  hipGraphNodeType type;HIP_OK(hipGraphNodeGetType(chain[i],&type));checked(type==types[i],__LINE__);
  hipGraphNode_t dependencies[2]{};n=0;HIP_OK(hipGraphNodeGetDependencies(chain[i],nullptr,&n));
  checked(n==(i==0?0u:i==5?2u:1u),__LINE__);
  HIP_OK(hipGraphNodeGetDependencies(chain[i],dependencies,&n));
  if(i==5)checked((dependencies[0]==kernels[0]&&dependencies[1]==chain[4])||(dependencies[1]==kernels[0]&&dependencies[0]==chain[4]),__LINE__);
  else if(i)checked(dependencies[0]==chain[i-1],__LINE__);
 }
 n=0;HIP_OK(hipGraphNodeGetDependentNodes(chain[5],nullptr,&n));checked(n==0,__LINE__);
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 constexpr unsigned warmup_rounds=4,verification_rounds=16;
 // Both phases check every value; this workload does not measure elapsed time.
 for(unsigned round=0;round<warmup_rounds+verification_rounds;++round){
  for(unsigned i=0;i<162;++i)input[i]=0xa5000000u+i*7+round*113;
  count=73;salts[0]=19+round*101;salts[1]=56+round*101;
  for(unsigned k=0;k<2;++k)HIP_OK(hipGraphExecKernelNodeSetParams(exec,kernels[k],&params[k]));
  if(round==19){HIP_OK(hipGraphDestroy(graph));graph=nullptr;}
  if(before_replay)before_replay(exec,kernels);
  for(unsigned pass=0;pass<2;++pass){
   const bool graph_pass=((round+pass)&1)!=0;
   HIP_OK(hipMemset(a,0x6b,sizeof(input)));HIP_OK(hipMemset(b,0x91,sizeof(input)));
   for(auto& value:output)value=0xdeadbeefu;for(auto& value:eager)value=0xdeadbeefu;
   const auto before_backing=ps5hip::native_memory_stats();
   if(graph_pass){
    count=0;salts[0]=salts[1]=999;HIP_OK(hipGraphLaunch(exec,origin));
   }else{
    count=73;salts[0]=19+round*101;salts[1]=56+round*101;
    HIP_OK(hipMemcpyAsync(a,input,sizeof(input),hipMemcpyHostToDevice,origin));
    HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,origin,args0,nullptr));
    HIP_OK(hipEventRecord(ready,origin));HIP_OK(hipStreamWaitEvent(peer,ready,0));
    HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,peer,args1,nullptr));
    HIP_OK(hipMemcpyAsync(b,a,sizeof(input),hipMemcpyDeviceToDevice,peer));
    HIP_OK(hipMemcpyAsync(eager,b,sizeof(eager),hipMemcpyDeviceToHost,peer));
    HIP_OK(hipEventRecord(done,peer));HIP_OK(hipStreamWaitEvent(origin,done,0));
   }
   HIP_OK(hipStreamSynchronize(origin));
   if(round>=warmup_rounds){
    if(before_sample)before_sample();
    const auto sample=ps5hip::native_memory_stats();
    const auto acquired=sample.allocations-before_backing.allocations,released=sample.releases-before_backing.releases;
    if(sample.live_bytes!=backing.live_bytes||sample.allocations-sample.releases!=backing.outstanding||
       !acquired||acquired!=released){
     std::fprintf(stderr,"PUBLIC_MULTI_KERNEL_REPLAY_BACKING_MISMATCH round=%u pass=%u\n",round,pass);checked(false,__LINE__);
    }
    ++backing.samples;backing.peak_bytes=sample.peak_bytes;
    if(graph_pass){backing.graph_allocations+=acquired;backing.graph_releases+=released;}
    else {backing.eager_allocations+=acquired;backing.eager_releases+=released;}
   }
   const auto* result=graph_pass?output:eager;
   for(unsigned i=0;i<162;++i){const unsigned region=i/81,local=i%81;
    const unsigned expected=local<73?local*17+19+round*101+region*37:input[i];
    if(result[i]!=expected){std::fprintf(stderr,"PUBLIC_MULTI_KERNEL_REPLAY_OUTPUT_MISMATCH round=%u index=%u\n",round,i);checked(false,__LINE__);}
   }
  }
  checked(callbacks==round+1,__LINE__);
  if(round+1==warmup_rounds){const auto baseline=ps5hip::native_memory_stats();
   backing.live_bytes=baseline.live_bytes;backing.outstanding=baseline.allocations-baseline.releases;}

 }
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipEventDestroy(done));HIP_OK(hipEventDestroy(ready));
 HIP_OK(hipStreamDestroy(peer));HIP_OK(hipStreamDestroy(origin));HIP_OK(hipFree(b));HIP_OK(hipFree(a));HIP_OK(hipModuleUnload(module));
 const auto exit_backing=ps5hip::native_memory_stats();
 checked(backing.samples==32&&exit_backing.live_bytes==entry_backing.live_bytes&&
         exit_backing.allocations-exit_backing.releases==entry_backing.allocations-entry_backing.releases,__LINE__);
 backing.final_live_bytes=exit_backing.live_bytes;
 return backing;
}
}
