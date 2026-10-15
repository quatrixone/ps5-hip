#pragma once
// Included after the copy model's HIP_OK and checked helpers.
static void nested_updates(hipStream_t stream,void* a,void* b){
 unsigned char input[128],alternate[128],output[128],target[128];
 std::memset(input,0x31,128);std::memset(alternate,0x72,128);
 std::memset(output,0,128);std::memset(target,0xdd,128);
 hipGraph_t child=nullptr,parent=nullptr;
 HIP_OK(hipGraphCreate(&child,0));HIP_OK(hipGraphCreate(&parent,0));
 hipGraphNode_t first=nullptr,last=nullptr,child_node=nullptr;
 HIP_OK(hipGraphAddMemcpyNode1D(&first,child,nullptr,0,a,input,128,hipMemcpyHostToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&last,child,&first,1,output,a,128,hipMemcpyDeviceToHost));
 HIP_OK(hipGraphAddChildGraphNode(&child_node,parent,nullptr,0,child));
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,parent,nullptr,nullptr,0));
 hipGraph_t changed=nullptr;HIP_OK(hipGraphCreate(&changed,0));
 HIP_OK(hipGraphAddMemcpyNode1D(&first,changed,nullptr,0,a,alternate,128,hipMemcpyHostToDevice));
 hipMemcpy3DParms params{};params.srcPtr=make_hipPitchedPtr(a,128,128,1);
 params.dstPtr=make_hipPitchedPtr(target,128,128,1);params.extent=make_hipExtent(128,1,1);params.kind=hipMemcpyDeviceToHost;
 // Public 3D graph creation is unsupported; use its actual registered representation.
 auto* three_d=new hip::GraphMemcpyNode(&params);
 reinterpret_cast<hip::Graph*>(changed)->AddNode(three_d);
 reinterpret_cast<hip::GraphNode*>(first)->AddEdge(three_d);
 last=reinterpret_cast<hipGraphNode_t>(three_d);
 for(int scenario=0;scenario<3;++scenario){
  hipGraph_t replacement=nullptr;HIP_OK(hipGraphCreate(&replacement,0));
  hipGraphNode_t replacement_child=nullptr;
  HIP_OK(hipGraphAddChildGraphNode(&replacement_child,replacement,nullptr,0,changed));
  hipGraphNode_t error=nullptr;hipGraphExecUpdateResult result=hipGraphExecUpdateSuccess;
  auto status=hipGraphExecUpdate(exec,replacement,&error,&result);
  if(scenario<2){
   checked(status==hipErrorGraphExecUpdateFailure,__LINE__);
   checked(result==(scenario==0?hipGraphExecUpdateErrorNodeTypeChanged:hipGraphExecUpdateErrorTopologyChanged),__LINE__);
   checked(error==replacement_child,__LINE__);
   std::memset(output,0,128);
   HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
   checked(std::memcmp(output,input,128)==0,__LINE__);
   for(auto v:target)checked(v==0xdd,__LINE__);
  }else{
   HIP_OK(status);checked(result==hipGraphExecUpdateSuccess,__LINE__);
   HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
   checked(std::memcmp(target,alternate,128)==0,__LINE__);
  }
  HIP_OK(hipGraphDestroy(replacement));
  if(scenario==0)HIP_OK(hipGraphDestroyNode(last));
  if(scenario==1)HIP_OK(hipGraphAddMemcpyNode1D(&last,changed,&first,1,target,a,128,hipMemcpyDeviceToHost));
 }
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(parent));
 HIP_OK(hipGraphDestroy(child));HIP_OK(hipGraphDestroy(changed));
 // Driver memcpy nodes share the public enum but not GraphMemcpyNode's C++ base.
 hipGraph_t driver=nullptr,replacement=nullptr;
 HIP_OK(hipGraphCreate(&driver,0));HIP_OK(hipGraphCreate(&replacement,0));
 HIP_MEMCPY3D driver_params{};driver_params.srcMemoryType=hipMemoryTypeHost;
 driver_params.dstMemoryType=hipMemoryTypeHost;driver_params.srcHost=input;driver_params.dstHost=output;
 driver_params.srcPitch=driver_params.dstPitch=128;driver_params.srcHeight=driver_params.dstHeight=1;
 driver_params.WidthInBytes=128;driver_params.Height=driver_params.Depth=1;
 auto* old_driver=new hip::GraphDrvMemcpyNode(&driver_params);
 reinterpret_cast<hip::Graph*>(driver)->AddNode(old_driver);
 HIP_OK(hipGraphInstantiate(&exec,driver,nullptr,nullptr,0));
 driver_params.srcHost=alternate;driver_params.dstHost=target;
 auto* new_driver=new hip::GraphDrvMemcpyNode(&driver_params);
 reinterpret_cast<hip::Graph*>(replacement)->AddNode(new_driver);
 hipGraphNode_t error=nullptr;hipGraphExecUpdateResult result=hipGraphExecUpdateError;
 HIP_OK(hipGraphExecUpdate(exec,replacement,&error,&result));
 checked(result==hipGraphExecUpdateSuccess,__LINE__);std::memset(target,0,128);
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(target,alternate,128)==0,__LINE__);
 driver_params.dstMemoryType=hipMemoryTypeDevice;driver_params.dstDevice=reinterpret_cast<hipDeviceptr_t>(b);
 HIP_OK(new_driver->SetParams(&driver_params));
 checked(hipGraphExecUpdate(exec,replacement,&error,&result)==hipErrorGraphExecUpdateFailure,__LINE__);
 checked(result==hipGraphExecUpdateErrorParametersChanged&&error==reinterpret_cast<hipGraphNode_t>(new_driver),__LINE__);
 std::memset(target,0,128);HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(target,alternate,128)==0,__LINE__);
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(driver));HIP_OK(hipGraphDestroy(replacement));
 std::puts("PUBLIC_NESTED_UPDATE_PASS child_type_and_size_rejected=true compatible_child_replay=true internal_driver_update=true driver_kind_rejected=true gpu_execution=false");
}
