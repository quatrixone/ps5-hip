// Genuine public HIP admission while an earlier native packet holds the GPU gate.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final { ~Final(){
 checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
 checked(observed.allocations==2&&observed.frees==2&&observed.launches==2&&submits==2&&allocations.empty(),__LINE__);
 std::puts("PUBLIC_POINTER_FINAL ownership=clean gpu_execution=false");
}};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t s,int line){if(s!=hipSuccess){std::fprintf(stderr,"POINTER_ERROR line=%d status=%d\n",line,int(s));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
int main(int argc,char**argv){
 checked(argc==2,__LINE__);HIP_OK(hipInit(0));
 hipModule_t module=nullptr;HIP_OK(hipModuleLoad(&module,argv[1]));
 hipFunction_t function=nullptr;HIP_OK(hipModuleGetFunction(&function,module,"thread_words"));
 void* buffer=nullptr;HIP_OK(hipMalloc(&buffer,328));void* freed=nullptr;HIP_OK(hipMalloc(&freed,16));HIP_OK(hipFree(freed));
 void* output=buffer;unsigned count=73,salt=19;void* args[]={&output,&count,&salt};
 hipGraph_t graph=nullptr;HIP_OK(hipGraphCreate(&graph,0));
 hipKernelNodeParams params{};params.func=reinterpret_cast<void*>(function);params.gridDim=dim3(2);params.blockDim=dim3(64);params.kernelParams=args;
 hipGraphNode_t node=nullptr;HIP_OK(hipGraphAddKernelNode(&node,graph,nullptr,0,&params));
 for(void* invalid:{reinterpret_cast<void*>(1),static_cast<void*>(static_cast<char*>(buffer)+328),freed}){
  output=invalid;
  checked(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,args,nullptr)==hipErrorInvalidValue,__LINE__);
  checked(hipGraphKernelNodeSetParams(node,&params)==hipErrorInvalidValue,__LINE__);
  hipGraphNode_t absent=nullptr;checked(hipGraphAddKernelNode(&absent,graph,nullptr,0,&params)==hipErrorInvalidValue,__LINE__);
 }
 output=nullptr;HIP_OK(hipGraphKernelNodeSetParams(node,&params));
 output=static_cast<char*>(buffer)+4;HIP_OK(hipGraphKernelNodeSetParams(node,&params));HIP_OK(hipGraphDestroy(graph));
 checked(submits==0&&backend_calls==0&&!observed.terminal,__LINE__);
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,args,nullptr));
 {std::unique_lock<std::mutex> lock(mutex);checked(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),__LINE__);}
 // The transport cannot complete until this second asynchronous enqueue returns.
 HIP_OK(hipModuleLaunchKernel(function,2,1,1,64,1,1,0,nullptr,args,nullptr));
 {std::lock_guard<std::mutex> lock(mutex);checked(entered&&!resume&&submits==1,__LINE__);resume=true;changed.notify_all();}
 HIP_OK(hipDeviceSynchronize());unsigned values[73];HIP_OK(hipMemcpy(values,output,sizeof(values),hipMemcpyDeviceToHost));
 for(unsigned i=0;i<73;++i)checked(values[i]==i*17+19,__LINE__);
 HIP_OK(hipFree(buffer));HIP_OK(hipModuleUnload(module));completed=true;
 std::puts("PUBLIC_POINTER_PASS unknown_onepast_freed_rejected=true null_interior_graph=true stalled_second_enqueue=true gpu_execution=false");
}
