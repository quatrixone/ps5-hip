// Actual public HIP/ROCclr lifetime with modeled memory; no GPU execution.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include <thread>
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {~Final(){
 checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
 checked(observed.allocations==6&&observed.frees==6&&observed.launches==0&&backend_calls==0,__LINE__);
 checked(observed.queues==observed.retired_queues&&allocations.empty(),__LINE__);
 std::puts("PUBLIC_ALLOCATION_LIFETIME_FINAL ownership=clean gpu_execution=false");
}};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t s,int l){if(s!=hipSuccess){std::fprintf(stderr,"LIFETIME_ERROR line=%d status=%d\n",l,int(s));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
static void queued_free(hipStream_t stream){
 struct Gate {std::mutex lock;std::condition_variable cv;bool entered=false,released=false,free_entered=false,free_returned=false;} gate;
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));
 unsigned char input[128],output[128]{};for(unsigned i=0;i<128;++i)input[i]=i*7+3;
 HIP_OK(hipMemcpy(a,input,128,hipMemcpyHostToDevice));
 hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;hipGraphNode_t nodes[3]{};
 HIP_OK(hipGraphCreate(&graph,0));
 hipHostNodeParams host{[](void* p){auto& g=*static_cast<Gate*>(p);std::unique_lock<std::mutex> l(g.lock);
  g.entered=true;g.cv.notify_all();checked(g.cv.wait_for(l,std::chrono::seconds(5),[&]{return g.released;}),__LINE__);},&gate};
 HIP_OK(hipGraphAddHostNode(&nodes[0],graph,nullptr,0,&host));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[1],graph,&nodes[0],1,b,a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],graph,&nodes[1],1,output,b,128,hipMemcpyDeviceToHost));
 HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));HIP_OK(hipGraphLaunch(exec,stream));
 {std::unique_lock<std::mutex> l(gate.lock);checked(gate.cv.wait_for(l,std::chrono::seconds(5),[&]{return gate.entered;}),__LINE__);}
 std::thread freeing([&]{
  {std::lock_guard<std::mutex> l(gate.lock);gate.free_entered=true;gate.cv.notify_all();}
  HIP_OK(hipFree(a));
  std::lock_guard<std::mutex> l(gate.lock);checked(gate.released,__LINE__);gate.free_returned=true;gate.cv.notify_all();
 });
 {std::unique_lock<std::mutex> l(gate.lock);
  checked(gate.cv.wait_for(l,std::chrono::seconds(5),[&]{return gate.free_entered;}),__LINE__);
  // free_entered precedes the API call; this finite observation cannot prove runtime entry.
  checked(!gate.cv.wait_for(l,std::chrono::milliseconds(100),[&]{return gate.free_returned;}),__LINE__);
  gate.released=true;gate.cv.notify_all();
 }
 freeing.join();checked(gate.free_returned&&std::memcmp(input,output,128)==0,__LINE__);
 checked(amd::MemObjMap::FindMemObj(a)==nullptr,__LINE__);
 HIP_OK(hipFree(b));HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(graph));
 std::puts("PUBLIC_QUEUED_FREE_PASS gated_free_observed=true output_exact=true exec_alive_at_free=true gpu_execution=false");
}
static void migration(hipStream_t stream){
 void *a=nullptr,*b=nullptr,*next_a=nullptr,*next_b=nullptr;
 HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));HIP_OK(hipMalloc(&next_a,128));HIP_OK(hipMalloc(&next_b,128));
 unsigned char input[128],output[128]{};for(unsigned i=0;i<128;++i)input[i]=i*11+5;
 hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;hipGraphNode_t nodes[3]{};HIP_OK(hipGraphCreate(&graph,0));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[0],graph,nullptr,0,a,input,128,hipMemcpyHostToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[1],graph,&nodes[0],1,b,a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],graph,&nodes[1],1,output,b,128,hipMemcpyDeviceToHost));
 HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
 checked(std::memcmp(input,output,128)==0,__LINE__);
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[0],next_a,input,128,hipMemcpyHostToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[1],next_b,next_a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphExecMemcpyNodeSetParams1D(exec,nodes[2],output,next_b,128,hipMemcpyDeviceToHost));
 HIP_OK(hipFree(a));HIP_OK(hipFree(b));
 checked(!amd::MemObjMap::FindMemObj(a)&&!amd::MemObjMap::FindMemObj(b),__LINE__);
 for(unsigned i=0;i<128;++i)input[i]=i*13+17;std::memset(output,0,128);
 HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));checked(std::memcmp(input,output,128)==0,__LINE__);
 HIP_OK(hipFree(next_a));HIP_OK(hipFree(next_b));HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(graph));
 std::puts("PUBLIC_ALLOCATION_MIGRATION_PASS old_freed=true source_graph_alive=true replay_exact=true gpu_execution=false");
}
int main(){mode="normal";resume=true;HIP_OK(hipInit(0));hipStream_t stream=nullptr;
 HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));queued_free(stream);migration(stream);
 HIP_OK(hipStreamDestroy(stream));completed=true;return 0;}
