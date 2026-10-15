// Real public HIP/ROCclr path; only providers and GPU transport are modeled.
#define PS5HIP_MODEL_PAIRED_SALT
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false,combined_case=false;
struct Final {
 ~Final(){
  std::fprintf(stderr,"PUBLIC_GRAPH_GPU_COUNTS queues=%u retired=%u allocations=%u frees=%u\n",unsigned(observed.queues),unsigned(observed.retired_queues),unsigned(observed.allocations),unsigned(observed.frees));
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==(combined_case?4u:2u)&&observed.frees==observed.allocations&&observed.launches==(combined_case?84u:std::strncmp(mode,"replay",6)==0?80u:4u),__LINE__);
  const unsigned queues=combined_case?7:(std::strncmp(mode,"joined",6)==0||std::strncmp(mode,"replay",6)==0)?4:3;
  checked(observed.queues==queues&&observed.retired_queues==queues&&allocations.empty(),__LINE__);
  const auto backing=ps5hip::native_memory_stats();checked(backing.live_bytes==0&&backing.allocations==backing.releases,__LINE__);
  std::printf("PUBLIC_COPY_GPU_FINAL queues_retired=true allocations=%u frees=%u ownership=clean gpu_execution=false\n",unsigned(observed.allocations),unsigned(observed.frees));
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
static bool join_omitted=false;
static hipError_t joined_wait(hipStream_t stream,hipEvent_t event,unsigned flags){
 static unsigned waits=0;
 if(!std::strcmp(mode,"joined-missing-join")&&++waits==2){join_omitted=true;return hipSuccess;}
 return hipStreamWaitEvent(stream,event,flags);
}
static hipError_t joined_end(hipStream_t stream,hipGraph_t* graph){
 auto status=hipStreamEndCapture(stream,graph);
 if(join_omitted){
  checked(status==hipErrorStreamCaptureUnjoined,__LINE__);
  std::fputs("PUBLIC_COPY_GPU_UNJOINED_REJECTED\n",stderr);std::abort();
 }
 return status;
}
#define hipStreamWaitEvent joined_wait
#define hipStreamEndCapture joined_end
#include "public_hip_copy_gpu_fixture.h"
#undef hipStreamWaitEvent
#undef hipStreamEndCapture
static void report_backing(const copy_gpu_fixture::ReplayBacking& backing){
 std::printf("PUBLIC_REPLAY_BACKING samples=%u live_bytes=%llu live_allocations=%llu eager_allocations=%llu eager_releases=%llu graph_allocations=%llu graph_releases=%llu peak_bytes=%llu final_live_bytes=%llu gpu_execution=false\n",backing.samples,
 (unsigned long long)backing.live_bytes,(unsigned long long)backing.outstanding,
 (unsigned long long)backing.eager_allocations,(unsigned long long)backing.eager_releases,(unsigned long long)backing.graph_allocations,(unsigned long long)backing.graph_releases,
 (unsigned long long)backing.peak_bytes,(unsigned long long)backing.final_live_bytes);
}
int main(int argc,char**argv){
 checked(argc==3,__LINE__);mode=argv[2];resume=true;
 if(!std::strcmp(mode,"combined")){
  combined_case=true;mode="joined";copy_gpu_fixture::run(argv[1],true,nullptr,true);
  mode="replay-combined";const auto backing=copy_gpu_fixture::run_replay(argv[1]);report_backing(backing);
  checked(submits==84&&backend_calls==1,__LINE__);completed=true;
  std::puts("PUBLIC_COPY_GPU_COMBINED_PASS launches=84 allocations=4 frees=4 queues=7 gpu_execution=false");
  return 0;
 }
 if(!std::strncmp(mode,"replay",6)){
  HIP_OK(hipInit(0));const auto backing=copy_gpu_fixture::run_replay(argv[1],[](hipGraphExec_t exec,const hipGraphNode_t* nodes){
   if(!std::strcmp(mode,"replay-omit-first"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[0],0));
   if(!std::strcmp(mode,"replay-omit-second"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[1],0));
  },[]{if(!std::strcmp(mode,"replay-retained-backing")){
   static ps5hip::NativeMemory* retained=nullptr;
   if(!retained)retained=new ps5hip::NativeMemory(1,256,false);
  }});
  report_backing(backing);
  checked(submits==80&&backend_calls==1,__LINE__);completed=true;
  std::puts("PUBLIC_MULTI_KERNEL_REPLAY_PASS replay_rounds=20 warmup_rounds=4 verification_rounds=16 kernels_per_replay=2 replay_values=5840 replay_guards=640 alternating_order=true launches=80 gpu_execution=false");
  return 0;
 }
 copy_gpu_fixture::run(argv[1],std::strcmp(mode,"explicit")!=0,[](hipGraphExec_t exec,const hipGraphNode_t* nodes){
  if(!std::strcmp(mode,"copy-h2d"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[0],0));
  if(!std::strcmp(mode,"copy-d2d"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[2],0));
  if(!std::strcmp(mode,"copy-d2h"))HIP_OK(hipGraphNodeSetEnabled(exec,nodes[3],0));
 },std::strncmp(mode,"joined",6)==0);
 checked(submits==4&&backend_calls==1,__LINE__);completed=true;
 if(!std::strcmp(mode,"joined"))std::puts("PUBLIC_COPY_GPU_JOINED_PASS streams=2 rejoined=true events_reused=true");
 std::puts("PUBLIC_COPY_GPU_PASS rounds=2 launches=4 values=292 guards=32 recovery_cases=27 setter_rejections=4 update_rejections=2 eager_equal=true disabled_replays=6 callbacks=8 gpu_execution=false");
 return 0;
}
