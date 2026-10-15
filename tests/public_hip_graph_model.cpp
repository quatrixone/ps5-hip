// Public HIP host-node graph replay, updates and lifetime; no GPU submission.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
static bool combined=false;
static unsigned expected_queues=3;

struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==0&&observed.frees==0&&observed.launches==0&&backend_calls==0,__LINE__);
  checked(observed.queues==expected_queues&&observed.retired_queues==expected_queues&&allocations.empty(),__LINE__);
  std::printf("PUBLIC_GRAPH_FINAL queues=%u retired=%u allocations=0 frees=0 ownership=clean gpu_execution=false\n",expected_queues,expected_queues);
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_GRAPH_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_graph_fixture.h"

static void session_recovery(bool rejoin=false){
 HIP_OK(hipInit(0));
 hipStream_t streams[4]{};
 hipEvent_t events[4]{};
 hipEvent_t ordinary=nullptr;HIP_OK(hipEventCreateWithFlags(&ordinary,hipEventDisableTiming));
 unsigned callbacks=0;
 for(auto& stream:streams)HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 for(auto& event:events)HIP_OK(hipEventCreateWithFlags(&event,hipEventDisableTiming));
 for(auto mode:{hipStreamCaptureModeGlobal,hipStreamCaptureModeThreadLocal,hipStreamCaptureModeRelaxed}){
  for(bool external:{false,true}){
   for(unsigned victim=0;victim<4;++victim){
    hipGraph_t supplied=nullptr;
    if(external){HIP_OK(hipGraphCreate(&supplied,0));HIP_OK(hipStreamBeginCaptureToGraph(streams[0],supplied,nullptr,nullptr,0,mode));}
    else HIP_OK(hipStreamBeginCapture(streams[0],mode));
    HIP_OK(hipLaunchHostFunc(streams[0],[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
    HIP_OK(hipEventRecord(events[0],streams[0]));
    HIP_OK(hipStreamWaitEvent(streams[1],events[0],0));
    HIP_OK(hipStreamWaitEvent(streams[2],events[0],0));
    HIP_OK(hipLaunchHostFunc(streams[1],[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
    HIP_OK(hipEventRecord(events[1],streams[1]));
    HIP_OK(hipStreamWaitEvent(streams[3],events[1],0));
    for(unsigned i:{2u,3u})HIP_OK(hipLaunchHostFunc(streams[i],[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
    if(rejoin){
     HIP_OK(hipEventRecord(events[2],streams[3]));
     HIP_OK(hipStreamWaitEvent(streams[1],events[2],0));
     HIP_OK(hipEventRecord(events[3],streams[2]));
     HIP_OK(hipStreamWaitEvent(streams[0],events[2],0));
     HIP_OK(hipStreamWaitEvent(streams[0],events[3],0));
    }
    checked(hipStreamSynchronize(streams[victim])==hipErrorStreamCaptureUnsupported,__LINE__);
    if(rejoin){
     checked(hipStreamWaitEvent(streams[1],events[2],0)==hipErrorStreamCaptureInvalidated,__LINE__);
     checked(hipStreamWaitEvent(streams[1],ordinary,0)==hipErrorStreamCaptureInvalidated,__LINE__);
    }
    for(auto stream:streams){hipStreamCaptureStatus state;HIP_OK(hipStreamIsCapturing(stream,&state));checked(state==hipStreamCaptureStatusInvalidated,__LINE__);}
    hipGraph_t failed=reinterpret_cast<hipGraph_t>(0x12345);
    checked(hipStreamEndCapture(streams[0],&failed)==hipErrorStreamCaptureInvalidated&&failed==nullptr,__LINE__);
    for(auto stream:streams){
     hipStreamCaptureStatus state;HIP_OK(hipStreamIsCapturing(stream,&state));checked(state==hipStreamCaptureStatusNone,__LINE__);
     HIP_OK(hipStreamSynchronize(stream));
     // Each former child must be able to become a fresh origin.
     HIP_OK(hipStreamBeginCapture(stream,mode));hipGraph_t reused=nullptr;
     HIP_OK(hipStreamEndCapture(stream,&reused));HIP_OK(hipGraphDestroy(reused));
    }
    checked(callbacks==0,__LINE__);
    if(external){size_t count=0;HIP_OK(hipGraphGetNodes(supplied,nullptr,&count));checked(count==4,__LINE__);HIP_OK(hipGraphDestroy(supplied));}
   }
  }
 }
 if(rejoin){
  for(auto mode:{hipStreamCaptureModeGlobal,hipStreamCaptureModeThreadLocal,hipStreamCaptureModeRelaxed}){
   HIP_OK(hipStreamBeginCapture(streams[0],mode));HIP_OK(hipStreamBeginCapture(streams[1],mode));
   HIP_OK(hipEventRecord(events[0],streams[0]));
   checked(hipStreamWaitEvent(streams[1],events[0],0)==hipErrorStreamCaptureMerge,__LINE__);
   for(unsigned i:{0u,1u}){
    hipStreamCaptureStatus state;HIP_OK(hipStreamIsCapturing(streams[i],&state));checked(state==hipStreamCaptureStatusInvalidated,__LINE__);
    hipGraph_t graph=reinterpret_cast<hipGraph_t>(0x12345);
    checked(hipStreamEndCapture(streams[i],&graph)==hipErrorStreamCaptureInvalidated&&graph==nullptr,__LINE__);
    HIP_OK(hipStreamIsCapturing(streams[i],&state));checked(state==hipStreamCaptureStatusNone,__LINE__);
   }
  }
 }

 if(rejoin){
  struct Replay {unsigned input=0,seed=0,left=0,right=0,output=0;} values;
  HIP_OK(hipStreamBeginCapture(streams[0],hipStreamCaptureModeThreadLocal));
  HIP_OK(hipLaunchHostFunc(streams[0],[](void* p){auto& v=*static_cast<Replay*>(p);v.seed=v.input;},&values));
  HIP_OK(hipEventRecord(events[0],streams[0]));
  checked(hipStreamWaitEvent(streams[1],events[0],1)==hipErrorInvalidValue,__LINE__);
  hipStreamCaptureStatus untouched;HIP_OK(hipStreamIsCapturing(streams[1],&untouched));checked(untouched==hipStreamCaptureStatusNone,__LINE__);
  HIP_OK(hipStreamWaitEvent(streams[1],events[0],0));
  HIP_OK(hipLaunchHostFunc(streams[1],[](void* p){auto& v=*static_cast<Replay*>(p);v.left=v.seed*3;},&values));
  HIP_OK(hipEventRecord(events[1],streams[1]));
  HIP_OK(hipStreamWaitEvent(streams[3],events[1],0));
  HIP_OK(hipLaunchHostFunc(streams[3],[](void* p){auto& v=*static_cast<Replay*>(p);v.left*=5;},&values));
  HIP_OK(hipEventRecord(events[2],streams[3]));
  HIP_OK(hipStreamWaitEvent(streams[1],events[2],0));
  HIP_OK(hipStreamWaitEvent(streams[2],events[0],0));
  HIP_OK(hipLaunchHostFunc(streams[2],[](void* p){auto& v=*static_cast<Replay*>(p);v.right=v.seed*7;},&values));
  HIP_OK(hipEventRecord(events[3],streams[2]));
  HIP_OK(hipStreamWaitEvent(streams[0],events[2],0));HIP_OK(hipStreamWaitEvent(streams[0],events[3],0));
  HIP_OK(hipLaunchHostFunc(streams[0],[](void* p){auto& v=*static_cast<Replay*>(p);v.output=v.left+v.right;},&values));
  hipGraph_t graph=nullptr;HIP_OK(hipStreamEndCapture(streams[0],&graph));
  size_t nodes=0;HIP_OK(hipGraphGetNodes(graph,nullptr,&nodes));checked(nodes==5,__LINE__);
  hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));HIP_OK(hipGraphDestroy(graph));
  for(unsigned round=0;round<8;++round){
   values={round*13+1,0xabcdef,0x123456,0x987654,0xfedcba};
   HIP_OK(hipGraphLaunch(exec,streams[0]));HIP_OK(hipStreamSynchronize(streams[0]));
   checked(values.seed==values.input&&values.left==values.input*15&&values.right==values.input*7&&values.output==values.input*22,__LINE__);
  }
  HIP_OK(hipGraphExecDestroy(exec));
 }
 // Former captured events must also work as ordinary queue dependencies.
 HIP_OK(hipLaunchHostFunc(streams[0],[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
 HIP_OK(hipEventRecord(events[0],streams[0]));
 HIP_OK(hipStreamWaitEvent(streams[1],events[0],0));
 HIP_OK(hipEventRecord(events[1],streams[1]));
 HIP_OK(hipStreamWaitEvent(streams[3],events[1],0));
 HIP_OK(hipStreamSynchronize(streams[3]));checked(callbacks==1,__LINE__);
 if(rejoin){
  HIP_OK(hipEventRecord(events[2],streams[3]));HIP_OK(hipStreamWaitEvent(streams[2],events[2],0));
  HIP_OK(hipEventRecord(events[3],streams[2]));HIP_OK(hipStreamWaitEvent(streams[0],events[3],0));
  HIP_OK(hipStreamSynchronize(streams[0]));checked(callbacks==1,__LINE__);
 }
 HIP_OK(hipEventDestroy(ordinary));
 for(auto event:events)HIP_OK(hipEventDestroy(event));
 for(auto stream:streams)HIP_OK(hipStreamDestroy(stream));
 if(rejoin)std::puts("PUBLIC_CAPTURE_REJOIN_PASS invalidations=24 streams=4 modes=3 external_graphs=12 merges_rejected=3 replays=8 replay_nodes=5 ordinary_callbacks=1 events_reused=4");
 else std::puts("PUBLIC_CAPTURE_SESSION_PASS invalidations=24 streams=4 modes=3 external_graphs=12 callbacks=1 events_reusable=true");
}

int main(int argc,char**argv){
 checked(argc==2,__LINE__);combined=std::strcmp(argv[1],"combined")==0;
 if(std::strcmp(argv[1],"session")==0){expected_queues=4;session_recovery();completed=true;return 0;}
 if(std::strcmp(argv[1],"rejoin")==0){expected_queues=7;session_recovery(true);completed=true;return 0;}
 if(combined){expected_queues=5;
  run_graph_fixture("explicit",3,false);run_graph_fixture("recover",5,false);
  checked(observed.queues==5&&observed.retired_queues==4,__LINE__);
  std::puts("PUBLIC_GRAPH_COMBINED_PASS graphs=2 replays=32 callbacks=64 invalidations=6 gpu_execution=false");
 }else run_graph_fixture(argv[1],3,true);
 completed=true;
}
