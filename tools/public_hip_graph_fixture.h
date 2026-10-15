// Shared host/native oracle. Caller supplies observed, checked and HIP_OK.
#pragma once
#include <cstring>
#include <initializer_list>
struct Values { unsigned seed=0, middle=0, output=0, produced=0, consumed=0; };
static void produce(void* ptr){auto& v=*static_cast<Values*>(ptr);v.middle=v.seed*3; ++v.produced;}
static void noop(void* ptr){++static_cast<Values*>(ptr)->produced;}
static void alternate(void* ptr){auto& v=*static_cast<Values*>(ptr);v.middle=v.seed*7; ++v.produced;}
static void consume(void* ptr){auto& v=*static_cast<Values*>(ptr);v.output=v.middle+5; ++v.consumed;}
static void run_graph_fixture(const char* mode,unsigned expected_queues,bool log){
 checked((std::strcmp(mode,"explicit")==0||std::strcmp(mode,"capture")==0||std::strcmp(mode,"explicit-noop")==0||std::strcmp(mode,"recover")==0),__LINE__);
 const bool recover=std::strcmp(mode,"recover")==0;
 const bool capture=std::strcmp(mode,"capture")==0||recover;
 HIP_OK(hipInit(0));
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 if(recover){
  static unsigned callbacks=0;callbacks=0;
  for(auto mode:{hipStreamCaptureModeGlobal,hipStreamCaptureModeThreadLocal,hipStreamCaptureModeRelaxed}){
   for(bool external:{false,true}){
    hipGraph_t supplied=nullptr;
    if(external){
     HIP_OK(hipGraphCreate(&supplied,0));
     HIP_OK(hipStreamBeginCaptureToGraph(stream,supplied,nullptr,nullptr,0,mode));
    }else HIP_OK(hipStreamBeginCapture(stream,mode));
    HIP_OK(hipLaunchHostFunc(stream,[](void* p){++*static_cast<unsigned*>(p);},&callbacks));
    checked(hipStreamSynchronize(stream)==hipErrorStreamCaptureUnsupported,__LINE__);
    hipStreamCaptureStatus status;
    HIP_OK(hipStreamIsCapturing(stream,&status));checked(status==hipStreamCaptureStatusInvalidated,__LINE__);
    checked(hipStreamBeginCapture(stream,mode)==hipErrorIllegalState,__LINE__);
    hipGraph_t failed=reinterpret_cast<hipGraph_t>(0x12345);
    checked(hipStreamEndCapture(stream,&failed)==hipErrorStreamCaptureInvalidated&&failed==nullptr,__LINE__);
    HIP_OK(hipStreamIsCapturing(stream,&status));checked(status==hipStreamCaptureStatusNone,__LINE__);
    failed=reinterpret_cast<hipGraph_t>(0x12345);
    checked(hipStreamEndCapture(stream,&failed)==hipErrorIllegalState&&failed==nullptr,__LINE__);
    HIP_OK(hipStreamSynchronize(stream));checked(callbacks==0,__LINE__);
    if(external){size_t count=0;HIP_OK(hipGraphGetNodes(supplied,nullptr,&count));checked(count==1,__LINE__);HIP_OK(hipGraphDestroy(supplied));}
   }
  }
  if(log)std::puts("PUBLIC_CAPTURE_RECOVERY_PASS invalidations=6 modes=3 external_graphs=3 callbacks=0 stream_reusable=true");
 }
 static Values original,updated;
 original={};updated={};
 hipGraph_t graph=nullptr;hipGraphNode_t first=nullptr,second=nullptr;
 hipHostNodeParams a{std::strcmp(mode,"explicit-noop")==0?noop:produce,&original},b{consume,&original};
 if(capture){
  HIP_OK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
  HIP_OK(hipLaunchHostFunc(stream,produce,&original));
  HIP_OK(hipLaunchHostFunc(stream,consume,&original));
  HIP_OK(hipStreamEndCapture(stream,&graph));
  size_t count=0;HIP_OK(hipGraphGetRootNodes(graph,nullptr,&count));checked(count==1,__LINE__);
  HIP_OK(hipGraphGetRootNodes(graph,&first,&count));
  count=0;HIP_OK(hipGraphNodeGetDependentNodes(first,nullptr,&count));checked(count==1,__LINE__);
  HIP_OK(hipGraphNodeGetDependentNodes(first,&second,&count));
 }else{
  HIP_OK(hipGraphCreate(&graph,0));
  HIP_OK(hipGraphAddHostNode(&first,graph,nullptr,0,&a));
  HIP_OK(hipGraphAddHostNode(&second,graph,&first,1,&b));
 }
 hipGraphExec_t exec=nullptr;HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 checked(original.produced==0&&original.consumed==0&&updated.produced==0&&updated.consumed==0,__LINE__);
 for(unsigned round=0;round<16;++round){
  if(round==4){
   a={alternate,&updated};b={consume,&updated};
   HIP_OK(hipGraphExecHostNodeSetParams(exec,first,&a));
   HIP_OK(hipGraphExecHostNodeSetParams(exec,second,&b));
  }
  if(round==8){HIP_OK(hipGraphDestroy(graph));graph=nullptr;}
  auto& v=round<4?original:updated;
  const unsigned before=v.produced;
  v.seed=round*11+3;v.middle=0xabcdef;v.output=0x123456;
  HIP_OK(hipGraphLaunch(exec,stream));HIP_OK(hipStreamSynchronize(stream));
  if(v.middle!=v.seed*(round<4?3:7)||v.output!=v.middle+5){
   std::fprintf(stderr,"PUBLIC_GRAPH_OUTPUT_MISMATCH round=%u\n",round);checked(false,__LINE__);
  }
  checked(v.produced==before+1&&v.consumed==v.produced,__LINE__);
  checked(observed.allocations==0&&observed.frees==0&&observed.queues==expected_queues,__LINE__);
  if(round>=4)checked(original.produced==4&&original.consumed==4&&original.output==(3*11+3)*3+5,__LINE__);
 }
 checked(original.produced==4&&updated.produced==12,__LINE__);
 HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipStreamDestroy(stream));
 if(log)std::printf("PUBLIC_GRAPH_PASS mode=%s replays=16 callbacks=32 parameter_update=true source_destroyed=true gpu_execution=false\n",capture?"capture":"explicit");
}
