// Shared native/model allocation retirement oracle. Caller supplies checked and HIP_OK.
#pragma once
// ROCclr's memory declarations require the command header's include order.
#include "platform/command.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <pthread.h>
namespace allocation_lifetime_fixture {
struct Gate {std::mutex lock;std::condition_variable cv;bool entered=false,released=false,free_entered=false,free_returned=false;void* source=nullptr;};
inline void* free_source(void* p){
 try {
 auto& gate=*static_cast<Gate*>(p);
 {std::lock_guard<std::mutex> l(gate.lock);gate.free_entered=true;gate.cv.notify_all();}
 HIP_OK(hipFree(gate.source));
 std::lock_guard<std::mutex> l(gate.lock);checked(gate.released,__LINE__);gate.free_returned=true;gate.cv.notify_all();
 return nullptr;
 }catch(...){checked(false,__LINE__);return nullptr;}
}
inline void queued_free(hipStream_t stream){
 static Gate gate;
 {std::lock_guard<std::mutex> l(gate.lock);gate.entered=gate.released=gate.free_entered=gate.free_returned=false;}
 void *a=nullptr,*b=nullptr;HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));
 static unsigned char input[128],output[128];std::memset(output,0,128);for(unsigned i=0;i<128;++i)input[i]=i*7+3;
 HIP_OK(hipMemcpy(a,input,128,hipMemcpyHostToDevice));
 hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;hipGraphNode_t nodes[3]{};
 HIP_OK(hipGraphCreate(&graph,0));
 hipHostNodeParams host{[](void* p){try {auto& g=*static_cast<Gate*>(p);std::unique_lock<std::mutex> l(g.lock);
  g.entered=true;g.cv.notify_all();checked(g.cv.wait_for(l,std::chrono::seconds(5),[&]{return g.released;}),__LINE__);
 }catch(...){checked(false,__LINE__);}},&gate};
 HIP_OK(hipGraphAddHostNode(&nodes[0],graph,nullptr,0,&host));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[1],graph,&nodes[0],1,b,a,128,hipMemcpyDeviceToDevice));
 HIP_OK(hipGraphAddMemcpyNode1D(&nodes[2],graph,&nodes[1],1,output,b,128,hipMemcpyDeviceToHost));
 HIP_OK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));HIP_OK(hipGraphLaunch(exec,stream));
 {std::unique_lock<std::mutex> l(gate.lock);checked(gate.cv.wait_for(l,std::chrono::seconds(5),[&]{return gate.entered;}),__LINE__);}
 gate.source=a;pthread_t freeing{};
 const int create_status=pthread_create(&freeing,nullptr,free_source,&gate);
 if(create_status)std::fputs("PUBLIC_ALLOCATION_THREAD_FAILURE operation=create\n",stderr);
 checked(create_status==0,__LINE__);
 {std::unique_lock<std::mutex> l(gate.lock);
  checked(gate.cv.wait_for(l,std::chrono::seconds(5),[&]{return gate.free_entered;}),__LINE__);
  // free_entered precedes the API call; this finite observation cannot prove runtime entry.
  checked(!gate.cv.wait_for(l,std::chrono::milliseconds(100),[&]{return gate.free_returned;}),__LINE__);
  gate.released=true;gate.cv.notify_all();
 }
 {std::unique_lock<std::mutex> l(gate.lock);checked(gate.cv.wait_for(l,std::chrono::seconds(5),[&]{return gate.free_returned;}),__LINE__);}
 const int join_status=pthread_join(freeing,nullptr);
 if(join_status)std::fputs("PUBLIC_ALLOCATION_THREAD_FAILURE operation=join\n",stderr);
 checked(join_status==0,__LINE__);checked(std::memcmp(input,output,128)==0,__LINE__);
 checked(amd::MemObjMap::FindMemObj(a)==nullptr,__LINE__);
 HIP_OK(hipFree(b));HIP_OK(hipGraphExecDestroy(exec));HIP_OK(hipGraphDestroy(graph));
}
inline void migration(hipStream_t stream){
 void *a=nullptr,*b=nullptr,*next_a=nullptr,*next_b=nullptr;
 HIP_OK(hipMalloc(&a,128));HIP_OK(hipMalloc(&b,128));HIP_OK(hipMalloc(&next_a,128));HIP_OK(hipMalloc(&next_b,128));
 static unsigned char input[128],output[128];std::memset(output,0,128);for(unsigned i=0;i<128;++i)input[i]=i*11+5;
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
}
inline void run(){
 hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 queued_free(stream);migration(stream);HIP_OK(hipStreamDestroy(stream));
}
}
