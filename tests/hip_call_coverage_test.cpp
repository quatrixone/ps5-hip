// Real backend and pinned GGML callbacks; HIP transport is an injected host test double.
#include "ps5_backend.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
static hipError_t init_error=hipSuccess,count_error=hipSuccess,alloc_error=hipSuccess,
 free_error=hipSuccess,copy_error=hipSuccess,fill_error=hipSuccess,sync_error=hipSuccess;
static bool fail_buffer_init;
static std::map<void*,size_t> owners;
static bool block_copy=false,release_copy=false,throw_copy=false;
static unsigned entered_copies=0;
static std::mutex copy_mutex;
static std::condition_variable copy_changed;
struct Failure{};
static void entered(m0::HipCall call){
 const auto r=m0::hip_call_statistics().calls[static_cast<unsigned>(call)];
 assert(r.attempts>r.completions&&r.completions==r.successes+r.failures);
}
extern "C" [[noreturn]] void ps5hip_llm_fail(const char*){throw Failure{};}
extern "C" hipError_t hipInit(unsigned int flags){entered(m0::HipCall::init);assert(flags==0);return init_error;}
extern "C" hipError_t hipGetDeviceCount(int* count){entered(m0::HipCall::device_count);*count=1;return count_error;}
extern "C" hipError_t hipMalloc(void** p,size_t bytes){
 entered(m0::HipCall::allocate);
 if(alloc_error!=hipSuccess)return alloc_error;
 assert(posix_memalign(p,256,bytes)==0);assert(*p);owners.emplace(*p,bytes);return hipSuccess;
}
extern "C" hipError_t hipFree(void* p){
 entered(fail_buffer_init?m0::HipCall::cleanup_free:m0::HipCall::free);
 assert(owners.count(p));if(free_error!=hipSuccess)return free_error;
 owners.erase(p);std::free(p);return hipSuccess;
}
extern "C" hipError_t hipMemcpy(void* dst,const void* src,size_t n,hipMemcpyKind kind){
 assert(kind==hipMemcpyHostToDevice||kind==hipMemcpyDeviceToHost||kind==hipMemcpyDeviceToDevice);
 entered(kind==hipMemcpyHostToDevice?m0::HipCall::h2d:kind==hipMemcpyDeviceToHost?m0::HipCall::d2h:m0::HipCall::d2d);
 if(throw_copy)throw Failure{};
 if(block_copy){std::unique_lock<std::mutex> guard(copy_mutex);++entered_copies;copy_changed.notify_all();copy_changed.wait(guard,[]{return release_copy;});}
 if(copy_error!=hipSuccess)return copy_error;std::memcpy(dst,src,n);return hipSuccess;
}
extern "C" hipError_t hipMemset(void* dst,int value,size_t n){
 const auto stats=m0::hip_call_statistics();const auto fill=stats.calls[static_cast<unsigned>(m0::HipCall::fill)];
 entered(fill.attempts>fill.completions?m0::HipCall::fill:m0::HipCall::clear);
 if(fill_error!=hipSuccess)return fill_error;std::memset(dst,value,n);return hipSuccess;
}
extern "C" hipError_t hipDeviceSynchronize(){entered(m0::HipCall::synchronize);return sync_error;}
extern "C" ggml_backend_buffer_t __real_ggml_backend_buffer_init(ggml_backend_buffer_type_t,ggml_backend_buffer_i,void*,size_t);
extern "C" ggml_backend_buffer_t __wrap_ggml_backend_buffer_init(ggml_backend_buffer_type_t type,ggml_backend_buffer_i iface,void* p,size_t size){
 if(fail_buffer_init)return nullptr;return __real_ggml_backend_buffer_init(type,iface,p,size);
}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){return ggml_backend_sched_graph_compute_async(scheduler,graph);}
using m0::HipCall;
static auto row(HipCall call){return m0::hip_call_statistics().calls[static_cast<unsigned>(call)];}
static void check_row(HipCall call,uint64_t attempts,uint64_t successes,uint64_t bytes,int last=0,int first=0){
 auto r=row(call);assert(r.attempts==attempts&&r.completions==attempts&&r.successes==successes&&r.failures==attempts-successes);
 assert(r.requested_bytes==bytes&&r.last_result==last&&r.first_error==first);
}
template<class F> static void failure(F fn){bool caught=false;try{fn();}catch(const Failure&){caught=true;}assert(caught);}
static void same_calls(const m0::HipCallStatistics& before){
 auto after=m0::hip_call_statistics();for(unsigned i=0;i<static_cast<unsigned>(HipCall::count);++i){
  const auto& a=after.calls[i];const auto& b=before.calls[i];
  assert(a.attempts==b.attempts&&a.completions==b.completions&&a.successes==b.successes&&a.failures==b.failures&&a.requested_bytes==b.requested_bytes&&a.last_result==b.last_result&&a.first_error==b.first_error);
 }
}
int main(int argc,char** argv){
 assert(argc==2);std::string mode=argv[1];auto reg=m0::registration();auto device=ggml_backend_reg_dev_get(reg,0);
 if(mode=="init-failure")init_error=hipErrorNotInitialized;
 if(mode=="count-failure")count_error=hipErrorNoDevice;
 auto backend=ggml_backend_dev_init(device,nullptr);
 if(mode=="init-failure"){
  assert(!backend);check_row(HipCall::init,1,0,0,hipErrorNotInitialized,hipErrorNotInitialized);check_row(HipCall::device_count,0,0,0);return 0;
 }
 if(mode=="count-failure"){
  assert(!backend);check_row(HipCall::init,1,1,0);check_row(HipCall::device_count,1,0,0,hipErrorNoDevice,hipErrorNoDevice);return 0;
 }
 assert(backend);check_row(HipCall::init,1,1,0);check_row(HipCall::device_count,1,1,0);
 auto type=ggml_backend_get_default_buffer_type(backend);
 if(mode=="allocation-failure"){
  alloc_error=hipErrorOutOfMemory;assert(!type->iface.alloc_buffer(type,1024));
  check_row(HipCall::allocate,1,0,1024,hipErrorOutOfMemory,hipErrorOutOfMemory);assert(owners.empty());
 }else if(mode=="buffer-init-failure"||mode=="cleanup-failure"){
  fail_buffer_init=true;if(mode=="cleanup-failure")free_error=hipErrorUnknown;
  if(mode=="cleanup-failure")failure([&]{type->iface.alloc_buffer(type,1024);});
  else assert(!type->iface.alloc_buffer(type,1024));
  check_row(HipCall::allocate,1,1,1024);
  check_row(HipCall::cleanup_free,1,mode=="cleanup-failure"?0:1,1024,free_error,free_error);
  assert(owners.size()==(mode=="cleanup-failure"?1u:0u));
  if(!owners.empty()){std::free(owners.begin()->first);owners.clear();}
 }else{
  const auto initial=m0::hip_call_statistics();assert(!type->iface.alloc_buffer(type,0));assert(!type->iface.alloc_buffer(type,m0::allocation_limit+1));same_calls(initial);
  auto ctx=ggml_init({1<<20,nullptr,true});assert(ctx);
  auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);
  auto storage=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(storage);const auto storage_size=ggml_backend_buffer_get_size(storage);
  check_row(HipCall::allocate,1,1,storage_size);unsigned char input[64],output[64];std::memset(input,0x11,64);
  ggml_backend_tensor_set(a,input,0,64);check_row(HipCall::h2d,1,1,64);
  if(mode=="normal"){
   ggml_backend_tensor_copy(a,b);check_row(HipCall::d2d,1,1,64);
   ggml_backend_tensor_get(b,output,0,64);assert(!std::memcmp(input,output,64));check_row(HipCall::d2h,1,1,64);
   storage->iface.memset_tensor(storage,b,0x7a,12,20);check_row(HipCall::fill,1,1,20);
   ggml_backend_tensor_get(b,output,0,64);for(unsigned i=0;i<64;++i)assert(output[i]==(i>=12&&i<32?0x7a:0x11));
   ggml_backend_buffer_clear(storage,0);check_row(HipCall::clear,1,1,storage_size);
   ggml_backend_tensor_get(a,output,0,64);for(auto byte:output)assert(byte==0);
   ggml_backend_synchronize(backend);check_row(HipCall::synchronize,1,1,0);
   const auto before=m0::hip_call_statistics();m0::reset_statistics();same_calls(before);
  }else if(mode=="in-flight"||mode=="concurrent"){
   const unsigned threads=mode=="in-flight"?1:8,iterations=mode=="in-flight"?1:128;
   block_copy=true;std::vector<std::thread> workers;
   for(unsigned i=0;i<threads;++i)workers.emplace_back([&]{unsigned char out[64];for(unsigned j=0;j<iterations;++j){ggml_backend_tensor_get(a,out,0,64);assert(!std::memcmp(input,out,64));}});
   {std::unique_lock<std::mutex> guard(copy_mutex);copy_changed.wait(guard,[&]{return entered_copies==threads;});
    const auto in_flight=row(HipCall::d2h);assert(in_flight.attempts==threads&&in_flight.completions==0&&in_flight.successes==0&&in_flight.failures==0&&in_flight.requested_bytes==64*threads);
    release_copy=true;copy_changed.notify_all();}
   for(auto& worker:workers)worker.join();block_copy=false;
   check_row(HipCall::d2h,threads*iterations,threads*iterations,64*threads*iterations);
  }else if(mode=="no-return"){
   throw_copy=true;failure([&]{ggml_backend_tensor_get(a,output,0,64);});throw_copy=false;
   const auto incomplete=row(HipCall::d2h);assert(incomplete.attempts==1&&incomplete.completions==0&&incomplete.successes==0&&incomplete.failures==0&&incomplete.requested_bytes==64&&incomplete.first_error==0);
  }else if(mode=="bounds"){
   const auto before=m0::hip_call_statistics();
   failure([&]{storage->iface.set_tensor(storage,a,input,64,1);});
   failure([&]{storage->iface.get_tensor(storage,a,output,SIZE_MAX,1);});
   failure([&]{storage->iface.memset_tensor(storage,a,1,63,2);});
   auto saved=a->data;a->data=static_cast<char*>(ggml_backend_buffer_get_base(storage))+storage_size;
   failure([&]{storage->iface.get_tensor(storage,a,output,0,1);});
   failure([&]{storage->iface.cpy_tensor(storage,a,b);});a->data=saved;
   b->ne[0]=8;assert(!storage->iface.cpy_tensor(storage,a,b));b->ne[0]=16;same_calls(before);
  }else if(mode=="h2d-failure"||mode=="d2h-failure"||mode=="d2d-failure"){
   copy_error=hipErrorInvalidValue;
   if(mode=="h2d-failure"){
    failure([&]{ggml_backend_tensor_set(a,input,0,64);});check_row(HipCall::h2d,2,1,128,copy_error,copy_error);
    copy_error=hipSuccess;ggml_backend_tensor_set(a,input,0,64);check_row(HipCall::h2d,3,2,192,0,hipErrorInvalidValue);
   }else if(mode=="d2h-failure"){
    failure([&]{ggml_backend_tensor_get(a,output,0,64);});check_row(HipCall::d2h,1,0,64,copy_error,copy_error);
   }else{failure([&]{ggml_backend_tensor_copy(a,b);});check_row(HipCall::d2d,1,0,64,copy_error,copy_error);}
   copy_error=hipSuccess;
  }else if(mode=="fill-failure"||mode=="clear-failure"){
   fill_error=hipErrorInvalidValue;
   if(mode=="fill-failure"){
    failure([&]{storage->iface.memset_tensor(storage,a,2,4,8);});check_row(HipCall::fill,1,0,8,fill_error,fill_error);
   }else{failure([&]{ggml_backend_buffer_clear(storage,0);});check_row(HipCall::clear,1,0,storage_size,fill_error,fill_error);}
   fill_error=hipSuccess;
  }else if(mode=="sync-failure"){
   sync_error=hipErrorUnknown;failure([&]{ggml_backend_synchronize(backend);});check_row(HipCall::synchronize,1,0,0,sync_error,sync_error);sync_error=hipSuccess;
  }else if(mode=="free-failure"){
   free_error=hipErrorUnknown;failure([&]{ggml_backend_buffer_free(storage);});
   check_row(HipCall::free,1,0,storage_size,free_error,free_error);assert(owners.size()==1);free_error=hipSuccess;
  }else assert(false);
  ggml_backend_buffer_free(storage);check_row(HipCall::free,mode=="free-failure"?2:1,1,storage_size*(mode=="free-failure"?2:1),0,mode=="free-failure"?hipErrorUnknown:0);
  ggml_free(ctx);
 }
 ggml_backend_free(backend);assert(owners.empty());puts("HOST_BACKEND_CALL_COVERAGE_PASS transport=mock native=false");
}
