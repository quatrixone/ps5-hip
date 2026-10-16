#include "ps5_backend.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <atomic>
#include "prefix_qualification.h"
#include "prefix_dispatch.h"
#include <algorithm>
#include <array>
namespace m0 {
static ggml_backend_device device{};
static ggml_backend_reg registry{};
static ggml_backend_buffer_type buffer_type{};
static unsigned graph_calls=0;
static const ggml_cgraph* last_graph=nullptr;
static unsigned scheduled_compute_calls=0;
static unsigned gate_calls=0, rejected=0;
static ActualPrefixQualification* prefix_scope=nullptr;
static bool prefix_active=false;
static bool prefix_owned(const ggml_tensor* t){
 if(!t||!t->data||!t->buffer||t->buffer->buft!=&buffer_type||t->view_src)return false;
 const auto begin=reinterpret_cast<uintptr_t>(t->buffer->context),p=reinterpret_cast<uintptr_t>(t->data);
 return p>=begin&&p-begin<=t->buffer->size&&ggml_nbytes(t)<=t->buffer->size-(p-begin);
}
[[noreturn]] static void prefix_fail(const char* site,hipError_t result){
 std::fprintf(stderr,"M2 prefix HIP failure site=%s result=%d\n",site,int(result));ps5hip_llm_fail("prefix-hip");
}
struct PrefixBackend {
 ps5hip_prefix::Dispatch dispatch{prefix_owned,prefix_fail};
 PrefixStatistics statistics{};
};
static HipCallStatistics hip_calls{};
static std::atomic_flag hip_calls_lock=ATOMIC_FLAG_INIT;
struct HipCallsGuard {
 HipCallsGuard(){while(hip_calls_lock.test_and_set(std::memory_order_acquire)){}}
 ~HipCallsGuard(){hip_calls_lock.clear(std::memory_order_release);}
};
void require(bool yes){if(!yes){std::fputs("M0 backend invariant/error\n",stderr);ps5hip_llm_fail("backend-invariant");}}
HipCallStatistics hip_call_statistics(){HipCallsGuard guard;return hip_calls;}
static void begin_call(HipCall call,size_t bytes=0){
 bool valid;
 {HipCallsGuard guard;auto& row=hip_calls.calls[static_cast<unsigned>(call)];
  valid=row.attempts<UINT64_MAX&&bytes<=UINT64_MAX-row.requested_bytes;
  if(valid){++row.attempts;row.requested_bytes+=bytes;}
 }
 require(valid);
}
static hipError_t finish_call(HipCall call,hipError_t result){
 bool valid;
 {HipCallsGuard guard;auto& row=hip_calls.calls[static_cast<unsigned>(call)];
  valid=row.completions<row.attempts;
  if(valid){++row.completions;row.last_result=int(result);
   if(result==hipSuccess)++row.successes;else{++row.failures;if(!row.first_error)row.first_error=int(result);}
  }
 }
 require(valid);
 return result;
}
static void hip_ok(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"M0 HIP error=%d\n",int(e));ps5hip_llm_fail("backend-invariant");}}
static const char* name(ggml_backend_t){return "PS5HIP-M0-STORAGE-ONLY";}
static const char* buffer_name(ggml_backend_buffer_type_t){return "PS5HIP";}
static bool supports(ggml_backend_dev_t,const ggml_tensor*){return false;}
static void* base(ggml_backend_buffer_t b){return b->context;}
static void free_buffer(ggml_backend_buffer_t b){begin_call(HipCall::free,b->size);hip_ok(finish_call(HipCall::free,hipFree(b->context)));}
static uint8_t* span(ggml_backend_buffer_t b,const ggml_tensor* t,size_t offset,size_t size){
 const auto begin=reinterpret_cast<uintptr_t>(b->context),p=reinterpret_cast<uintptr_t>(t->data);
 require(p>=begin&&p-begin<=b->size&&offset<=ggml_nbytes(t)&&size<=ggml_nbytes(t)-offset&&offset<=b->size-(p-begin)&&size<=b->size-(p-begin)-offset);
 return reinterpret_cast<uint8_t*>(p)+offset;
}
static void set(ggml_backend_buffer_t b,ggml_tensor* t,const void* data,size_t off,size_t size){
 auto* pointer=span(b,t,off,size);begin_call(HipCall::h2d,size);hip_ok(finish_call(HipCall::h2d,hipMemcpy(pointer,data,size,hipMemcpyHostToDevice)));
}
static void get(ggml_backend_buffer_t b,const ggml_tensor* t,void* data,size_t off,size_t size){
 auto* pointer=span(b,t,off,size);begin_call(HipCall::d2h,size);hip_ok(finish_call(HipCall::d2h,hipMemcpy(data,pointer,size,hipMemcpyDeviceToHost)));
}
static void fill(ggml_backend_buffer_t b,ggml_tensor* t,uint8_t value,size_t off,size_t size){
 auto* pointer=span(b,t,off,size);begin_call(HipCall::fill,size);hip_ok(finish_call(HipCall::fill,hipMemset(pointer,value,size)));
}
static void clear(ggml_backend_buffer_t b,uint8_t value){begin_call(HipCall::clear,b->size);hip_ok(finish_call(HipCall::clear,hipMemset(b->context,value,b->size)));}
static bool copy(ggml_backend_buffer_t b,const ggml_tensor* src,ggml_tensor* dst){
 if(!src->buffer||src->buffer->buft!=&buffer_type||ggml_nbytes(src)!=ggml_nbytes(dst))return false;
 const auto size=ggml_nbytes(dst);auto* to=span(b,dst,0,size);auto* from=span(src->buffer,src,0,size);
 begin_call(HipCall::d2d,size);hip_ok(finish_call(HipCall::d2d,hipMemcpy(to,from,size,hipMemcpyDeviceToDevice)));return true;
}
static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type,size_t size){
 if(!size || size>allocation_limit)return nullptr;void* pointer=nullptr;begin_call(HipCall::allocate,size);if(finish_call(HipCall::allocate,hipMalloc(&pointer,size))!=hipSuccess)return nullptr;
 ggml_backend_buffer_i iface{};iface.free_buffer=free_buffer;iface.get_base=base;iface.set_tensor=set;iface.get_tensor=get;iface.memset_tensor=fill;iface.clear=clear;iface.cpy_tensor=copy;
 auto b=ggml_backend_buffer_init(type,iface,pointer,size);if(!b){begin_call(HipCall::cleanup_free,size);hip_ok(finish_call(HipCall::cleanup_free,hipFree(pointer)));}return b;
}
static size_t maximum_size(ggml_backend_buffer_type_t){return allocation_limit;}
static size_t alignment(ggml_backend_buffer_type_t){return 256;}
static const char* dev_name(ggml_backend_dev_t){return "PS5HIP";}
static const char* description(ggml_backend_dev_t){return "M0 storage-only prototype; no tensor ops";}
static void memory(ggml_backend_dev_t,size_t* free,size_t* total){*free=*total=0;}
static enum ggml_backend_dev_type type(ggml_backend_dev_t){return GGML_BACKEND_DEVICE_TYPE_GPU;}
static void props(ggml_backend_dev_t d,ggml_backend_dev_props* p){*p={};p->name=dev_name(d);p->description=description(d);p->type=type(d);memory(d,&p->memory_free,&p->memory_total);}
static ggml_backend_buffer_type_t buft(ggml_backend_dev_t){return &buffer_type;}
static bool supports_buft(ggml_backend_dev_t,ggml_backend_buffer_type_t t){return t==&buffer_type;}
static void sync(ggml_backend_t){begin_call(HipCall::synchronize);hip_ok(finish_call(HipCall::synchronize,hipDeviceSynchronize()));}
static void release(ggml_backend_t b){
 require(!b->context&&(!prefix_scope||prefix_scope->backend!=b));delete b;
}
static ggml_status graph(ggml_backend_t b,ggml_cgraph* g){
 ++graph_calls;
 if(!b->context)return ggml_graph_n_nodes(g)==0?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;
 auto& state=*static_cast<PrefixBackend*>(b->context);
 try{return state.dispatch.execute_graph(g)?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;}
 catch(...){ps5hip_llm_fail("prefix-graph-invariant");}
}
static ggml_backend_t init(ggml_backend_dev_t d,const char*){
 begin_call(HipCall::init);if(finish_call(HipCall::init,hipInit(0))!=hipSuccess)return nullptr;
 int count=0;begin_call(HipCall::device_count);if(finish_call(HipCall::device_count,hipGetDeviceCount(&count))!=hipSuccess||count!=1)return nullptr;
 auto* b=new(std::nothrow)ggml_backend{};if(!b)return nullptr;
 static uint8_t guid[16]={0x50,0x53,0x35,0x48,0x49,0x50,0x4d,0x30};b->guid=&guid;b->device=d;b->iface.get_name=name;b->iface.free=release;b->iface.synchronize=sync;b->iface.graph_compute=graph;return b;
}
static const char* reg_name(ggml_backend_reg_t){return "PS5HIP-M0";}
static size_t count(ggml_backend_reg_t){return 1;}
static ggml_backend_dev_t get_device(ggml_backend_reg_t,size_t index){require(index==0);return &device;}
ggml_backend_reg_t registration(){
 static bool ready=false;if(ready)return &registry;ready=true;
 registry.api_version=GGML_BACKEND_API_VERSION;registry.iface.get_name=reg_name;registry.iface.get_device_count=count;registry.iface.get_device=get_device;
 device.reg=&registry;device.iface.get_name=dev_name;device.iface.get_description=description;device.iface.get_memory=memory;device.iface.get_type=type;device.iface.get_props=props;device.iface.init_backend=init;device.iface.get_buffer_type=buft;device.iface.supports_op=supports;device.iface.supports_buft=supports_buft;
 buffer_type.device=&device;buffer_type.iface.get_name=buffer_name;buffer_type.iface.alloc_buffer=allocate;buffer_type.iface.get_alignment=alignment;buffer_type.iface.get_max_size=maximum_size;return &registry;
}
// Caller must use this release boundary rather than invoking scheduler compute directly.
ggml_status release_compute(ggml_backend_sched_t scheduler,ggml_backend_t ps5,ggml_cgraph* graph){
 ggml_backend_sched_split_graph(scheduler,graph);
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* node=ggml_graph_node(graph,i);
  if(ggml_backend_sched_get_tensor_backend(scheduler,node)!=ps5||!supports(ps5->device,node))return GGML_STATUS_FAILED;
 }
 return ggml_backend_sched_graph_compute(scheduler,graph);
}
}

namespace m0 {
void configure_prefix(ggml_backend_t b,const std::array<const char*,5>& modules){
 require(b&&b->device==&device&&!b->context&&!prefix_active);
 auto* state=new(std::nothrow)PrefixBackend;require(state);b->context=state;
 try{state->dispatch.initialize(modules);}catch(...){ps5hip_llm_fail("prefix-module-invariant");}
}
PrefixCallStatistics disable_prefix(ggml_backend_t b){
 require(b&&b->device==&device&&b->context&&!prefix_active&&(!prefix_scope||prefix_scope->backend!=b));
 auto* state=static_cast<PrefixBackend*>(b->context);
 try{state->dispatch.shutdown();}catch(...){ps5hip_llm_fail("prefix-release-invariant");}
 auto calls=state->dispatch.calls;delete state;b->context=nullptr;return calls;
}
PrefixCallStatistics prefix_call_statistics(ggml_backend_t b){
 require(b&&b->device==&device&&b->context);return static_cast<PrefixBackend*>(b->context)->dispatch.calls;
}
PrefixStatistics prefix_statistics(ggml_backend_t b){
 require(b&&b->device==&device&&b->context);
 auto& state=*static_cast<PrefixBackend*>(b->context);auto result=state.statistics;
 result.completed_launches=state.dispatch.completed_launches;return result;
}
ActualPrefixQualification::ActualPrefixQualification(ggml_backend_t b,Readback sink,void* user):backend(b),readback(sink),user(user){
 require(b&&b->device==&device&&b->context&&sink&&!prefix_scope&&!prefix_active);prefix_scope=this;
}
ActualPrefixQualification::~ActualPrefixQualification(){require(prefix_scope==this&&!prefix_active);prefix_scope=nullptr;}
static void qualify_actual_prefix(ggml_cgraph* original){
 require(prefix_scope&&!prefix_active);prefix_active=true;
 auto& scope=*prefix_scope;auto& state=*static_cast<PrefixBackend*>(scope.backend->context);
 try{
  const auto selected=ps5hip_prefix::select(original);
  auto* ctx=ggml_init({1024*1024,nullptr,true});require(ctx);
  const auto prefix=ps5hip_prefix::materialize(ctx,selected);
  auto storage=ggml_backend_alloc_ctx_tensors(ctx,scope.backend);require(storage);
  std::array<unsigned char,64*1024> chunk{};size_t total=0;int32_t token=-1;
  for(size_t i=0;i<selected.leaves.size();++i){
   auto* source=selected.leaves[i];require(source->buffer&&source->data);
   require(ggml_nbytes(source)==ggml_nbytes(prefix.leaves[i]));
   const size_t bytes=ggml_nbytes(source);require(bytes<=allocation_limit&&total<=SIZE_MAX-bytes);total+=bytes;
   for(size_t offset=0;offset<bytes;){
    const size_t count=std::min(chunk.size(),bytes-offset);
    ggml_backend_tensor_get(source,chunk.data(),offset,count);
    if(i==1){require(offset==0&&count==4);std::memcpy(&token,chunk.data(),4);require(token>=0&&token<32000);}
    ggml_backend_tensor_set(prefix.leaves[i],chunk.data(),offset,count);offset+=count;
   }
  }
  require(total==39821316);
  const unsigned before=state.dispatch.completed_launches;
  require(ggml_backend_graph_compute(scope.backend,prefix.graph)==GGML_STATUS_SUCCESS);
  require(state.dispatch.completed_launches-before==9);
  scope.readback(prefix.outputs.data(),prefix.outputs.size(),token,scope.user);
  ggml_backend_buffer_free(storage);ggml_free(ctx);
  require(state.statistics.invocations<UINT64_MAX&&total<=UINT64_MAX-state.statistics.uploaded_leaf_bytes);
  ++state.statistics.invocations;state.statistics.uploaded_leaf_bytes+=total;state.statistics.last_token=token;
 }catch(...){ps5hip_llm_fail("prefix-qualification-invariant");}
 prefix_active=false;
}
ggml_backend_dev_t device_handle(){return &device;}
GateStatistics statistics(){return {gate_calls,scheduled_compute_calls,rejected,graph_calls,last_graph};}
void reset_statistics(){gate_calls=scheduled_compute_calls=rejected=graph_calls=0;last_graph=nullptr;}
}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 m0::last_graph=graph;
 ++m0::gate_calls;
 if(m0::prefix_scope)m0::qualify_actual_prefix(graph);
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){
  auto* tensor=ggml_graph_node(graph,i);
  auto selected=ggml_backend_sched_get_tensor_backend(scheduler,tensor);
  if(!selected || ggml_backend_get_device(selected)!=m0::device_handle() || !m0::supports(m0::device_handle(),tensor)){
   ++m0::rejected;
   std::printf("LLM_GATE REJECT node=%d op=%s selected=%s\n",i,ggml_op_desc(tensor),selected?ggml_backend_name(selected):"unassigned");
   return GGML_STATUS_FAILED;
  }
 }
 ++m0::scheduled_compute_calls;
 return __real_ggml_backend_sched_graph_compute_async(scheduler,graph);
}
