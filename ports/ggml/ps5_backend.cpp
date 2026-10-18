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
#include "layer_qualification.h"
#include "dense_layer_qualification.h"
#include "model_layer.h"
#include "layer_dispatch.h"
#include "full_dispatch.h"
#include "full_qualification.h"
#include <map>
#include <string>
#include <limits>
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
// Layer mode has a distinct owner; the existing prefix context keeps its exact type.
static ActualLayerQualification* layer_scope=nullptr;
static bool layer_active=false;
static bool layer_owned(const ggml_tensor* t){
 if(!t||!t->data)return false;
 const ggml_tensor* owner=t;
 for(unsigned depth=0;owner->view_src;++depth){if(depth>=64)return false;owner=owner->view_src;}
 auto* buffer=owner->buffer;
 if(!buffer||buffer->buft!=&buffer_type||(t->buffer&&t->buffer!=buffer))return false;
 const auto begin=reinterpret_cast<uintptr_t>(buffer->context),p=reinterpret_cast<uintptr_t>(t->data);
 return p>=begin&&p-begin<=buffer->size&&ggml_nbytes(t)<=buffer->size-(p-begin);
}
[[noreturn]] static void layer_fail(const char* site,hipError_t result){
 std::fprintf(stderr,"M3 layer HIP failure site=%s result=%d\n",site,int(result));ps5hip_llm_fail("layer-hip");
}
struct LayerBackend {
 bool dense_fixture=false;
 ggml_backend_t owner=nullptr;
 ps5hip_layer::Dispatch dispatch{layer_owned,layer_fail};
 LayerStatistics statistics{};
 ggml_context* persistent=nullptr;
 std::array<ggml_backend_buffer_t,12> storage{};
 ps5hip_layer::Weights weights{};
 ps5hip_layer::Cache cache{};
};
static LayerBackend* layer_backend=nullptr;
static bool full_enabled=false,full_active=false;
static unsigned live_backends=0;
static std::array<std::string,6> full_modules;
static std::map<ggml_backend_t,ps5hip_full::Dispatch*> full_backends;
static FullStatistics full_stats;
void enable_full(const std::array<const char*,6>& modules){
 require(!full_enabled&&!live_backends&&!layer_backend&&!prefix_scope&&!layer_scope);
 for(size_t i=0;i<modules.size();++i){require(modules[i]&&modules[i][0]);full_modules[i]=modules[i];}
 full_stats={};full_enabled=true;
}
void disable_full(){require(full_enabled&&!live_backends&&full_backends.empty()&&!full_active);full_enabled=false;full_modules={};}
void reset_full(){require(full_enabled&&!full_active&&!full_backends.empty());for(auto& item:full_backends)item.second->reset();}
FullStatistics full_statistics(){
 auto stats=full_stats;for(const auto& item:full_backends)stats.launches+=item.second->layer.base.completed_launches;return stats;
}
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
static const char* name(ggml_backend_t){return full_enabled?"PS5HIP-FULL":"PS5HIP-M0-STORAGE-ONLY";}
static const char* buffer_name(ggml_backend_buffer_type_t){return "PS5HIP";}
static bool supports(ggml_backend_dev_t,const ggml_tensor* t){
 if(!full_enabled||!t)return false;
 // Placement only. The complete fixed model graph is admitted before execution.
 switch(t->op){
 case GGML_OP_NONE:case GGML_OP_GET_ROWS:case GGML_OP_RMS_NORM:case GGML_OP_MUL:case GGML_OP_MUL_MAT:
 case GGML_OP_RESHAPE:case GGML_OP_VIEW:case GGML_OP_PERMUTE:case GGML_OP_ROPE:
 case GGML_OP_SET_ROWS:case GGML_OP_SOFT_MAX:case GGML_OP_CONT:case GGML_OP_ADD:
 case GGML_OP_GLU:case GGML_OP_DUP:return true;
 default:return false;
 }
}
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
static const char* description(ggml_backend_dev_t){return full_enabled?"Fixed TinyLlama full-graph qualification":"M0 storage-only prototype; no tensor ops";}
static void memory(ggml_backend_dev_t,size_t* free,size_t* total){*free=*total=0;}
static enum ggml_backend_dev_type type(ggml_backend_dev_t){return GGML_BACKEND_DEVICE_TYPE_GPU;}
static void props(ggml_backend_dev_t d,ggml_backend_dev_props* p){*p={};p->name=dev_name(d);p->description=description(d);p->type=type(d);memory(d,&p->memory_free,&p->memory_total);}
static ggml_backend_buffer_type_t buft(ggml_backend_dev_t){return &buffer_type;}
static bool supports_buft(ggml_backend_dev_t,ggml_backend_buffer_type_t t){return t==&buffer_type;}
static void sync(ggml_backend_t){begin_call(HipCall::synchronize);hip_ok(finish_call(HipCall::synchronize,hipDeviceSynchronize()));}
static void release(ggml_backend_t b){
 auto full=full_backends.find(b);
 if(full!=full_backends.end()){
  require(!full_active);full_stats.launches+=full->second->layer.base.completed_launches;
  full->second->shutdown();delete full->second;full_backends.erase(full);++full_stats.instances_freed;
 }
 require(live_backends>0);--live_backends;
 require(!b->context&&(!prefix_scope||prefix_scope->backend!=b)&&(!layer_backend||layer_backend->owner!=b));delete b;
}
static ggml_status graph(ggml_backend_t b,ggml_cgraph* g){
 ++graph_calls;
 auto full=full_backends.find(b);
 if(full!=full_backends.end()){
  require(full_active);
  try{return full->second->execute_graph(g)?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;}
  catch(...){ps5hip_llm_fail("full-graph-invariant");}
 }
 if(layer_backend&&layer_backend->owner==b){
  require(layer_active);
  try{return layer_backend->dispatch.execute_graph(g)?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;}
  catch(...){ps5hip_llm_fail("layer-graph-invariant");}
 }
 if(!b->context)return ggml_graph_n_nodes(g)==0?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;
 auto& state=*static_cast<PrefixBackend*>(b->context);
 try{return state.dispatch.execute_graph(g)?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;}
 catch(...){ps5hip_llm_fail("prefix-graph-invariant");}
}
static ggml_backend_t init(ggml_backend_dev_t d,const char*){
 begin_call(HipCall::init);if(finish_call(HipCall::init,hipInit(0))!=hipSuccess)return nullptr;
 int count=0;begin_call(HipCall::device_count);if(finish_call(HipCall::device_count,hipGetDeviceCount(&count))!=hipSuccess||count!=1)return nullptr;
 auto* b=new(std::nothrow)ggml_backend{};if(!b)return nullptr;
 static uint8_t guid[16]={0x50,0x53,0x35,0x48,0x49,0x50,0x4d,0x30};b->guid=&guid;b->device=d;b->iface.get_name=name;b->iface.free=release;b->iface.synchronize=sync;b->iface.graph_compute=graph;
 if(full_enabled){
  auto* state=new(std::nothrow)ps5hip_full::Dispatch(layer_owned,layer_fail);require(state);
  std::array<const char*,6> paths{};for(size_t i=0;i<paths.size();++i)paths[i]=full_modules[i].c_str();
  try{state->initialize(paths);}catch(...){ps5hip_llm_fail("full-initialize");}
  require(full_backends.emplace(b,state).second);++full_stats.instances_created;
 }
 ++live_backends;return b;
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
 require(!full_enabled&&b&&b->device==&device&&!b->context&&!prefix_active&&(!layer_backend||layer_backend->owner!=b));
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
 require(b&&b->device==&device&&b->context&&sink&&!prefix_scope&&!prefix_active&&!layer_scope&&!layer_active);prefix_scope=this;
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
void configure_layer(ggml_backend_t b,const std::array<const char*,6>& modules){
 require(!full_enabled&&b&&b->device==&device&&!b->context&&!layer_backend&&!layer_scope&&!prefix_scope&&!prefix_active);
 layer_backend=new(std::nothrow)LayerBackend;require(layer_backend);layer_backend->owner=b;
 try{layer_backend->dispatch.initialize(modules);}catch(...){ps5hip_llm_fail("layer-module-invariant");}
}
void configure_dense_layer(ggml_backend_t b,const std::array<const char*,6>& modules){
 require(!full_enabled&&b&&b->device==&device&&!b->context&&!layer_backend&&!layer_scope&&!prefix_scope&&!prefix_active);
 layer_backend=new(std::nothrow)LayerBackend;require(layer_backend);layer_backend->owner=b;layer_backend->dense_fixture=true;
 try{layer_backend->dispatch.initialize_dense_fixture(modules);}catch(...){ps5hip_llm_fail("dense-layer-module-invariant");}
}
static LayerBackend& layer_state(ggml_backend_t b){require(layer_backend&&layer_backend->owner==b);return *layer_backend;}
void execute_dense_layer(ggml_backend_t b,ggml_cgraph* graph){
 auto& state=layer_state(b);require(state.dense_fixture&&!layer_active&&!layer_scope&&!prefix_scope&&!prefix_active&&graph);
 layer_active=true;
 // Fatal errors retain active state and resources; uncertain GPU work is quarantined.
 try{require(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);}catch(...){ps5hip_llm_fail("dense-layer-graph-invariant");}
 layer_active=false;++state.statistics.invocations;++state.statistics.request_cursor;
 state.statistics.last_token=0;state.statistics.last_position=int32_t(state.statistics.request_cursor-1);
}
void reset_layer_cache(ggml_backend_t b){
 require(!layer_active&&!layer_scope);auto& state=layer_state(b);
 sync(b);
 if(state.persistent){for(auto* t:{state.cache.k,state.cache.v})ggml_backend_tensor_memset(t,0,0,ggml_nbytes(t));sync(b);}
 state.dispatch.reset();state.statistics.request_cursor=0;state.statistics.last_token=state.statistics.last_position=-1;
 require(state.statistics.cache_resets<UINT64_MAX);++state.statistics.cache_resets;
}
ps5hip_prefix::CallStatistics disable_layer(ggml_backend_t b){
 require(!layer_active&&!layer_scope);auto& state=layer_state(b);
 try{state.dispatch.shutdown();}catch(...){ps5hip_llm_fail("layer-release-invariant");}
 for(auto buffer:state.storage)if(buffer)ggml_backend_buffer_free(buffer);
 if(state.persistent)ggml_free(state.persistent);
 auto calls=state.dispatch.base.calls;delete layer_backend;layer_backend=nullptr;return calls;
}
LayerStatistics layer_statistics(ggml_backend_t b){auto& state=layer_state(b);auto result=state.statistics;result.completed_launches=state.dispatch.base.completed_launches;return result;}
ps5hip_prefix::CallStatistics layer_call_statistics(ggml_backend_t b){return layer_state(b).dispatch.base.calls;}
ActualLayerQualification::ActualLayerQualification(ggml_backend_t b,const LayerRequest* sequence,size_t size,Readback sink,void* pointer):backend(b),requests(sequence),count(size),readback(sink),user(pointer){
 require(sequence&&size&&size<=512&&sink&&!layer_scope&&!layer_active&&!prefix_scope&&!prefix_active);
 auto& state=layer_state(b);require(!state.dense_fixture&&state.statistics.request_cursor==0);
 for(size_t i=0;i<size;++i)require(sequence[i].token>=0&&sequence[i].token<32000&&sequence[i].position==int32_t(i));
 layer_scope=this;
}
ActualLayerQualification::~ActualLayerQualification(){require(layer_scope==this&&!layer_active);layer_scope=nullptr;}
static void qualify_actual_layer(ggml_cgraph* original){
 require(layer_scope&&!layer_active&&!prefix_scope&&!prefix_active);layer_active=true;
 auto& scope=*layer_scope;auto& state=layer_state(scope.backend);
 try{
  const auto selection=ps5hip_layer::select(original);
  if(!state.persistent){
   state.persistent=ggml_init({1024*1024,nullptr,true});require(state.persistent);
   state.weights=ps5hip_layer::make_weights(state.persistent,selection);state.cache=ps5hip_layer::make_cache(state.persistent);
   // Submission coherency flushes whole owners. Keep immutable weights and each
   // KV cache in separate allocations so a cache write never flushes all weights.
   for(size_t i=0;i<state.storage.size();++i){
    auto* tensor=i<10?state.weights.uploads[i].clone:i==10?state.cache.k:state.cache.v;
    const size_t size=ggml_backend_buft_get_alloc_size(&buffer_type,tensor);
    state.storage[i]=ggml_backend_alloc_buffer(scope.backend,size);require(state.storage[i]);
    require(ggml_backend_tensor_alloc(state.storage[i],tensor,ggml_backend_buffer_get_base(state.storage[i]))==GGML_STATUS_SUCCESS);
   }
   std::array<unsigned char,64*1024> chunk{};
   for(const auto& pair:state.weights.uploads){
    require(pair.original&&pair.original->buffer&&pair.original->data&&ggml_nbytes(pair.original)==ggml_nbytes(pair.clone));
    const auto bytes=ggml_nbytes(pair.original);require(bytes<=allocation_limit&&bytes<=UINT64_MAX-state.statistics.uploaded_weight_bytes);
    for(size_t off=0;off<bytes;){const auto n=std::min(chunk.size(),bytes-off);ggml_backend_tensor_get(pair.original,chunk.data(),off,n);ggml_backend_tensor_set(pair.clone,chunk.data(),off,n);off+=n;}
    state.statistics.uploaded_weight_bytes+=bytes;
   }
   for(auto* t:{state.cache.k,state.cache.v})ggml_backend_tensor_memset(t,0,0,ggml_nbytes(t));
  }
  for(size_t i=0;i<scope.count;++i){
   const auto request=scope.requests[i];require(request.position==int32_t(state.statistics.request_cursor));
   auto* ctx=ggml_init({1024*1024,nullptr,true});require(ctx);
   const int64_t extent=request.position<256?256:512;
   const auto selected=ps5hip_layer::materialize(ctx,selection,state.weights,state.cache,extent);
   auto storage=ggml_backend_alloc_ctx_tensors(ctx,scope.backend);require(storage);
   const int64_t slot=request.position;
   std::array<int64_t,256> vslots{};for(size_t j=0;j<vslots.size();++j)vslots[j]=int64_t(j*512)+slot;
   const void* controls[]={&request.token,&request.position,&slot,vslots.data()};
   const size_t sizes[]={4,4,8,sizeof(vslots)};size_t bytes=0;
   for(size_t j=0;j<4;++j){auto* t=selected.uploads[j].clone;require(ggml_nbytes(t)==sizes[j]);ggml_backend_tensor_set(t,controls[j],0,sizes[j]);bytes+=sizes[j];}
   std::array<float,512> mask{};for(int64_t j=0;j<extent;++j)mask[j]=j<=request.position?0.f:-std::numeric_limits<float>::infinity();
   auto* mask_tensor=selected.uploads[4].clone;require(mask_tensor->type==GGML_TYPE_F32&&ggml_nbytes(mask_tensor)==size_t(extent)*sizeof(float));
   ggml_backend_tensor_set(mask_tensor,mask.data(),0,ggml_nbytes(mask_tensor));bytes+=ggml_nbytes(mask_tensor);
   require(ggml_backend_graph_compute(scope.backend,selected.graph)==GGML_STATUS_SUCCESS);
   scope.readback(selected,request,scope.user);
   ggml_backend_buffer_free(storage);ggml_free(ctx);
   require(state.statistics.invocations<UINT64_MAX&&bytes<=UINT64_MAX-state.statistics.uploaded_request_bytes);
   ++state.statistics.invocations;state.statistics.uploaded_request_bytes+=bytes;++state.statistics.request_cursor;
   state.statistics.last_token=request.token;state.statistics.last_position=request.position;
  }
 }catch(...){ps5hip_llm_fail("layer-qualification-invariant");}
 layer_active=false;
}
ggml_backend_dev_t device_handle(){return &device;}
GateStatistics statistics(){return {gate_calls,scheduled_compute_calls,rejected,graph_calls,last_graph};}
void reset_statistics(){gate_calls=scheduled_compute_calls=rejected=graph_calls=0;last_graph=nullptr;}
}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 m0::last_graph=graph;
 ++m0::gate_calls;
 if(m0::full_enabled){
  m0::require(!m0::full_active);
  ggml_backend_t owner=nullptr;
  for(int i=0;i<ggml_graph_n_nodes(graph);++i){
   auto* t=ggml_graph_node(graph,i);if(!(t->flags&GGML_TENSOR_FLAG_COMPUTE))continue;
   auto* selected=ggml_backend_sched_get_tensor_backend(scheduler,t);
   if(!selected||!m0::full_backends.count(selected)||(owner&&owner!=selected)){
    ++m0::rejected;std::fprintf(stderr,"FULL_GATE REJECT node=%d name=%s selected=%s\n",i,t->name,selected?ggml_backend_name(selected):"unassigned");return GGML_STATUS_FAILED;
   }
   owner=selected;
  }
  if(!owner){++m0::rejected;return GGML_STATUS_FAILED;}
  auto& dispatch=*m0::full_backends.at(owner);
  try{
   if(!dispatch.begin(graph)){++m0::rejected;return GGML_STATUS_FAILED;}
   m0::full_active=true;++m0::scheduled_compute_calls;
   const auto result=__real_ggml_backend_sched_graph_compute_async(scheduler,graph);
   const bool complete=dispatch.finish(result==GGML_STATUS_SUCCESS);m0::full_active=false;
   if(!complete){++m0::rejected;return GGML_STATUS_FAILED;}
   ++m0::full_stats.requests;m0::full_stats.completed_nodes+=798;return GGML_STATUS_SUCCESS;
  }catch(...){ps5hip_llm_fail("full-scheduler-invariant");}
 }
 if(m0::prefix_scope)m0::qualify_actual_prefix(graph);
 if(m0::layer_scope)m0::qualify_actual_layer(graph);
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
