#include "ps5_backend.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <new>
namespace m0 {
static ggml_backend_device device{};
static ggml_backend_reg registry{};
static ggml_backend_buffer_type buffer_type{};
static unsigned graph_calls=0;
static const ggml_cgraph* last_graph=nullptr;
static unsigned scheduled_compute_calls=0;
static unsigned gate_calls=0, rejected=0;
void require(bool yes){if(!yes){std::fputs("M0 backend invariant/error\n",stderr);ps5hip_llm_fail("backend-invariant");}}
static void hip_ok(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"M0 HIP error=%d\n",int(e));ps5hip_llm_fail("backend-invariant");}}
static const char* name(ggml_backend_t){return "PS5HIP-M0-STORAGE-ONLY";}
static const char* buffer_name(ggml_backend_buffer_type_t){return "PS5HIP";}
static bool supports(ggml_backend_dev_t,const ggml_tensor*){return false;}
static void* base(ggml_backend_buffer_t b){return b->context;}
static void free_buffer(ggml_backend_buffer_t b){hip_ok(hipFree(b->context));}
static uint8_t* span(ggml_backend_buffer_t b,const ggml_tensor* t,size_t offset,size_t size){
 const auto begin=reinterpret_cast<uintptr_t>(b->context),p=reinterpret_cast<uintptr_t>(t->data);
 require(p>=begin&&p-begin<=b->size&&offset<=ggml_nbytes(t)&&size<=ggml_nbytes(t)-offset&&offset<=b->size-(p-begin)&&size<=b->size-(p-begin)-offset);
 return reinterpret_cast<uint8_t*>(p)+offset;
}
static void set(ggml_backend_buffer_t b,ggml_tensor* t,const void* data,size_t off,size_t size){hip_ok(hipMemcpy(span(b,t,off,size),data,size,hipMemcpyHostToDevice));}
static void get(ggml_backend_buffer_t b,const ggml_tensor* t,void* data,size_t off,size_t size){hip_ok(hipMemcpy(data,span(b,t,off,size),size,hipMemcpyDeviceToHost));}
static void fill(ggml_backend_buffer_t b,ggml_tensor* t,uint8_t value,size_t off,size_t size){hip_ok(hipMemset(span(b,t,off,size),value,size));}
static void clear(ggml_backend_buffer_t b,uint8_t value){hip_ok(hipMemset(b->context,value,b->size));}
static bool copy(ggml_backend_buffer_t b,const ggml_tensor* src,ggml_tensor* dst){
 if(!src->buffer||src->buffer->buft!=&buffer_type||ggml_nbytes(src)!=ggml_nbytes(dst))return false;
 hip_ok(hipMemcpy(span(b,dst,0,ggml_nbytes(dst)),span(src->buffer,src,0,ggml_nbytes(src)),ggml_nbytes(dst),hipMemcpyDeviceToDevice));return true;
}
static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type,size_t size){
 if(!size || size>allocation_limit)return nullptr;void* pointer=nullptr;if(hipMalloc(&pointer,size)!=hipSuccess)return nullptr;
 ggml_backend_buffer_i iface{};iface.free_buffer=free_buffer;iface.get_base=base;iface.set_tensor=set;iface.get_tensor=get;iface.memset_tensor=fill;iface.clear=clear;iface.cpy_tensor=copy;
 auto b=ggml_backend_buffer_init(type,iface,pointer,size);if(!b)hip_ok(hipFree(pointer));return b;
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
static void sync(ggml_backend_t){hip_ok(hipDeviceSynchronize());}
static void release(ggml_backend_t b){delete b;}
static ggml_status graph(ggml_backend_t,ggml_cgraph* g){++graph_calls;return ggml_graph_n_nodes(g)==0?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;}
static ggml_backend_t init(ggml_backend_dev_t d,const char*){
 if(hipInit(0)!=hipSuccess)return nullptr;int count=0;if(hipGetDeviceCount(&count)!=hipSuccess||count!=1)return nullptr;
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
ggml_backend_dev_t device_handle(){return &device;}
GateStatistics statistics(){return {gate_calls,scheduled_compute_calls,rejected,graph_calls,last_graph};}
void reset_statistics(){gate_calls=scheduled_compute_calls=rejected=graph_calls=0;last_graph=nullptr;}
}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 m0::last_graph=graph;
 ++m0::gate_calls;
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
