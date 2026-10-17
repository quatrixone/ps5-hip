#include "layer_qualification.h"
#include "llm_layer_numeric_mock.h"
#include <fstream>
static const char* output_dir=nullptr;
static unsigned sequence=0;
#include "layer_dispatch.h"
#include "ggml-impl.h"
#undef DEPRECATED
#include "llama.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
static std::map<void*,size_t> allocations;
static unsigned loads=0,unloads=0,launches=0,clears=0;
static size_t uploaded=0;
extern "C" hipError_t hipInit(unsigned){return hipSuccess;}
extern "C" hipError_t hipGetDeviceCount(int* n){*n=1;return hipSuccess;}
extern "C" hipError_t hipModuleLoad(hipModule_t* m,const char*){*m=reinterpret_cast<hipModule_t>(++loads);return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t,const char* name){
 for(unsigned i=0;i<5;++i)if(std::strcmp(name,ps5hip_prefix::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+1);return hipSuccess;}
 for(unsigned i=0;i<12;++i)if(std::strcmp(name,ps5hip_layer::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+6);return hipSuccess;}
 return hipErrorInvalidValue;
}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloads;return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){*p=std::aligned_alloc(256,(n+255)&~size_t(255));assert(*p);std::memset(*p,0,n);assert(allocations.emplace(*p,n).second);return hipSuccess;}
extern "C" hipError_t hipFree(void* p){assert(allocations.erase(p)==1);std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind kind){if(kind==hipMemcpyHostToDevice)uploaded+=n;std::memmove(d,s,n);return hipSuccess;}
extern "C" hipError_t hipMemset(void* p,int value,size_t n){++clears;std::memset(p,value,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=17&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;layer_numeric::execute(unsigned(uintptr_t(f)),gx*bx,args);return hipSuccess;}
extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason){std::fprintf(stderr,"TEST_TERMINAL %s\n",reason);std::abort();}
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*){assert(false&&"CPU graph forwarding forbidden");return GGML_STATUS_FAILED;}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s,ggml_cgraph* g){return __wrap_ggml_backend_sched_graph_compute_async(s,g);}
static bool owned(const ggml_tensor* t){if(!t||!t->data)return false;const auto p=uintptr_t(t->data);for(const auto& item:allocations){const auto base=uintptr_t(item.first);if(p>=base&&p-base<=item.second&&ggml_nbytes(t)<=item.second-(p-base))return true;}return false;}
static void fail(const char*,hipError_t){assert(false);}
struct Observed{unsigned count=0;bool negatives=false;};
static void readback(const ps5hip_layer::Graph& graph,const m0::LayerRequest& request,void* opaque){
 auto& result=*static_cast<Observed*>(opaque);assert(request.position==int32_t(result.count));assert(graph.extent==(request.position<256?256:512));
 for(auto* t:graph.outputs){assert(owned(t)&&t->type==GGML_TYPE_F32);std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));for(float f:values)assert(std::isfinite(f));size_t ni=0;while(ni<graph.nodes.size()&&graph.nodes[ni]!=t)++ni;assert(ni<graph.nodes.size());std::ofstream file(std::string(output_dir)+"/s"+std::to_string(sequence)+".p"+std::to_string(request.position)+".n"+std::to_string(ni)+".bin",std::ios::binary);file.write(reinterpret_cast<const char*>(values.data()),values.size()*4);assert(file.good());}
 for(unsigned which=0;which<2;++which){
  auto* t=graph.nodes[which?16:12];std::vector<_Float16> cache(131072);ggml_backend_tensor_get(t,cache.data(),0,cache.size()*2);
  for(unsigned feature=0;feature<256;feature++)for(unsigned slot=unsigned(request.position)+1;slot<512;slot++)assert(cache[which?feature*512+slot:slot*256+feature]==0);
  if(request.position==511||(sequence==1&&request.position==1)){std::ofstream file(std::string(output_dir)+"/run-"+std::to_string(sequence)+"-cache-"+(which?"v":"k")+".f16le",std::ios::binary);file.write(reinterpret_cast<const char*>(cache.data()),cache.size()*2);assert(file.good());}
 }
 if(!result.negatives){
  ps5hip_layer::Dispatch probe(owned,fail);probe.expected_position=unsigned(request.position);const auto before=launches;assert(probe.preflight(graph.graph));
  auto extra=*graph.graph;extra.n_nodes++;assert(!probe.preflight(&extra));
  auto* token=graph.uploads[0].clone;int32_t invalid_token=32000;ggml_backend_tensor_set(token,&invalid_token,0,4);assert(!probe.preflight(graph.graph));ggml_backend_tensor_set(token,&request.token,0,4);
  auto* pos=graph.uploads[1].clone;int32_t wrong=request.position+1;ggml_backend_tensor_set(pos,&wrong,0,4);assert(!probe.preflight(graph.graph));ggml_backend_tensor_set(pos,&request.position,0,4);
  auto* k=graph.uploads[2].clone;int64_t bad=511,slot=request.position;ggml_backend_tensor_set(k,&bad,0,8);assert(!probe.preflight(graph.graph));ggml_backend_tensor_set(k,&slot,0,8);
  auto* v=graph.uploads[3].clone;bad=99;ggml_backend_tensor_set(v,&bad,0,8);assert(!probe.preflight(graph.graph));ggml_backend_tensor_set(v,&slot,0,8);
  auto* mask=graph.uploads[4].clone;float nan=std::numeric_limits<float>::quiet_NaN(),zero=0;ggml_backend_tensor_set(mask,&nan,0,4);assert(!probe.preflight(graph.graph));ggml_backend_tensor_set(mask,&zero,0,4);
  auto* weight=graph.nodes[0]->src[0];void* saved=weight->data;weight->data=reinterpret_cast<void*>(uintptr_t(1));assert(!probe.preflight(graph.graph));weight->data=saved;
  assert(probe.preflight(graph.graph)&&launches==before);result.negatives=true;
 }
 ++result.count;
}
static bool completed(ggml_tensor*,bool ask,void* pointer){if(!ask)++*static_cast<unsigned*>(pointer);return true;}
int main(int argc,char** argv){
 assert(argc==3||argc==4);output_dir=argv[2];const unsigned steps=argc==4?unsigned(std::stoul(argv[3])):4;assert(steps==4||steps==512);llama_backend_init();ggml_backend_register(m0::registration());auto backend=ggml_backend_dev_init(m0::device_handle(),nullptr);assert(backend);
 m0::configure_layer(backend,{"getrows","norm","weighted","q8","q4","layer"});
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;ggml_backend_dev_t devices[]={m0::device_handle(),nullptr};mp.devices=devices;
 auto* model=llama_model_load_from_file(argv[1],mp);assert(model);auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;unsigned postcompute=0;cp.cb_eval=completed;cp.cb_eval_user_data=&postcompute;auto* context=llama_init_from_model(model,cp);assert(context);
 std::vector<m0::LayerRequest> requests;for(int i=0;i<512;++i)requests.push_back({i%4==3?31999:i%4,i});
 size_t weight_bytes=0;uint64_t expected_uploads=0;unsigned expected_total=0;
 for(unsigned count:{steps,2u}){
  if(count==2){const auto before=clears;m0::reset_layer_cache(backend);assert(clears==before+2&&m0::layer_statistics(backend).request_cursor==0);}
  Observed observed;auto* batch=llama_batch_ext_init(context);assert(batch);const int index=llama_batch_ext_add_token(batch,0,0);const llama_pos position=0;assert(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true));m0::reset_statistics();
  {m0::ActualLayerQualification scope(backend,requests.data(),count,readback,&observed);assert(llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch)==-3);}
  llama_batch_ext_free(batch);assert(observed.count==count&&observed.negatives);
  const auto gate=m0::statistics();assert(gate.calls==1&&gate.rejected==1&&gate.forwarded==0&&gate.direct_graph_calls==count&&postcompute==0);
  const auto stats=m0::layer_statistics(backend);if(!weight_bytes)weight_bytes=stats.uploaded_weight_bytes;assert(weight_bytes>0&&stats.uploaded_weight_bytes==weight_bytes);
  for(unsigned i=0;i<count;++i)expected_uploads+=2064+4*(i<256?256:512);expected_total+=count;
  ++sequence;assert(stats.uploaded_request_bytes==expected_uploads&&stats.invocations==expected_total&&stats.request_cursor==count&&stats.last_position==int32_t(count-1)&&stats.completed_launches==launches);
 }
 llama_free(context);llama_model_free(model);const auto stats=m0::layer_statistics(backend);assert(stats.cache_resets==1&&stats.invocations==steps+2);const auto calls=m0::disable_layer(backend);ggml_backend_free(backend);llama_backend_free();
 assert(allocations.empty()&&loads==6&&unloads==6);assert(calls[0].successes==6&&calls[1].successes==17&&calls[2].successes==2&&calls[7].successes==2&&calls[2].requested_bytes==6336+32768&&calls[7].requested_bytes==6336+32768);
 std::printf("PASS actual-model layer:numerical mocked requests,full sequence+reset2,weights_once=%zu,request_bytes=%llu,launches=%u,noCPUforwarding; GPU unproven; numerical outputs exported\n",weight_bytes,static_cast<unsigned long long>(expected_uploads),launches);
}

