#include "llm_layer_hip_mock.h"
static bool owned(const ggml_tensor* t){if(!t||!t->data)return false;const auto p=uintptr_t(t->data);for(const auto& item:allocations){const auto base=uintptr_t(item.first);if(p>=base&&p-base<=item.second&&ggml_nbytes(t)<=item.second-(p-base))return true;}return false;}
static void fail(const char*,hipError_t){assert(false);}
struct Observed{unsigned count=0;bool negatives=false;std::array<ggml_backend_buffer_t,12> resident{};};
static void readback(const ps5hip_layer::Graph& graph,const m0::LayerRequest& request,void* opaque){
 auto& result=*static_cast<Observed*>(opaque);assert(request.position==int32_t(result.count));assert(graph.extent==(request.position<256?256:512));
 const auto selected=ps5hip_layer::select(graph.graph);
 std::array<ggml_backend_buffer_t,12> resident{};
 for(size_t i=0;i<resident.size();++i){
  auto* t=i<10?selected.weights[i]:selected.caches[i-10];resident[i]=t->buffer;
  assert(resident[i]&&ggml_backend_buffer_get_base(resident[i])==t->data);
  assert(ggml_backend_buffer_get_size(resident[i])==ggml_nbytes(t));
  for(size_t j=0;j<i;++j)assert(resident[i]!=resident[j]);
 }
 if(!result.count)result.resident=resident;else assert(result.resident==resident);
 for(auto* t:graph.outputs){assert(owned(t)&&t->type==GGML_TYPE_F32);std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));for(float f:values)assert(f==0);}
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
 assert(argc==2);llama_backend_init();ggml_backend_register(m0::registration());auto backend=ggml_backend_dev_init(m0::device_handle(),nullptr);assert(backend);
 m0::configure_layer(backend,{"getrows","norm","weighted","q8","q4","layer"});
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;ggml_backend_dev_t devices[]={m0::device_handle(),nullptr};mp.devices=devices;
 auto* model=llama_model_load_from_file(argv[1],mp);assert(model);auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;unsigned postcompute=0;cp.cb_eval=completed;cp.cb_eval_user_data=&postcompute;auto* context=llama_init_from_model(model,cp);assert(context);
 std::vector<m0::LayerRequest> requests;for(int i=0;i<512;++i)requests.push_back({i%4==3?31999:i%4,i});
 size_t weight_bytes=0;uint64_t expected_uploads=0;unsigned expected_total=0;
 for(unsigned count:{512u,2u}){
  if(count==2){const auto before=clears;m0::reset_layer_cache(backend);assert(clears==before+2&&m0::layer_statistics(backend).request_cursor==0);}
  Observed observed;auto* batch=llama_batch_ext_init(context);assert(batch);const int index=llama_batch_ext_add_token(batch,0,0);const llama_pos position=0;assert(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true));m0::reset_statistics();
  {m0::ActualLayerQualification scope(backend,requests.data(),count,readback,&observed);assert(llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch)==-3);}
  llama_batch_ext_free(batch);assert(observed.count==count&&observed.negatives);
  const auto gate=m0::statistics();assert(gate.calls==1&&gate.rejected==1&&gate.forwarded==0&&gate.direct_graph_calls==count&&postcompute==0);
  const auto stats=m0::layer_statistics(backend);if(!weight_bytes)weight_bytes=stats.uploaded_weight_bytes;assert(weight_bytes>0&&stats.uploaded_weight_bytes==weight_bytes);
  for(unsigned i=0;i<count;++i)expected_uploads+=2064+4*(i<256?256:512);expected_total+=count;
  assert(stats.uploaded_request_bytes==expected_uploads&&stats.invocations==expected_total&&stats.request_cursor==count&&stats.last_position==int32_t(count-1)&&stats.completed_launches==launches);
 }
 llama_free(context);llama_model_free(model);const auto stats=m0::layer_statistics(backend);assert(stats.cache_resets==1&&stats.invocations==514);const auto calls=m0::disable_layer(backend);ggml_backend_free(backend);llama_backend_free();
 assert(allocations.empty()&&loads==6&&unloads==6);assert(calls[0].successes==6&&calls[1].successes==17&&calls[2].successes==2&&calls[7].successes==2&&calls[2].requested_bytes==6336+32768&&calls[7].requested_bytes==6336+32768);
 std::printf("PASS actual-model layer:514 mocked requests,512+reset2,weights_once=%zu,request_bytes=%llu,launches=%u,noCPUforwarding; GPU/numerics unproven\n",weight_bytes,static_cast<unsigned long long>(expected_uploads),launches);
}
