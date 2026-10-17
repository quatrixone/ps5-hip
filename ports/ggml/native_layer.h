#pragma once
#include "native_prefix.h"
#include "model_layer.h"
#include "layer_qualification.h"
#include <vector>

namespace m0 {
struct NativeLayerResult {LayerStatistics layer;PrefixCallStatistics calls;unsigned files;size_t bytes;};
struct NativeLayerSink {
 PrefixWrite write;void* user;PrefixLogSync sync;unsigned run=0,files=0;size_t bytes=0;
 std::vector<float> values=std::vector<float>(65536);
 void dump(const char* name,const ggml_tensor* tensor,bool f32) {
  const size_t size=ggml_nbytes(tensor);require(size<=values.size()*sizeof(float));
  ggml_backend_tensor_get(tensor,values.data(),0,size);
  if(f32)for(size_t i=0;i<size/4;++i)require(std::isfinite(values[i]));
  write(name,values.data(),size,user);++files;bytes+=size;
  std::printf("M2_LAYER RAW file=%s bytes=%zu exclusive=true synced=true\n",name,size);require(sync());
 }
 static void readback(const ps5hip_layer::Graph& graph,const LayerRequest& request,void* data) {
  auto& state=*static_cast<NativeLayerSink*>(data);const int p=request.position;
  if(p!=0&&p!=1&&p!=255&&p!=256&&p!=511)return;
  require(graph.outputs.size()==21);
  for(size_t i=0;i<graph.outputs.size();++i) {
   const auto* t=graph.outputs[i];require(t&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t));
   char name[96];const int n=std::snprintf(name,sizeof(name),"run-%u-pos-%d-out-%02u.f32le",state.run,p,unsigned(i));require(n>0&&size_t(n)<sizeof(name));state.dump(name,t,true);
  }
  if(p==511||(state.run==1&&p==1))for(unsigned i=0;i<2;++i) {
   const auto* t=graph.nodes[i?16:12];require(t&&t->type==GGML_TYPE_F16&&ggml_nbytes(t)==262144);
   char name[96];const int n=std::snprintf(name,sizeof(name),"run-%u-cache-%c.f16le",state.run,i?'v':'k');require(n>0&&size_t(n)<sizeof(name));state.dump(name,t,false);
  }
  std::printf("M2_LAYER CHECKPOINT run=%u position=%d token=%d extent=%lld nodes=37\n",state.run,p,request.token,static_cast<long long>(graph.extent));require(state.sync());
 }
};
inline NativeLayerResult run_native_layer(const char* model_path,const std::array<const char*,6>& modules,
 PrefixWrite writer,void* writer_state,PrefixLogSync log_sync,void (*memory_snapshot)(const char*)=nullptr) {
 require(model_path&&writer&&log_sync);
 std::puts("M2_LAYER PHASE load_begin");if(memory_snapshot)memory_snapshot("before_load");
 ModelGateLog logs;llama_log_set(ModelGateLog::callback,&logs);llama_backend_init();ggml_backend_register(registration());
 auto backend=ggml_backend_dev_init(device_handle(),nullptr);require(backend);configure_layer(backend,modules);
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.check_tensors=true;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path,mp),llama_model_free);require(bool(model)&&!logs.unexpected.load());
 std::puts("M2_LAYER PHASE load_complete");if(memory_snapshot)memory_snapshot("after_load");std::puts("M2_LAYER PHASE context_begin");
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;
 cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;
 unsigned completed=0;cp.cb_eval=count_completed;cp.cb_eval_user_data=&completed;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context)&&!logs.unexpected.load());
 std::puts("M2_LAYER PHASE context_complete");if(memory_snapshot)memory_snapshot("after_context");
 unsigned files=0;size_t bytes=0;
 {
  NativeLayerSink sink{writer,writer_state,log_sync};std::array<LayerRequest,512> requests{};
  constexpr int32_t tokens[]={0,1,2,31999};for(size_t i=0;i<requests.size();++i)requests[i]={tokens[i%4],int32_t(i)};
  for(unsigned run=0;run<2;++run) {
   sink.run=run;if(run)reset_layer_cache(backend);const size_t count=run?2:512;
   auto* batch=llama_batch_ext_init(context.get());require(batch);int index=llama_batch_ext_add_token(batch,0,0);const llama_pos position=0;
   require(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true));
   reset_statistics();completed=0;logs.begin();int result;
   {ActualLayerQualification qualification(backend,requests.data(),count,NativeLayerSink::readback,&sink);result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);}
   logs.finish();llama_batch_ext_free(batch);auto gate=statistics();auto state=layer_statistics(backend);
   require(result==-3&&completed==0&&gate.calls==1&&gate.rejected==1&&gate.forwarded==0&&gate.direct_graph_calls==count&&state.request_cursor==count);
   require(state.invocations==(run?514:512)&&state.completed_launches==state.invocations*33);
   std::printf("M2_LAYER SEQUENCE run=%u requests=%zu original_result=-3 rejected=1 forwarded=0 post_compute=0 expected_errors=2 launches=%zu control_source=declared layer_only=true inference=false\n",run,count,count*33);require(log_sync());
  }
  files=sink.files;bytes=sink.bytes;
 }
 auto state=layer_statistics(backend);require(state.invocations==514&&state.completed_launches==16962&&state.cache_resets==1&&state.uploaded_weight_bytes==61652992&&state.uploaded_request_bytes==1849376&&files==151);
 std::puts("M2_LAYER PHASE cleanup_begin");context.reset();model.reset();auto calls=disable_layer(backend);ggml_backend_free(backend);llama_backend_free();llama_log_set(nullptr,nullptr);require(!logs.unexpected.load());
 const char* sites[]={"module_load","module_function","scratch_allocate","token_read","launch","completion_sync","release_sync","scratch_free","module_unload"};
 const uint64_t counts[]={6,17,2,3084,16962,16962,1,2,6};
 for(size_t i=0;i<calls.size();++i) {
  const auto& c=calls[i];const uint64_t nbytes=i==2||i==7?39104:i==3?1851432:0;
  require(c.attempts==counts[i]&&c.completions==counts[i]&&c.successes==counts[i]&&!c.failures&&c.requested_bytes==nbytes&&!c.last_result&&!c.first_error);
  std::printf("M2_LAYER HIP_CALL site=%s attempts=%llu completions=%llu successes=%llu failures=0 bytes=%llu last_result=0 first_error=0\n",sites[i],static_cast<unsigned long long>(c.attempts),static_cast<unsigned long long>(c.completions),static_cast<unsigned long long>(c.successes),static_cast<unsigned long long>(c.requested_bytes));
 }
 std::puts("M2_LAYER PHASE cleanup_complete");if(memory_snapshot)memory_snapshot("after_cleanup");
 std::printf("M2_LAYER TRANSPORT_COMPLETE requests=514 nodes=37 launches=16962 raw_files=%u raw_bytes=%zu uploaded_weight_bytes=61652992 uploaded_request_bytes=1849376 cache_resets=1 numerical=pending inference=false finalization=pending\n",files,bytes);require(log_sync());
 return {state,calls,files,bytes};
}
}
