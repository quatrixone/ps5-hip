#pragma once
#include "native_prefix.h"
#include "full_qualification.h"
#include <vector>

namespace m0 {
struct NativeFullResult {FullStatistics full;unsigned files;size_t bytes;};
struct NativeFullSink {
 PrefixWrite write;void* user;PrefixLogSync sync;unsigned run=0,position=0,files=0;size_t bytes=0;
 std::vector<float> values=std::vector<float>(32000);
 bool checkpoint() const{return run||position==0||position==1||position==255||position==256||position==511;}
 static int boundary(const char* name){
  for(unsigned l=0;l<22;++l){char expected[24];std::snprintf(expected,sizeof(expected),"l_out-%u",l);if(!std::strcmp(name,expected))return int(l);}
  if(!std::strcmp(name,"result_norm"))return 22;
  if(!std::strcmp(name,"result_output"))return 23;
  return -1;
 }
 static bool callback(ggml_tensor* t,bool ask,void* opaque){
  auto& s=*static_cast<NativeFullSink*>(opaque);const int index=boundary(t->name);
  if(ask)return s.checkpoint()&&index>=0;
  require(s.checkpoint()&&index>=0&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t));
  const size_t size=ggml_nbytes(t);require(size==(index==23?128000:8192)&&size<=s.values.size()*sizeof(float));
  ggml_backend_tensor_get(t,s.values.data(),0,size);for(size_t i=0;i<size/4;++i)require(std::isfinite(s.values[i]));
  char stage[24];if(index<22)std::snprintf(stage,sizeof(stage),"layer-%d",index);else std::snprintf(stage,sizeof(stage),"%s",index==22?"norm":"logits");
  char name[96];const int n=std::snprintf(name,sizeof(name),"run-%u-pos-%u-%s.f32le",s.run,s.position,stage);require(n>0&&size_t(n)<sizeof(name));
  s.write(name,s.values.data(),size,s.user);++s.files;s.bytes+=size;
  std::printf("M3_FULL RAW file=%s bytes=%zu synced=true\n",name,size);require(s.sync());
  return true;
 }
};
inline NativeFullResult run_native_full(const char* model_path,const std::array<const char*,6>& modules,
 PrefixWrite writer,void* writer_state,PrefixLogSync log_sync,void (*memory_snapshot)(const char*)=nullptr){
 require(model_path&&writer&&log_sync);std::puts("M3_FULL PHASE load_begin");if(memory_snapshot)memory_snapshot("before_load");
 ModelGateLog logs;llama_log_set(ModelGateLog::callback,&logs);llama_backend_init();ggml_backend_register(registration());enable_full(modules);
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.use_extra_bufts=false;mp.check_tensors=true;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 llama_model_tensor_buft_override overrides[]={{"^token_embd\\.weight$",ggml_backend_dev_buffer_type(devices[0])},{nullptr,nullptr}};mp.tensor_buft_overrides=overrides;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path,mp),llama_model_free);require(bool(model)&&!logs.unexpected.load());
 std::puts("M3_FULL PHASE load_complete");if(memory_snapshot)memory_snapshot("after_load");
 NativeFullSink sink{writer,writer_state,log_sync};auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;
 cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;cp.cb_eval=NativeFullSink::callback;cp.cb_eval_user_data=&sink;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context)&&!logs.unexpected.load());
 std::puts("M3_FULL PHASE context_complete");if(memory_snapshot)memory_snapshot("after_context");reset_statistics();
 constexpr int32_t tokens[]={0,1,2,31999};
 for(unsigned run=0;run<2;++run){
  sink.run=run;if(run){llama_synchronize(context.get());llama_memory_clear(llama_get_memory(context.get()),true);reset_full();}
  require(llama_memory_seq_pos_max(llama_get_memory(context.get()),0)==-1);
  for(unsigned p=0;p<(run?2u:512u);++p){
   sink.position=p;auto* batch=llama_batch_ext_init(context.get());require(batch);
   const int index=llama_batch_ext_add_token(batch,0,tokens[p%4]);const llama_pos position=p;
   require(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true));
   const unsigned before=sink.files;const int result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);
   require(result==0&&!logs.unexpected.load()&&llama_memory_seq_pos_min(llama_get_memory(context.get()),0)==0&&llama_memory_seq_pos_max(llama_get_memory(context.get()),0)==position);
   require(sink.files-before==(sink.checkpoint()?24u:0u));const float* logits=llama_get_logits_ith(context.get(),-1);require(logits);for(unsigned i=0;i<32000;++i)require(std::isfinite(logits[i]));
   if(sink.checkpoint()){std::printf("M3_FULL CHECKPOINT run=%u position=%u token=%d committed_position=%u layers=22 logits=32000\n",run,p,tokens[p%4],p);require(log_sync());}
  }
 }
 const auto gate=statistics();require(gate.calls==514&&gate.forwarded==514&&gate.rejected==0);
 require(sink.files==168&&sink.bytes==2214912);std::puts("M3_FULL PHASE cleanup_begin");context.reset();model.reset();const auto state=full_statistics();
 require(state.requests==514&&state.completed_nodes==410172&&state.launches==376762&&state.instances_created>0&&state.instances_created==state.instances_freed);
 disable_full();llama_backend_free();llama_log_set(nullptr,nullptr);require(!logs.unexpected.load());if(memory_snapshot)memory_snapshot("after_cleanup");
 std::printf("M3_FULL TRANSPORT_COMPLETE requests=514 layers=22 logits=32000 nodes=410172 launches=376762 raw_files=%u raw_bytes=%zu instances=%u/%u numerical=pending inference=false finalization=pending\n",sink.files,sink.bytes,state.instances_created,state.instances_freed);require(log_sync());
 return {state,sink.files,sink.bytes};
}
}
