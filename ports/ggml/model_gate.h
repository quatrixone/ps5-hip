#pragma once
#include "ps5_backend.h"
#include "llama.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <atomic>
namespace m0 {
struct ModelGateResult { int result; unsigned post_compute; GateStatistics gate; unsigned attempts; bool graph_reused; };
// Only the two locked upstream errors after an observed wrapper rejection are expected.
// Loader/context/teardown errors remain sticky failures, including matching text outside a call.
struct ModelGateLog {
 std::atomic<bool> active{false}, unexpected{false};
 unsigned expected=0;
 void accept(ggml_log_level level,const char* text,GateStatistics gate={}) {
  if(level!=GGML_LOG_LEVEL_ERROR)return;
  static const char* messages[]={
   "graph_compute: ggml_backend_sched_graph_compute_async failed with error -1\n",
   "process_ubatch: failed to compute graph, compute status: -1\n"};
  if(active.load() && gate.calls==1 && gate.rejected==1 && gate.forwarded==0 && expected<2 && text && std::strcmp(text,messages[expected])==0){++expected;return;}
  unexpected.store(true);std::fprintf(stderr,"M0_APP LLAMA_ERROR %s",text?text:"<null>\n");
 }
 static void callback(ggml_log_level level,const char* text,void* state) {
  auto& self=*static_cast<ModelGateLog*>(state);
  self.accept(level,text,self.active.load()?statistics():GateStatistics{});
 }
 void begin(){require(!unexpected.load());expected=0;active.store(true);}
 void finish(){active.store(false);require(!unexpected.load()&&expected==2);}
};
static bool count_completed(ggml_tensor*,bool ask,void* state){if(!ask)++*static_cast<unsigned*>(state);return true;}
inline ModelGateResult reject_actual_model(const char* path){
 std::puts("LLM_GATE PHASE load_begin");
 ModelGateLog logs;llama_log_set(ModelGateLog::callback,&logs);
 llama_backend_init();ggml_backend_register(registration());
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.check_tensors=true;
 // Negative native gate uses ordinary CPU reads/allocation; no file mapping or page locking.
 mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path,mp),llama_model_free);require(bool(model)&&!logs.unexpected.load());
 std::puts("LLM_GATE PHASE load_complete");
 std::puts("LLM_GATE PHASE context_begin");
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;
 cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 cp.offload_kqv=false;cp.op_offload=false;cp.no_perf=false;unsigned completed=0;cp.cb_eval=count_completed;cp.cb_eval_user_data=&completed;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context)&&!logs.unexpected.load());
 std::puts("LLM_GATE PHASE context_complete");
 std::puts("LLM_GATE PHASE gate_begin");
 const ggml_cgraph* first=nullptr;GateStatistics gate{};int result=0;bool reused=false;
 for(unsigned attempt=0;attempt<2;++attempt){
  auto* batch=llama_batch_ext_init(context.get());require(batch);const int index=llama_batch_ext_add_token(batch,0,1);const llama_pos pos=0;
  require(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));
  completed=0;reset_statistics();const int before=llama_perf_context(context.get()).n_reused;
  logs.begin();result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);logs.finish();llama_batch_ext_free(batch);
  const int after=llama_perf_context(context.get()).n_reused;gate=statistics();
  require(result==-3&&completed==0&&gate.calls==1&&gate.forwarded==0&&gate.rejected==1&&gate.last_graph);
  if(attempt==0){first=gate.last_graph;require(after==before);}else{reused=gate.last_graph==first&&after==before+1;require(reused);}
  std::printf("LLM_GATE ACTUAL_CLIENT attempt=%u position=0 result=%d calls=%u rejected=%u forwarded=%u post_compute=%u expected_errors=%u graph_reused=%s inference=false\n",attempt,result,gate.calls,gate.rejected,gate.forwarded,completed,logs.expected,attempt&&reused?"true":"false");
 }
 std::puts("LLM_GATE PHASE gate_complete");
 std::puts("LLM_GATE PHASE cleanup_begin");
 context.reset();model.reset();llama_backend_free();llama_log_set(nullptr,nullptr);require(!logs.unexpected.load());
 std::puts("LLM_GATE PHASE cleanup_complete");
 return {result,completed,gate,2,reused};
}
}
