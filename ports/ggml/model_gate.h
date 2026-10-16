#pragma once
#include "ps5_backend.h"
#include "llama.h"
#include <memory>
#include <cstdio>
namespace m0 {
struct ModelGateResult { int result; unsigned post_compute; GateStatistics gate; };
static bool count_completed(ggml_tensor*,bool ask,void* state){if(!ask)++*static_cast<unsigned*>(state);return true;}
inline ModelGateResult reject_actual_model(const char* path){
 llama_backend_init();ggml_backend_register(registration());
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.check_tensors=true;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path,mp),llama_model_free);require(bool(model));
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;
 cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 cp.offload_kqv=false;cp.op_offload=false;cp.no_perf=false;unsigned completed=0;cp.cb_eval=count_completed;cp.cb_eval_user_data=&completed;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context));
 auto* batch=llama_batch_ext_init(context.get());require(batch);const int index=llama_batch_ext_add_token(batch,0,1);const llama_pos pos=0;
 require(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));
 reset_statistics();const int result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);
 auto gate=statistics();require(result!=0&&completed==0&&gate.calls==1&&gate.forwarded==0&&gate.rejected==1);
 context.reset();model.reset();llama_backend_free();
 std::printf("LLM_GATE ACTUAL_CLIENT result=%d calls=%u rejected=%u forwarded=%u post_compute=%u inference=false\n",result,gate.calls,gate.rejected,gate.forwarded,completed);
 return {result,completed,gate};
}
}
