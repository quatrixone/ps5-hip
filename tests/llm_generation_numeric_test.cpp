#define PS5HIP_FULL_NUMERIC_NO_MAIN
#include "llm_full_numeric_test.cpp"
#include "llama.h"
#include "ggml.h"
#include "../ports/ggml/generation.h"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static std::string hex(const std::string& s){const char* d="0123456789abcdef";std::string out;for(unsigned char c:s){out+=d[c>>4];out+=d[c&15];}return out;}
template<class T>static void ids(std::ostream& out,const T* p,size_t n){out<<'[';for(size_t i=0;i<n;++i){if(i)out<<',';out<<p[i];}out<<']';}
int main(int argc,char** argv){
 if(argc!=3)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();ggml_backend_register(m0::registration());m0::enable_full({"getrows","norm","weighted","q8","q4","layer"});int status=0;
 try{
  auto mp=llama_model_default_params();mp.n_gpu_layers=99;ggml_backend_dev_t devices[]={m0::device_handle(),nullptr};llama_model_tensor_buft_override overrides[]={{"^token_embd\\.weight$",ggml_backend_dev_buffer_type(devices[0])},{nullptr,nullptr}};mp.devices=devices;mp.tensor_buft_overrides=overrides;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;mp.check_tensors=true;mp.use_extra_bufts=false;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);check(bool(model),"model");
  const auto* vocab=llama_model_get_vocab(model.get());check(vocab&&llama_vocab_n_tokens(vocab)==32000,"vocab");
  const std::string prompt="Once upon a time";llama_token tokens[ps5hip_generation::context_size];const int n=llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens,int(ps5hip_generation::context_size),true,false);check(n>0&&size_t(n)<=ps5hip_generation::context_size-4,"prompt tokenization/context budget");
  auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;cp.cb_eval=callback;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx),"context");
  const std::string folder=argv[2];output_dir=folder;m0::reset_statistics();unsigned requests=0;std::ofstream steps(folder+"/steps.jsonl"),manifest(folder+"/generation.json");
  manifest<<"{\"schema\":1,\"prompt\":\"Once upon a time\",\"add_special\":true,\"parse_special\":false,\"max_new_tokens\":4,\"vocab_size\":32000,\"prompt_tokens\":";ids(manifest,tokens,n);manifest<<",\"eog_tokens\":[";
  bool first=true;for(int t=0;t<32000;++t)if(llama_vocab_is_eog(vocab,t)){if(!first)manifest<<',';manifest<<t;first=false;}manifest<<"],\"runs\":[";
  for(int run=0;run<2;++run){
   sequence=run;llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);m0::reset_full();check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"reset metadata");
   auto decode=[&](int32_t token,int32_t position)->ps5hip_generation::Logits{
    ::decode(ctx.get(),token,position);++requests;
    const float* logits=llama_get_logits_ith(ctx.get(),-1);check(logits,"logits");
    steps<<"{\"run\":"<<run<<",\"position\":"<<position<<",\"token\":"<<token<<",\"status\":0,\"committed_position\":"<<position<<"}\n";
    return {logits,32000};
   };
   const auto result=ps5hip_generation::generate(tokens,size_t(n),4,decode,[&](int32_t t){return llama_vocab_is_eog(vocab,t);});
   if(run)manifest<<',';manifest<<"{\"run\":"<<run<<",\"processed\":";ids(manifest,result.processed.data(),result.processed_count);manifest<<",\"generated\":";ids(manifest,result.generated.data(),result.generated_count);
   manifest<<",\"stop\":\""<<(result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit")<<"\",\"pieces_hex\":[";
   std::string text;for(size_t i=0;i<result.generated_count;++i){char piece[4096];int bytes=llama_token_to_piece(vocab,result.generated[i],piece,sizeof(piece),0,true);check(bytes>=0&&bytes<=int(sizeof(piece)),"piece");std::string value(piece,size_t(bytes));if(i)manifest<<',';manifest<<'"'<<hex(value)<<'"';text+=value;}manifest<<"],\"text_hex\":\""<<hex(text)<<"\"}";
  }
  manifest<<"]}\n";steps.close();manifest.close();check(steps.good()&&manifest.good(),"metadata write");
  const auto gate=m0::statistics();assert(gate.calls==requests&&gate.forwarded==requests&&!gate.rejected&&real_scheduler_calls==requests);
  assert(captured==requests*24&&device_graph_calls>0&&cpu_arithmetic_nodes==0&&postcompute>0);
  ctx.reset();model.reset();const auto retired=m0::full_statistics();assert(retired.requests==requests&&retired.completed_nodes==uint64_t(requests)*798&&retired.launches==launches&&launches==requests*733&&retired.instances_created>0&&retired.instances_created==retired.instances_freed);
  m0::disable_full();assert(allocations.empty()&&allocation_count==free_count&&loads==unloads);
  std::ofstream counters(folder+"/backend.json");counters<<"{\"requests\":"<<requests<<",\"nodes\":"<<retired.completed_nodes<<",\"launches\":"<<launches<<",\"CPU_arithmetic\":0,\"scheduler_forwarded\":"<<real_scheduler_calls<<",\"allocations\":"<<allocation_count<<",\"frees\":"<<free_count<<",\"instances_created\":"<<retired.instances_created<<",\"instances_freed\":"<<retired.instances_freed<<"}\n";counters.close();assert(counters.good());
  std::puts("GENERATION_NUMERIC_PASS actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());status=1;}llama_backend_free();return status;
}
