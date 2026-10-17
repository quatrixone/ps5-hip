// CPU-only oracle: actual tokenizer, committed decoder calls and greedy feedback.
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
 if(argc!=3)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();int status=0;
 try{
  auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);check(bool(model),"model");
  const auto* vocab=llama_model_get_vocab(model.get());check(vocab&&llama_vocab_n_tokens(vocab)==32000,"vocab");
  const std::string prompt="Once upon a time";llama_token tokens[32];const int n=llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens,32,true,false);check(n>0&&n<=32,"prompt tokenization");
  auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx)&&llama_n_ctx(ctx.get())==512,"context");
  const std::string folder=argv[2];std::ofstream steps(folder+"/steps.jsonl"),manifest(folder+"/generation.json");
  manifest<<"{\"schema\":1,\"prompt\":\"Once upon a time\",\"add_special\":true,\"parse_special\":false,\"max_new_tokens\":4,\"vocab_size\":32000,\"prompt_tokens\":";ids(manifest,tokens,n);manifest<<",\"eog_tokens\":[";
  bool first=true;for(int t=0;t<32000;++t)if(llama_vocab_is_eog(vocab,t)){if(!first)manifest<<',';manifest<<t;first=false;}manifest<<"],\"runs\":[";
  for(int run=0;run<2;++run){
   llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"reset metadata");
   auto decode=[&](int32_t token,int32_t position)->ps5hip_generation::Logits{
    auto* batch=llama_batch_ext_init(ctx.get());check(batch,"batch");const int index=llama_batch_ext_add_token(batch,0,token);const llama_pos pos=position;
    check(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true),"controls");
    const int rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);check(rc==0,"CPU decode");check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==position,"committed position");
    const float* logits=llama_get_logits_ith(ctx.get(),-1);check(logits,"logits");
    std::ofstream out(folder+"/run-"+std::to_string(run)+"-pos-"+std::to_string(position)+"-logits.f32le",std::ios::binary);out.write(reinterpret_cast<const char*>(logits),32000*4);out.close();check(out.good(),"logit write");
    steps<<"{\"run\":"<<run<<",\"position\":"<<position<<",\"token\":"<<token<<",\"status\":0,\"committed_position\":"<<position<<"}\n";
    return {logits,32000};
   };
   const auto result=ps5hip_generation::generate(tokens,size_t(n),4,decode,[&](int32_t t){return llama_vocab_is_eog(vocab,t);});
   if(run)manifest<<',';manifest<<"{\"run\":"<<run<<",\"processed\":";ids(manifest,result.processed.data(),result.processed_count);manifest<<",\"generated\":";ids(manifest,result.generated.data(),result.generated_count);
   manifest<<",\"stop\":\""<<(result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit")<<"\",\"pieces_hex\":[";
   std::string text;for(size_t i=0;i<result.generated_count;++i){char piece[4096];int bytes=llama_token_to_piece(vocab,result.generated[i],piece,sizeof(piece),0,true);check(bytes>=0&&bytes<=int(sizeof(piece)),"piece");std::string value(piece,size_t(bytes));if(i)manifest<<',';manifest<<'"'<<hex(value)<<'"';text+=value;}manifest<<"],\"text_hex\":\""<<hex(text)<<"\"}";
  }
  manifest<<"]}\n";steps.close();manifest.close();check(steps.good()&&manifest.good(),"metadata write");
  std::puts("GENERATION_REFERENCE_PASS runs=2 max_new_tokens=4 original_cpu_decode=true GPU_execution=false");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());status=1;}llama_backend_free();return status;
}
