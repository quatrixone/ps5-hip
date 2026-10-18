#ifdef PS5HIP_ORIGINAL_MODEL
#include "../ports/ggml/tinyllama_source_metadata.h"
#include "../ports/ggml/tinyllama_tensor_callback.h"
#endif
// CPU-only oracle: actual tokenizer, committed decoder calls and greedy feedback.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstring>
#include "../ports/ggml/generation.h"
#include "../ports/ggml/extended_generation.h"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static std::string hex(const std::string& s){const char* d="0123456789abcdef";std::string out;for(unsigned char c:s){out+=d[c>>4];out+=d[c&15];}return out;}
template<class T>static void ids(std::ostream& out,const T* p,size_t n){out<<'[';for(size_t i=0;i<n;++i){if(i)out<<',';out<<p[i];}out<<']';}
struct Trace{std::string folder;int run=-1,position=-1;unsigned captured=0;bool failed=false;std::array<bool,24> seen{};const ps5hip_generation::ExtendedCase* extended=nullptr;};
static bool checkpoint(const Trace& t){return t.run>=0&&t.position>=0;}
static std::string stem(const Trace& t){return t.folder+"/run-"+std::to_string(t.run)+"-pos-"+std::to_string(t.position);}
static bool capture(ggml_tensor* tensor,bool ask,void* pointer){
 if(ask)return true;auto& state=*static_cast<Trace*>(pointer);if(!checkpoint(state))return true;
 int index=-1;for(int layer=0;layer<22;++layer)if(std::string(tensor->name)=="l_out-"+std::to_string(layer))index=layer;
 if(!std::strcmp(tensor->name,"result_norm"))index=22;
 if(!std::strcmp(tensor->name,"result_output"))index=23;
 if(index<0)return true;
 if(state.extended&&index!=23&&!ps5hip_generation::extended_checkpoint(*state.extended,unsigned(state.position)))return true;
 try{
  check(!state.seen[index],"duplicate boundary callback");check(tensor->type==GGML_TYPE_F32&&ggml_is_contiguous(tensor)&&ggml_nelements(tensor)==(index==23?32000:2048),"boundary layout");
  std::vector<float> values(size_t(ggml_nelements(tensor)));ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);for(float v:values)check(std::isfinite(v),"nonfinite boundary");
  const std::string name=index<22?"layer-"+std::to_string(index):index==22?"norm":"logits";
  std::ofstream out(stem(state)+"-"+name+".f32le",std::ios::binary);out.write((const char*)values.data(),values.size()*4);out.close();check(out.good(),"boundary write");state.seen[index]=true;++state.captured;
 }catch(const std::exception& e){std::fprintf(stderr,"capture: %s\n",e.what());state.failed=true;return false;}return true;
}
int main(int argc,char** argv){
 const bool corpus=argc==4&&!std::strcmp(argv[3],"corpus");
 const bool extended=argc==4&&!std::strcmp(argv[3],"extended");
 bool original=false;
#ifdef PS5HIP_ORIGINAL_MODEL
 original=argc==4&&!std::strcmp(argv[3],"--original-source");
#endif
 if(argc!=3&&!corpus&&!extended&&!original)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();int status=0;
 try{
  auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
  #ifdef PS5HIP_ORIGINAL_MODEL
  direct_tiny::Metadata metadata(nullptr,gguf_free);
  std::unique_ptr<ps5hip::SafetensorsReader> source;
  std::unique_ptr<ps5hip::TinyLlamaTensorCallback> callback;
#endif
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(nullptr,llama_model_free);
#ifdef PS5HIP_ORIGINAL_MODEL
  if(original){
   metadata=direct_tiny::construct(argv[1]);
   auto config=direct_tiny::json(direct_tiny::read(std::filesystem::path(argv[1])/"config.json",65536));
   source=std::make_unique<ps5hip::SafetensorsReader>((std::filesystem::path(argv[1])/"model.safetensors").string());
   callback=std::make_unique<ps5hip::TinyLlamaTensorCallback>(*source,config);callback->add_descriptors(metadata.get());
   mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
   model.reset(llama_model_init_from_user(metadata.get(),ps5hip::TinyLlamaTensorCallback::callback,callback.get(),mp));
   check(bool(model)&&callback->complete(),"complete201 original model");
  }else{
#endif
   model.reset(llama_model_load_from_file(argv[1],mp));
#ifdef PS5HIP_ORIGINAL_MODEL
  }
#endif
  check(bool(model),"model");
  const auto* vocab=llama_model_get_vocab(model.get());check(vocab&&llama_vocab_n_tokens(vocab)==32000,"vocab");
  Trace state;state.folder=argv[2];
  auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;cp.cb_eval=capture;cp.cb_eval_user_data=&state;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx)&&llama_n_ctx(ctx.get())==512,"context");
  struct Case{const char* id;const char* prompt;};
  const Case cases[]={
#define PS5HIP_GENERATION_CASE(id,prompt) {id,prompt},
#include "../ports/ggml/generation_cases.def"
#undef PS5HIP_GENERATION_CASE
  };
  const unsigned case_count=extended?3:corpus?unsigned(sizeof(cases)/sizeof(cases[0])):1;
  const size_t budget=extended?64:corpus?16:4;
  for(unsigned case_index=0;case_index<case_count;++case_index){
  const auto* policy=extended?&ps5hip_generation::extended_cases[case_index]:nullptr;
  const Case selected=policy?Case{policy->id,policy->prompt}:cases[case_index];const std::string prompt=selected.prompt;state.extended=policy;
  llama_token tokens[ps5hip_generation::context_size];const int n=llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens,int(ps5hip_generation::context_size),true,false);check(n>0&&size_t(n)<=ps5hip_generation::context_size-budget,"prompt tokenization/context budget");
  check(!policy||size_t(n)==policy->prompt_count,"extended prompt tokenization");
  const std::string folder=std::string(argv[2])+((corpus||extended)?std::string("/")+selected.id:"");
  if(corpus||extended)check(std::filesystem::create_directory(folder),"fresh corpus case directory");
  state.folder=folder;
  std::ofstream steps(folder+"/steps.jsonl"),manifest(folder+"/generation.json");
  manifest<<"{\"schema\":1,\"prompt\":\"";
  for(char c:prompt){if(c=='\n')manifest<<"\\n";else if(c=='\"'||c=='\\')manifest<<'\\'<<c;else manifest<<c;}
  manifest<<"\",\"add_special\":true,\"parse_special\":false,\"max_new_tokens\":"<<budget<<",\"vocab_size\":32000,\"prompt_tokens\":";ids(manifest,tokens,n);manifest<<",\"eog_tokens\":[";
  bool first=true;for(int t=0;t<32000;++t)if(llama_vocab_is_eog(vocab,t)){if(!first)manifest<<',';manifest<<t;first=false;}manifest<<"],\"runs\":[";
  for(int run=0;run<2;++run){
   llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"reset metadata");
   state.run=run;
   auto decode=[&](int32_t token,int32_t position)->ps5hip_generation::Logits{
    state.position=position;state.seen={};
    auto* batch=llama_batch_ext_init(ctx.get());check(batch,"batch");const int index=llama_batch_ext_add_token(batch,0,token);const llama_pos pos=position;
    check(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true),"controls");
    const int rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);check(rc==0&&!state.failed,"CPU decode");for(unsigned stage=0;stage<24;++stage)check(state.seen[stage]==(!policy||stage==23||ps5hip_generation::extended_checkpoint(*policy,unsigned(position))),"missing/unexpected boundary callback");check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==position,"committed position");
    const float* logits=llama_get_logits_ith(ctx.get(),-1);check(logits,"logits");
    std::ifstream captured(stem(state)+"-logits.f32le",std::ios::binary);std::vector<float> sampled(32000);captured.read(reinterpret_cast<char*>(sampled.data()),128000);check(captured.good()&&!std::memcmp(sampled.data(),logits,128000),"sampler logits differ from callback");
    steps<<"{\"run\":"<<run<<",\"position\":"<<position<<",\"token\":"<<token<<",\"status\":0,\"committed_position\":"<<position<<"}\n";
    return {logits,32000};
   };
   const auto result=ps5hip_generation::generate(tokens,size_t(n),budget,decode,[&](int32_t t){return llama_vocab_is_eog(vocab,t);});
   check(!corpus||(result.generated_count==16&&result.stop==ps5hip_generation::StopReason::step_limit),"corpus requires sixteen tokens without EOS");
   check(!policy||(result.generated_count==policy->generated&&result.processed_count==policy->prompt_count+policy->generated-1&&((result.stop==ps5hip_generation::StopReason::eos)==!std::strcmp(policy->stop,"eos"))),"extended stop/count policy");
   if(run)manifest<<',';manifest<<"{\"run\":"<<run<<",\"processed\":";ids(manifest,result.processed.data(),result.processed_count);manifest<<",\"generated\":";ids(manifest,result.generated.data(),result.generated_count);
   manifest<<",\"stop\":\""<<(result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit")<<"\",\"pieces_hex\":[";
   std::string text;for(size_t i=0;i<result.generated_count;++i){char piece[4096];int bytes=llama_token_to_piece(vocab,result.generated[i],piece,sizeof(piece),0,true);check(bytes>=0&&bytes<=int(sizeof(piece)),"piece");std::string value(piece,size_t(bytes));if(i)manifest<<',';manifest<<'"'<<hex(value)<<'"';text+=value;}manifest<<"],\"text_hex\":\""<<hex(text)<<"\"}";
  }
  manifest<<"]}\n";steps.close();manifest.close();check(steps.good()&&manifest.good(),"metadata write");
  }
  if(corpus){std::ofstream index(std::string(argv[2])+"/corpus.json");index<<"{\"schema\":1,\"profile\":\"corpus\",\"max_new_tokens\":16,\"cases\":[";for(unsigned i=0;i<case_count;++i){if(i)index<<',';index<<'\"'<<cases[i].id<<'\"';}index<<"]}\n";index.close();check(index.good(),"corpus index write");}
  if(extended){std::ofstream index(std::string(argv[2])+"/extended.json");index<<"{\"schema\":1,\"profile\":\"extended\",\"max_new_tokens\":64,\"cases\":[\"short64\",\"eos\",\"boundary\"],\"capture\":\"logits-all-checkpoints-v1\"}\n";index.close();check(index.good(),"extended index write");}
  if(extended)std::puts("GENERATION_REFERENCE_PASS profile=extended cases=3 runs=2 max_new_tokens=64 original_cpu_decode=true GPU_execution=false");
  else if(corpus)std::puts("GENERATION_REFERENCE_PASS profile=corpus cases=5 runs=2 max_new_tokens=16 original_cpu_decode=true GPU_execution=false");
  else std::puts("GENERATION_REFERENCE_PASS runs=2 max_new_tokens=4 original_cpu_decode=true GPU_execution=false");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());status=1;}llama_backend_free();return status;
}
