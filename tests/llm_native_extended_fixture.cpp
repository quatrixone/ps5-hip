#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>
#include "generation.h"
#include "extended_generation.h"
struct ggml_tensor {const char* name;int type;size_t bytes;};
constexpr int GGML_TYPE_F32=0,LLAMA_PROCESS_TYPE_DECODE=0;
using llama_pos=int;
struct llama_model {};
struct llama_context {};
struct llama_vocab {};
struct Batch {int token=0,pos=-1;};
static llama_vocab vocab;
static int reset_count=0,position=-1,submissions=0,tokenized=5;
static bool is_extended=false,bad_stop=false,bad_reset=false;
static float logits[32000];
static bool (*callback)(ggml_tensor*,bool,void*);
static void* callback_state;
static std::vector<int> submitted;
static int next_token(int token){return 100+(token+1)%100;}
static bool ggml_is_contiguous(ggml_tensor*){return true;}
static size_t ggml_nbytes(ggml_tensor* t){return t->bytes;}
static void ggml_backend_tensor_get(ggml_tensor* t,void* out,size_t,size_t n){if(n==128000)std::memcpy(out,logits,n);else std::memset(out,0,n);assert(t);}
static const llama_vocab* llama_model_get_vocab(llama_model*){return &vocab;}
static int llama_vocab_n_tokens(const llama_vocab*){return 32000;}
static bool outer=false;
static int llama_tokenize(const llama_vocab*,const char* prompt,int,int32_t* out,int capacity,bool,bool){if(outer){tokenized=5;if(is_extended)for(const auto& c:ps5hip_generation::extended_cases)if(!std::strcmp(c.prompt,prompt))tokenized=int(c.prompt_count);}assert(capacity>=tokenized);for(int i=0;i<tokenized;++i)out[i]=i+10;return tokenized;}
static bool llama_vocab_is_eog(const llama_vocab*,int token){return token==2;}
static void llama_synchronize(llama_context*){}
static llama_context* llama_get_memory(llama_context* c){return c;}
static void llama_memory_clear(llama_context*,bool){position=-1;++reset_count;}
static int llama_memory_seq_pos_max(llama_context*,int){return position;}
static int llama_memory_seq_pos_min(llama_context*,int){return position<0?-1:0;}
static Batch* llama_batch_ext_init(llama_context*){return new Batch;}
static int llama_batch_ext_add_token(Batch* b,int,int token){b->token=token;return 0;}
static bool llama_batch_ext_set_pos(Batch* b,int,const int* p){b->pos=*p;return true;}
static bool llama_batch_ext_set_output_logits(Batch*,int,bool){return true;}
static void llama_batch_ext_free(Batch* b){delete b;}
static const float* llama_get_logits_ith(llama_context*,int){return logits;}
static int llama_token_to_piece(const llama_vocab*,int,char* out,int,int,bool){*out='x';return 1;}
static int llama_process(llama_context*,int,Batch* b){
 assert(b->pos==position+1);position=b->pos;++submissions;submitted.push_back(b->token);
 std::fill(std::begin(logits),std::end(logits),0.f);
 const bool eos=is_extended&&(reset_count==3||reset_count==4);
 logits[(eos&&!bad_stop)?2:next_token(b->token)+(bad_reset&&reset_count==2?1:0)]=1.f;
 for(int i=0;i<24;++i){std::string name=i<22?"l_out-"+std::to_string(i):i==22?"result_norm":"result_output";ggml_tensor t{name.c_str(),GGML_TYPE_F32,i==23?128000u:8192u};if(callback(&t,true,callback_state))callback(&t,false,callback_state);}
 return 0;
}
namespace m0 {
using PrefixWrite=void(*)(const char*,const void*,size_t,void*);
using PrefixLogSync=bool(*)();
struct FullStatistics {unsigned requests=0;uint64_t completed_nodes=0,launches=0;unsigned instances_created=0,instances_freed=0;};
static int live_models=0,live_contexts=0,loads=0;
static bool enabled=false,initialized=false;
struct Gate {unsigned calls,forwarded,rejected;};
static Gate statistics(){return {unsigned(submissions),unsigned(submissions),0};}
static void reset_statistics(){submissions=0;}
static FullStatistics full_statistics(){assert(!live_models&&!live_contexts);return {unsigned(submissions),uint64_t(submissions)*798,uint64_t(submissions)*733,1,1};}
static void* registration(){return nullptr;}
static void* device_handle(){return reinterpret_cast<void*>(1);}
static void enable_full(const std::array<const char*,6>&){assert(!enabled);enabled=true;}
static void disable_full(){assert(enabled&&!live_contexts&&!live_models);enabled=false;}
struct ModelGateLog {std::atomic<bool> unexpected{false};static void callback(int,const char*,void*){}};
static void require(bool ok){if(!ok)throw std::runtime_error("contract");}
static void reset_full(){}
}

using ggml_backend_dev_t=void*;
static void llama_log_set(void(*)(int,const char*,void*),void*){}
static void llama_backend_init(){assert(!m0::initialized);m0::initialized=true;}
static void llama_backend_free(){assert(m0::initialized&&!m0::enabled);m0::initialized=false;}
static void ggml_backend_register(void*){}
static void* ggml_backend_dev_buffer_type(void* d){return d;}
struct llama_model_tensor_buft_override {const char* pattern;void* buffer;};
struct ModelParams {int n_gpu_layers=0,load_mode=0,lazy_mode=0;bool use_extra_bufts=true,check_tensors=false;ggml_backend_dev_t* devices=nullptr;llama_model_tensor_buft_override* tensor_buft_overrides=nullptr;};
constexpr int LLAMA_LOAD_MODE_NONE=0,LLAMA_LAZY_MODE_OFF=0,GGML_TYPE_F16=1,LLAMA_FLASH_ATTN_TYPE_DISABLED=0;
static ModelParams llama_model_default_params(){return {};}
static llama_model* llama_model_load_from_file(const char*,ModelParams p){assert(p.n_gpu_layers==99&&!p.use_extra_bufts&&p.check_tensors&&p.tensor_buft_overrides);++m0::loads;++m0::live_models;return new llama_model;}
static void llama_model_free(llama_model* m){assert(!m0::live_contexts);--m0::live_models;delete m;}
struct ContextParams {int n_ctx=0,n_batch=0,n_ubatch=0,n_seq_max=0,n_threads=0,n_threads_batch=0,type_k=0,type_v=0,flash_attn_type=0;bool offload_kqv=false,op_offload=false;bool(*cb_eval)(ggml_tensor*,bool,void*)=nullptr;void* cb_eval_user_data=nullptr;};
static ContextParams llama_context_default_params(){return {};}
static llama_context* llama_init_from_model(llama_model*,ContextParams p){assert(p.n_ctx==512&&p.n_batch==1&&p.n_ubatch==1&&p.n_seq_max==1&&p.offload_kqv&&p.op_offload);callback=p.cb_eval;callback_state=p.cb_eval_user_data;++m0::live_contexts;return new llama_context;}
static void llama_free(llama_context* c){--m0::live_contexts;delete c;}
// PRODUCTION_HELPER
static std::set<std::string> names;
static size_t float_bytes=0;
static std::map<std::string,std::string> metadata;
static void writer(const char* name,const void* bytes,size_t size,void*){
 if(std::strstr(name,".f32le")){assert(names.insert(name).second);float_bytes+=size;}
 else assert(metadata.emplace(name,std::string(static_cast<const char*>(bytes),size)).second);
}
static bool sync_log(){return true;}
int main(int argc,char** argv){
 assert(argc==2);const std::string mode=argv[1];outer=mode=="outer-extended"||mode=="outer-short"||mode=="outer-corpus";is_extended=mode!="short"&&mode!="corpus"&&mode!="outer-short"&&mode!="outer-corpus";
 bad_stop=mode=="bad-stop";bad_reset=mode=="bad-reset";
 llama_model model;llama_context context;m0::ModelGateLog logs;m0::NativeFullClock clock;m0::NativeFullSink sink{writer,nullptr,sync_log};callback=m0::NativeFullSink::callback;callback_state=&sink;
 static_assert(unsigned(m0::NativeFullProfile::extended)==4);
 if(outer){
  const auto profile=mode=="outer-extended"?m0::NativeFullProfile::extended:mode=="outer-corpus"?m0::NativeFullProfile::corpus:m0::NativeFullProfile::generation;
  const auto result=m0::run_native_full("mock",{"a","b","c","d","e","f"},writer,nullptr,sync_log,nullptr,profile);
  assert(m0::loads==1&&!m0::live_models&&!m0::live_contexts&&!m0::initialized&&!m0::enabled);
  if(is_extended){assert(result.full.requests==1176&&result.files==2103&&reset_count==6&&names.size()==2096&&float_bytes==158064640&&metadata.size()==7);
   assert(metadata.at("extended.json")=="{\"schema\":1,\"profile\":\"extended\",\"max_new_tokens\":64,\"cases\":[\"short64\",\"eos\",\"boundary\"],\"capture\":\"logits-all-checkpoints-v1\"}\n");}
  else assert(result.files==result.full.requests*24+metadata.size()&&reset_count==(mode=="outer-corpus"?10:2));
  std::printf("PASS %s\n",mode.c_str());return 0;
 }
 unsigned requests=0;std::set<std::string> expected;std::vector<int> expected_tokens;
 try{
  const unsigned cases=is_extended?3:mode=="corpus"?5:1;
  for(unsigned c=0;c<cases;++c){
   const auto* ext=is_extended?&ps5hip_generation::extended_cases[c]:nullptr;
   const auto* corpus=mode=="corpus"?&m0::native_generation_cases[c]:nullptr;
   tokenized=ext?int(ext->prompt_count):5;if(mode=="bad-prompt")++tokenized;if(mode=="overflow")tokenized=449;
   const auto result=m0::run_full_generation(&model,&context,sink,clock,logs,corpus,ext);requests+=result.requests;
   const int generated=ext?int(ext->generated):corpus?16:4,commits=tokenized+generated-1;
   const std::string prefix=ext?std::string(ext->id)+"/":corpus?std::string(corpus->id)+"/":"";
   for(unsigned run=0;run<2;++run){
    for(int i=0;i<tokenized;++i)expected_tokens.push_back(i+10);
    int token=tokenized+9;for(int i=1;i<generated;++i){token=next_token(token);expected_tokens.push_back(token);}
    for(int pos=0;pos<commits;++pos)for(unsigned stage=0;stage<24;++stage){
     if(ext&&stage<23&&!ps5hip_generation::extended_checkpoint(*ext,unsigned(pos)))continue;
     const std::string label=stage<22?"layer-"+std::to_string(stage):stage==22?"norm":"logits";
     expected.insert(prefix+"run-"+std::to_string(run)+"-pos-"+std::to_string(pos)+"-"+label+".f32le");
    }
   }
   const auto& evidence=metadata.at(prefix+"generation.json");assert(evidence.find("\"stop\":\""+std::string(ext?ext->stop:"step_limit")+"\"")!=std::string::npos);
  }
  assert(mode=="extended"||mode=="short"||mode=="corpus");assert(names==expected&&submitted==expected_tokens&&submissions==int(requests));
  assert(reset_count==(is_extended?6:mode=="corpus"?10:2));
  if(is_extended){assert(requests==1176&&names.size()==2096&&float_bytes==158064640&&metadata.size()==6);}
  else assert(names.size()==requests*24&&float_bytes==size_t(requests)*316416);
 }catch(const std::runtime_error&){assert(mode=="bad-prompt"||mode=="bad-stop"||mode=="bad-reset"||mode=="overflow");if(mode=="bad-prompt"||mode=="overflow")assert(submissions==0&&reset_count==0);}
 std::printf("PASS %s\n",mode.c_str());
}
