// CPU oracle: read each boundary in its completed-node callback before activation reuse.
#include "llama.h"
#include "ggml.h"

#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool yes,const char* text){if(!yes)throw std::runtime_error(text);}
struct Trace{std::string folder;int run=-1,position=-1;unsigned captured=0;bool failed=false;std::array<bool,24> seen{};};
static bool checkpoint(const Trace& t){return t.run>=0&&(t.run==1||t.position==0||t.position==1||t.position==255||t.position==256||t.position==511);}
static std::string stem(const Trace& t){return t.folder+"/run-"+std::to_string(t.run)+"-pos-"+std::to_string(t.position);}
static bool capture(ggml_tensor* tensor,bool ask,void* pointer){
 if(ask)return true;auto& state=*static_cast<Trace*>(pointer);if(!checkpoint(state))return true;
 int index=-1;for(int layer=0;layer<22;++layer)if(std::string(tensor->name)=="l_out-"+std::to_string(layer))index=layer;
 if(!std::strcmp(tensor->name,"result_norm"))index=22;
 if(!std::strcmp(tensor->name,"result_output"))index=23;
 if(index<0)return true;
 try{
  check(!state.seen[index],"duplicate boundary callback");check(tensor->type==GGML_TYPE_F32&&ggml_is_contiguous(tensor)&&ggml_nelements(tensor)==(index==23?32000:2048),"boundary layout");
  std::vector<float> values(size_t(ggml_nelements(tensor)));ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);for(float v:values)check(std::isfinite(v),"nonfinite boundary");
  const std::string name=index<22?"layer-"+std::to_string(index):index==22?"norm":"logits";
  std::ofstream out(stem(state)+"-"+name+".f32le",std::ios::binary);out.write((const char*)values.data(),values.size()*4);out.close();check(out.good(),"boundary write");state.seen[index]=true;++state.captured;
 }catch(const std::exception& e){std::fprintf(stderr,"capture: %s\n",e.what());state.failed=true;return false;}return true;
}
int main(int argc,char** argv){
 if(argc!=3)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();int result=0;
 try{
  Trace state;state.folder=argv[2];auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);check(bool(model),"model");
  auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;cp.cb_eval=capture;cp.cb_eval_user_data=&state;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx),"context");std::ofstream steps(state.folder+"/steps.jsonl");const int tokens[]={0,1,2,31999};
  for(int run=0;run<2;++run){if(run)llama_memory_clear(llama_get_memory(ctx.get()),true);check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"reset metadata");state.run=run;
   for(int pos=0;pos<(run?2:512);++pos){state.position=pos;state.seen={};auto* batch=llama_batch_ext_init(ctx.get());check(batch,"batch");int index=llama_batch_ext_add_token(batch,0,tokens[pos%4]);const llama_pos position=pos;check(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true),"controls");int status=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);check(status==0&&!state.failed,"successful original scheduler");check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==pos,"committed position");float* logits=llama_get_logits_ith(ctx.get(),-1);check(logits,"logits");for(int i=0;i<32000;++i)check(std::isfinite(logits[i]),"finite logits");if(checkpoint(state)){for(bool seen:state.seen)check(seen,"missing boundary callback");}steps<<"{\"run\":"<<run<<",\"position\":"<<pos<<",\"token\":"<<tokens[pos%4]<<",\"status\":0,\"committed_position\":"<<pos<<"}\n";
   }
  }steps.close();check(steps.good()&&state.captured==168,"trace completion");std::puts("FULL_MODEL_REFERENCE_PASS requests=514 checkpoints=7 boundaries=24 raw_files=168 original_cpu_decode=true GPU_execution=false");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());result=1;}llama_backend_free();return result;
}
