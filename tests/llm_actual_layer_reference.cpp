#include "model_layer.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
using ps5hip_layer::check;
struct Capture {
 bool done=true;
 std::vector<ggml_tensor*> nodes;
 std::map<ggml_tensor*,std::vector<char>> outputs;
 static bool callback(ggml_tensor* t,bool ask,void* user){
  if(ask)return true;auto& state=*static_cast<Capture*>(user);if(state.done)return true;
  state.nodes.push_back(t);
  if(t->type==GGML_TYPE_F32&&!t->view_src&&std::strcmp(t->name,"node_1")&&std::strcmp(t->name,"node_2")){
   auto& raw=state.outputs[t];raw.resize(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
  }
  if(!std::strcmp(t->name,"l_out-0"))state.done=true;
  return true;
 }
 void reset(){done=false;nodes.clear();outputs.clear();}
};
struct Oracle {
 ggml_context* resident=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t storage=nullptr;
 ps5hip_layer::Weights weights{};ps5hip_layer::Cache cache{};
 ~Oracle(){if(storage)ggml_backend_buffer_free(storage);if(backend)ggml_backend_free(backend);if(resident)ggml_free(resident);}
 void initialize(const ps5hip_layer::Selection& selected){
  check(!resident,"oracle initialized twice");resident=ggml_init({2*1024*1024,nullptr,true});check(resident,"resident context");
  weights=ps5hip_layer::make_weights(resident,selected);cache=ps5hip_layer::make_cache(resident);
  backend=ggml_backend_cpu_init();check(backend,"CPU oracle backend");ggml_backend_cpu_set_n_threads(backend,4);
  storage=ggml_backend_alloc_ctx_tensors(resident,backend);check(storage,"resident allocation");
  for(auto pair:weights.uploads){std::vector<char> data(ggml_nbytes(pair.original));ggml_backend_tensor_get(pair.original,data.data(),0,data.size());ggml_backend_tensor_set(pair.clone,data.data(),0,data.size());}
  reset();
 }
 void reset(){std::vector<char> zero(262144);ggml_backend_tensor_set(cache.k,zero.data(),0,zero.size());ggml_backend_tensor_set(cache.v,zero.data(),0,zero.size());}
};
static void write(const std::string& name,ggml_tensor* tensor){
 std::vector<char> raw(ggml_nbytes(tensor));ggml_backend_tensor_get(tensor,raw.data(),0,raw.size());
 if(tensor->type==GGML_TYPE_F32){for(size_t i=0;i<raw.size();i+=4){float value;std::memcpy(&value,raw.data()+i,4);check(std::isfinite(value),"nonfinite reference");}}
 std::ifstream prior(name);check(!prior.good(),"reference file exists");std::ofstream file(name,std::ios::binary);file.write(raw.data(),raw.size());file.close();check(file.good(),"reference write");
}
static void execute(Oracle& oracle,const ps5hip_layer::Selection& selected,int token,int position,int run,const std::string& output,const Capture* original){
 std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({2*1024*1024,nullptr,true}),ggml_free);check(bool(context),"request context");
 auto graph=ps5hip_layer::materialize(context.get(),selected,oracle.weights,oracle.cache,position<256?256:512);
 auto* buffer=ggml_backend_alloc_ctx_tensors(context.get(),oracle.backend);check(buffer,"request allocation");
 const int32_t tok=token,pos=position;const int64_t slot=position;std::array<int64_t,256> indices{};for(int i=0;i<256;++i)indices[i]=int64_t(i)*512+position;
 std::vector<float> mask(size_t(graph.extent),-std::numeric_limits<float>::infinity());std::fill(mask.begin(),mask.begin()+position+1,0.f);
 const void* data[]={&tok,&pos,&slot,indices.data(),mask.data()};const size_t sizes[]={4,4,8,2048,mask.size()*4};
 for(size_t i=0;i<5;++i){check(ggml_nbytes(graph.uploads[i].clone)==sizes[i],"control size");ggml_backend_tensor_set(graph.uploads[i].clone,data[i],0,sizes[i]);}
 check(ggml_backend_graph_compute(oracle.backend,graph.graph)==GGML_STATUS_SUCCESS,"CPU layer compute");
 if(original){
  for(size_t i=0;i<graph.nodes.size();++i){auto* t=graph.nodes[i];if(t->view_src||t->type!=GGML_TYPE_F32)continue;
   std::vector<char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());check(raw==original->outputs.at(selected.nodes[i]),"derived/original CPU intermediate differs");
  }
 }
 if(run>=0&&(run==1||position==0||position==1||position==255||position==256||position==511)){
  for(size_t i=0;i<graph.outputs.size();++i){char name[128];std::snprintf(name,sizeof(name),"/run-%d-pos-%d-out-%02zu.f32le",run,position,i);write(output+name,graph.outputs[i]);}
 }
 ggml_backend_buffer_free(buffer);
}
static void run(const char* model_path,const std::string& output){
 Capture capture;auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path,mp),llama_model_free);check(bool(model),"model load");
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;cp.cb_eval=Capture::callback;cp.cb_eval_user_data=&capture;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);check(bool(context),"model context");
 Oracle oracle;ps5hip_layer::Selection selected{};const int tokens[]={0,1,2,31999};
 for(int position=0;position<4;++position){
  capture.reset();auto* batch=llama_batch_ext_init(context.get());check(batch,"batch");int index=llama_batch_ext_add_token(batch,0,tokens[position]);const llama_pos pos=position;
  check(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true),"batch setup");
  const int result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);check(result==0&&capture.done&&capture.nodes.size()==40,"original CPU layer capture");
  auto* meta=ggml_init({1024*1024,nullptr,true});check(meta,"capture graph context");auto* graph=ggml_new_graph_custom(meta,64,false);for(auto* t:capture.nodes)ggml_graph_add_node(graph,t);selected=ps5hip_layer::select(graph);
  if(position==0)oracle.initialize(selected);execute(oracle,selected,tokens[position],position,-1,output,&capture);ggml_free(meta);
 }
 for(int sequence=0;sequence<2;++sequence){
  oracle.reset();const int count=sequence==0?512:2;
  for(int position=0;position<count;++position)execute(oracle,selected,tokens[position%4],position,sequence,output,nullptr);
  write(output+"/run-"+std::to_string(sequence)+"-cache-k.f16le",oracle.cache.k);write(output+"/run-"+std::to_string(sequence)+"-cache-v.f16le",oracle.cache.v);
 }
 std::puts("ACTUAL_LAYER_REFERENCE_PASS requests=514 original_cpu_validation=4 nodes=37 outputs=21 raw_files=151 cache_reset=true GPU_execution=false");
}
int main(int argc,char** argv){
 if(argc!=3)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();int result=0;
 try{run(argv[1],argv[2]);}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());result=1;}
 llama_backend_free();return result;
}
