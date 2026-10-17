// Explicit CPU diagnostic: intercept the complete graph, then execute the real scheduler.
#include "llama.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include <dlfcn.h>
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
static void quoted(std::ostream& out,const char* s){out<<'"';for(const unsigned char* p=(const unsigned char*)s;*p;++p){if(*p=='"'||*p=='\\')out<<'\\'<<*p;else if(*p<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(*p)<<std::dec;else out<<*p;}out<<'"';}
struct Trace{std::string folder;int run=-1,position=-1;unsigned calls=0,captured=0;bool failed=false;};
static Trace* trace=nullptr;
static bool checkpoint(){return trace&&trace->run>=0&&(trace->run==1||trace->position==0||trace->position==1||trace->position==255||trace->position==256||trace->position==511);}
static std::string stem(){return trace->folder+"/run-"+std::to_string(trace->run)+"-pos-"+std::to_string(trace->position);}
static void capture(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 check(ggml_graph_n_nodes(graph)>0&&ggml_graph_n_nodes(graph)<4096,"graph bound");
 std::map<const ggml_tensor*,int> ids;std::vector<const ggml_tensor*> tensors;
 auto add=[&](const ggml_tensor* t){if(t&&!ids.count(t)){check(tensors.size()<8192,"tensor bound");ids[t]=int(tensors.size());tensors.push_back(t);}};
 for(int i=0;i<ggml_graph_n_nodes(graph);++i)add(ggml_graph_node(graph,i));
 for(int i=0;i<graph->n_leafs;++i)add(graph->leafs[i]);
 for(size_t i=0;i<tensors.size();++i){for(auto* s:tensors[i]->src)add(s);add(tensors[i]->view_src);}
 std::ofstream out(stem()+".graph.json");check(out.good(),"graph open");out<<"{\"run\":"<<trace->run<<",\"position\":"<<trace->position<<",\"nodes\":[";
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){if(i)out<<',';out<<ids.at(ggml_graph_node(graph,i));}out<<"],\"leaves\":[";
 for(int i=0;i<graph->n_leafs;++i){if(i)out<<',';out<<ids.at(graph->leafs[i]);}out<<"],\"tensors\":[";
 for(size_t i=0;i<tensors.size();++i){auto* t=tensors[i];if(i)out<<',';out<<"{\"id\":"<<i<<",\"name\":";quoted(out,t->name);out<<",\"type\":";quoted(out,ggml_type_name(t->type));out<<",\"op\":";quoted(out,ggml_op_name(t->op));out<<",\"description\":";quoted(out,ggml_op_desc(t));out<<",\"flags\":"<<t->flags<<",\"compute\":"<<((t->flags&GGML_TENSOR_FLAG_COMPUTE)?"true":"false")<<",\"bytes\":"<<ggml_nbytes(t)<<",\"ne\":[";for(int j=0;j<4;++j){if(j)out<<',';out<<t->ne[j];}out<<"],\"nb\":[";for(int j=0;j<4;++j){if(j)out<<',';out<<t->nb[j];}out<<"],\"sources\":[";for(int j=0;j<GGML_MAX_SRC;++j){if(j)out<<',';out<<(t->src[j]?ids.at(t->src[j]):-1);}out<<"],\"view_source\":"<<(t->view_src?ids.at(t->view_src):-1)<<",\"view_offset\":"<<t->view_offs<<",\"op_params_hex\":\"";for(unsigned char c:*(const unsigned char(*)[sizeof(t->op_params)])t->op_params)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);out<<std::dec<<"\",\"backend\":";auto* backend=ggml_backend_sched_get_tensor_backend(scheduler,const_cast<ggml_tensor*>(t));quoted(out,backend?ggml_backend_name(backend):"unassigned");out<<",\"control_file\":";
 bool control=t->op==GGML_OP_NONE&&(std::strncmp(t->name,"inp_",4)==0||std::strncmp(t->name,"attn_inp_",9)==0||std::strcmp(t->name,"out_ids")==0);
 if(control){check(t->data&&t->buffer&&ggml_nbytes(t)<=65536,"control allocation/bound");std::vector<char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());auto file="run-"+std::to_string(trace->run)+"-pos-"+std::to_string(trace->position)+"-control-"+std::to_string(i)+".bin";std::ofstream stream(trace->folder+"/"+file,std::ios::binary);stream.write(raw.data(),raw.size());stream.close();check(stream.good(),"control write");quoted(out,file.c_str());}else out<<"null";out<<'}';
 }
 out<<"]}\n";out.close();check(out.good(),"graph write");++trace->captured;
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 using Function=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static Function real=(Function)dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async");
 if(!real)return GGML_STATUS_FAILED;
 try{if(trace&&trace->run>=0){++trace->calls;if(checkpoint())capture(scheduler,graph);}}
 catch(const std::exception& e){std::fprintf(stderr,"trace: %s\n",e.what());if(trace)trace->failed=true;return GGML_STATUS_FAILED;}
 return real(scheduler,graph);
}
int main(int argc,char** argv){
 if(argc!=3)return 2;setenv("GGML_CPU_DISABLE_FUSION","1",1);llama_backend_init();int result=0;
 try{
  Trace state;state.folder=argv[2];trace=&state;auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);check(bool(model),"model");
  auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx),"context");std::ofstream steps(state.folder+"/steps.jsonl");const int tokens[]={0,1,2,31999};
  for(int run=0;run<2;++run){if(run)llama_memory_clear(llama_get_memory(ctx.get()),true);check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"reset metadata");state.run=run;
   for(int pos=0;pos<(run?2:512);++pos){state.position=pos;const unsigned before=state.calls;auto* batch=llama_batch_ext_init(ctx.get());check(batch,"batch");int index=llama_batch_ext_add_token(batch,0,tokens[pos%4]);const llama_pos position=pos;check(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true),"controls");int status=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);check(status==0&&!state.failed&&state.calls==before+1,"successful original scheduler");check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==pos,"committed position");float* logits=llama_get_logits_ith(ctx.get(),-1);check(logits,"logits");for(int i=0;i<32000;++i)check(std::isfinite(logits[i]),"finite logits");if(checkpoint()){std::ofstream out(stem()+".logits.f32le",std::ios::binary);out.write((char*)logits,128000);out.close();check(out.good(),"logits write");}steps<<"{\"run\":"<<run<<",\"position\":"<<pos<<",\"token\":"<<tokens[pos%4]<<",\"status\":0,\"scheduler_calls\":1,\"committed_position\":"<<pos<<"}\n";
   }
  }steps.close();check(steps.good()&&state.calls==514&&state.captured==7,"trace completion");trace=nullptr;std::puts("FULL_GRAPH_TRACE_PASS requests=514 graphs=7 original_cpu_scheduler=true GPU_execution=false");
 }catch(const std::exception& e){trace=nullptr;std::fprintf(stderr,"%s\n",e.what());result=1;}llama_backend_free();return result;
}
