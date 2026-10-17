// Reuse transport mocks; replace the selected-layer scheduler rejection stubs.
#define __real_ggml_backend_sched_graph_compute_async unused_layer_scheduler_rejection
#define ggml_backend_sched_graph_compute_async unused_layer_scheduler_entry
#define hipModuleLaunchKernel unused_transport_launch
#include "llm_layer_hip_mock.h"
#undef hipModuleLaunchKernel
#include "llm_layer_numeric_mock.h"
#include <fstream>
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=17&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;layer_numeric::execute(unsigned(uintptr_t(f)),gx*bx,args);return hipSuccess;}
#undef __real_ggml_backend_sched_graph_compute_async
#undef ggml_backend_sched_graph_compute_async
#include "full_qualification.h"
#include <dlfcn.h>
#include "ggml-backend-impl.h"
#include <cmath>
static unsigned real_scheduler_calls=0,device_graph_calls=0,cpu_arithmetic_nodes=0,postcompute=0;
static bool metadata(ggml_op op){return op==GGML_OP_NONE||op==GGML_OP_VIEW||op==GGML_OP_RESHAPE||op==GGML_OP_PERMUTE||op==GGML_OP_TRANSPOSE;}
static void check_backend(ggml_backend_t backend,ggml_cgraph* graph){
 const bool ours=ggml_backend_get_device(backend)==m0::device_handle();if(ours)++device_graph_calls;
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if(!(t->flags&GGML_TENSOR_FLAG_COMPUTE)||metadata(t->op))continue;
  if(!ours){++cpu_arithmetic_nodes;std::fprintf(stderr,"CPU_ARITHMETIC name=%s op=%s backend=%s\n",t->name,ggml_op_name(t->op),ggml_backend_name(backend));assert(false&&"CPU tensor arithmetic forbidden");}
 }
}
using GraphCompute=ggml_status(*)(ggml_backend_t,ggml_cgraph*);
static std::map<ggml_backend_t,GraphCompute> graph_compute_originals;
static ggml_status checked_compute(ggml_backend_t backend,ggml_cgraph* graph){check_backend(backend,graph);return graph_compute_originals.at(backend)(backend,graph);}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async"));assert(fn);++real_scheduler_calls;assert(graph_compute_originals.empty());for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto* backend=ggml_backend_sched_get_backend(scheduler,i);assert(backend&&backend->iface.graph_compute);graph_compute_originals.emplace(backend,backend->iface.graph_compute);backend->iface.graph_compute=checked_compute;}const auto result=fn(scheduler,graph);for(auto& entry:graph_compute_originals)entry.first->iface.graph_compute=entry.second;graph_compute_originals.clear();return result;
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if(!(t->flags&GGML_TENSOR_FLAG_COMPUTE)||metadata(t->op))continue;auto* b=ggml_backend_sched_get_tensor_backend(scheduler,t);assert(b&&ggml_backend_get_device(b)==m0::device_handle());}
 return __wrap_ggml_backend_sched_graph_compute_async(scheduler,graph);
}
static std::string output_dir;static unsigned sequence=0,captured=0;static int position=0;static bool seen[24]{};
static bool callback(ggml_tensor* t,bool ask,void*){
 if(ask)return true;++postcompute;
 if((t->flags&GGML_TENSOR_FLAG_COMPUTE)&&!metadata(t->op)){assert(t->buffer&&ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer))==m0::device_handle());}
 int index=-1;
 for(int i=0;i<22;++i)if(std::string(t->name)=="l_out-"+std::to_string(i))index=i;
 if(!std::strcmp(t->name,"result_norm"))index=22;if(!std::strcmp(t->name,"result_output"))index=23;
 if(index<0)return true;assert(!seen[index]&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t));
 std::vector<float> values(ggml_nelements(t));assert(values.size()==(index==23?32000u:2048u));ggml_backend_tensor_get(t,values.data(),0,values.size()*4);for(float x:values)assert(std::isfinite(x));
 std::string name=index<22?"layer-"+std::to_string(index):index==22?"norm":"logits";
 std::ofstream out(output_dir+"/run-"+std::to_string(sequence)+"-pos-"+std::to_string(position)+"-"+name+".f32le",std::ios::binary);out.write((const char*)values.data(),values.size()*4);out.close();assert(out.good());seen[index]=true;++captured;return true;
}
static void decode(llama_context* ctx,int token,int requested_position){
 position=requested_position;std::memset(seen,0,sizeof(seen));
 auto* batch=llama_batch_ext_init(ctx);assert(batch);int index=llama_batch_ext_add_token(batch,0,token);llama_pos pos=position;
 assert(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));
 const int result=llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);
 if(result){auto gate=m0::statistics();std::fprintf(stderr,"DECODE_FAILED position=%d token=%d result=%d gate=%u/%u/%u real=%u device=%u\n",position,token,result,gate.calls,gate.forwarded,gate.rejected,real_scheduler_calls,device_graph_calls);}assert(result==0);
 auto memory=llama_get_memory(ctx);assert(memory&&llama_memory_seq_pos_min(memory,0)==0&&llama_memory_seq_pos_max(memory,0)==position);
 const float* logits=llama_get_logits_ith(ctx,0);assert(logits);for(unsigned i=0;i<32000;++i)assert(std::isfinite(logits[i]));for(bool value:seen)assert(value);
}
int main(int argc,char**argv){
 assert(argc==3);const bool split=true;output_dir=argv[2];
 llama_backend_init();ggml_backend_register(m0::registration());m0::enable_full({"getrows","norm","weighted","q8","q4","layer"});
 ggml_backend_dev_t devices[]={m0::device_handle(),nullptr};llama_model_tensor_buft_override overrides[]={{"^token_embd\\.weight$",ggml_backend_dev_buffer_type(devices[0])},{nullptr,nullptr}};
 auto mp=llama_model_default_params();mp.devices=devices;mp.tensor_buft_overrides=overrides;mp.n_gpu_layers=99;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 auto* model=llama_model_load_from_file(argv[1],mp);assert(model);auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=true;cp.op_offload=true;
 if(split){cp.cb_eval=callback;cp.cb_eval_user_data=nullptr;}auto* context=llama_init_from_model(model,cp);assert(context);
 m0::reset_statistics();decode(context,0,0);decode(context,1,1);
 llama_synchronize(context);llama_memory_clear(llama_get_memory(context),true);m0::reset_full();assert(llama_memory_seq_pos_max(llama_get_memory(context),0)==-1);sequence=1;decode(context,0,0);
 assert(captured==72);const auto gate=m0::statistics();const auto live=m0::full_statistics();assert(gate.calls==3&&gate.forwarded==3&&gate.rejected==0&&real_scheduler_calls==3);assert(live.requests==3&&live.completed_nodes==3*798&&live.launches==launches&&launches>0);assert(device_graph_calls>0&&cpu_arithmetic_nodes==0);if(split)assert(postcompute>0);
 llama_free(context);llama_model_free(model);const auto retired=m0::full_statistics();assert(retired.instances_created>0&&retired.instances_created==retired.instances_freed&&retired.launches==launches);m0::disable_full();llama_backend_free();
 assert(allocations.empty()&&allocation_count==free_count&&loads==unloads);
 std::printf("PASS full-model backend mode=%s requests=3 committed_positions=0,1,reset0 nodes=%llu launches=%u instances=%u/%u scheduler_forwarded=%u device_graphs=%u CPU_arithmetic=0 postcompute=%u allocations=%u/%u; numerical HIP interpretation,72 raw boundaries, GPU unproven\n",argv[2],static_cast<unsigned long long>(retired.completed_nodes),launches,retired.instances_created,retired.instances_freed,real_scheduler_calls,device_graph_calls,postcompute,allocation_count,free_count);
}
