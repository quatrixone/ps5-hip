#include "llm_prefix_hip_mock.h"
#include "prefix_qualification.h"
#undef DEPRECATED
#include "llama.h"
#include "ggml-backend.h"
extern "C" [[noreturn]] void ps5hip_llm_fail(const char * reason){std::fprintf(stderr,"TEST_TERMINAL %s\n",reason);std::abort();}
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*){assert(false && "full graph must not forward");return GGML_STATUS_FAILED;}
// Host shared-library interception only. Native production retains mandatory --wrap.
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s,ggml_cgraph * g){return __wrap_ggml_backend_sched_graph_compute_async(s,g);}
struct Readbacks {int token=1;unsigned count=0;ggml_backend_t backend=nullptr;};
static void readback(ggml_tensor* const * outputs,size_t count,int32_t token,void * data){
 auto& state=*static_cast<Readbacks*>(data);assert(count==6&&token==state.token);
 for(size_t i=0;i<count;++i){
  auto* t=outputs[i];assert(ggml_backend_buffer_get_type(t->buffer)==ggml_backend_get_default_buffer_type(state.backend));
  assert(t->type==GGML_TYPE_F32&&ggml_nelements(t)==(i<4?2048:256));
  std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
  // Mock launches do not calculate: zero outputs demonstrate no hidden CPU tensor fallback.
  for(auto value:values)assert(value==0);
 }
 ++state.count;
}
static bool completed(ggml_tensor*,bool ask,void * counter){if(!ask)++*static_cast<unsigned*>(counter);return true;}
int main(int argc,char ** argv){
 assert(argc==2);llama_backend_init();ggml_backend_register(m0::registration());
 auto backend=ggml_backend_dev_init(m0::device_handle(),nullptr);assert(backend);
 m0::configure_prefix(backend,{"getrows","norm","weighted","q8","q4"});
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;
 mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={m0::device_handle(),nullptr};mp.devices=devices;
 auto * model=llama_model_load_from_file(argv[1],mp);assert(model);
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;
 cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 cp.offload_kqv=false;cp.op_offload=false;unsigned postcompute=0;cp.cb_eval=completed;cp.cb_eval_user_data=&postcompute;
 auto * context=llama_init_from_model(model,cp);assert(context);
 Readbacks observed;observed.backend=backend;const ggml_cgraph* first=nullptr;
 for(int token:{0,1,2,31999}){
  observed.token=token;
  auto * batch=llama_batch_ext_init(context);assert(batch);const int index=llama_batch_ext_add_token(batch,0,token);const llama_pos pos=0;
  assert(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));
  m0::reset_statistics();const auto uploaded_before=uploaded_bytes;const auto launched_before=launches.size();
  {m0::ActualPrefixQualification scope(backend,readback,&observed);
   assert(llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch)==-3);
  }
  llama_batch_ext_free(batch);const auto gate=m0::statistics();
  assert(gate.calls==1&&gate.rejected==1&&gate.forwarded==0&&gate.direct_graph_calls==1&&postcompute==0);
  assert(uploaded_bytes-uploaded_before==39821316&&launches.size()-launched_before==9);
  if(token==0)first=gate.last_graph;else assert(gate.last_graph==first);
  std::printf("INTEGRATED_PREFIX_ROUTE token=%d uploaded_leaves=39821316 launches=9 original_graph_rejected=true CPU_postcompute=0 mock_GPU=true\n",token);
 }
 const auto stats=m0::prefix_statistics(backend);assert(stats.invocations==4&&stats.uploaded_leaf_bytes==159285264&&stats.completed_launches==36&&stats.last_token==31999&&observed.count==4);
 llama_free(context);llama_model_free(model);const auto calls=m0::disable_prefix(backend);ggml_backend_free(backend);llama_backend_free();
 const unsigned expected[]={5,5,1,8,36,36,1,1,5};
 for(size_t i=0;i<calls.size();++i){const auto& row=calls[i];assert(row.attempts==expected[i]&&row.completions==expected[i]&&row.successes==expected[i]&&!row.failures&&!row.first_error&&!row.last_result);}
 assert(calls[2].requested_bytes==2304&&calls[3].requested_bytes==32&&calls[7].requested_bytes==2304);
 assert(allocations.empty()&&unloads==5&&frees==5);
 std::puts("PASS actual model -> mandatory wrapper -> selected prefix -> genuine ggml backend graph_compute -> mocked HIP; GPU/numerics unproven");
}
