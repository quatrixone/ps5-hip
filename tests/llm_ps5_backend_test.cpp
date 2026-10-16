#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include "ggml-cpu.h"
#include "ps5_backend.h"
#include "model_gate.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
static ps5hip::clr::Observation observed;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t*p,const pthread_attr_t*a,void*(*f)(void*),void*d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void**v){return __real_pthread_join(p,v);}

struct BackendFailure{};
extern "C" [[noreturn]] void ps5hip_llm_fail(const char*){throw BackendFailure{};}
static void storage_and_gate_check() {
 ggml_backend_register(m0::registration());auto ps5=ggml_backend_dev_init(ggml_backend_reg_dev_get(m0::registration(),0),nullptr);m0::require(ps5);
 auto type=ggml_backend_get_default_buffer_type(ps5);m0::require(ggml_backend_buft_get_max_size(type)==m0::allocation_limit);
 const unsigned before=observed.allocations;m0::require(!type->iface.alloc_buffer(type,0));m0::require(!type->iface.alloc_buffer(type,m0::allocation_limit+1));m0::require(observed.allocations==before);
 size_t free=1,total=1;ggml_backend_dev_memory(m0::device_handle(),&free,&total);m0::require(free==0&&total==0);
 ggml_init_params params{1<<20,nullptr,true};auto ctx=ggml_init(params);m0::require(ctx);
 auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);
 auto storage=ggml_backend_alloc_ctx_tensors(ctx,ps5);m0::require(storage);
 float input[16],output[16];for(unsigned i=0;i<16;++i)input[i]=float(i+1);
 ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_copy(a,b);ggml_backend_tensor_get(b,output,0,sizeof(output));m0::require(!std::memcmp(input,output,sizeof(input)));
 unsigned invalid=0;
 try {storage->iface.set_tensor(storage,a,input,sizeof(input),1);}catch(const BackendFailure&){++invalid;}
 try {storage->iface.get_tensor(storage,a,output,size_t(-1),1);}catch(const BackendFailure&){++invalid;}
 auto saved=a->data;a->data=static_cast<char*>(ggml_backend_buffer_get_base(storage))+ggml_backend_buffer_get_size(storage);
 try {storage->iface.get_tensor(storage,a,output,0,1);}catch(const BackendFailure&){++invalid;}a->data=saved;
 m0::require(invalid==3);ggml_backend_tensor_get(b,output,0,sizeof(output));m0::require(!std::memcmp(input,output,sizeof(input)));
 const auto before_fill=m0::hip_call_statistics();
 storage->iface.memset_tensor(storage,b,0x5a,4,8);
 ggml_backend_tensor_get(b,output,0,sizeof(output));
 const auto* actual=reinterpret_cast<const unsigned char*>(output);const auto* expected=reinterpret_cast<const unsigned char*>(input);
 for(size_t i=0;i<sizeof(output);++i)m0::require(actual[i]==(i>=4&&i<12?0x5a:expected[i]));
 const auto fill=m0::hip_call_statistics().calls[static_cast<unsigned>(m0::HipCall::fill)];
 const auto prior=before_fill.calls[static_cast<unsigned>(m0::HipCall::fill)];
 m0::require(fill.attempts==prior.attempts+1&&fill.completions==prior.completions+1&&fill.successes==prior.successes+1&&fill.failures==prior.failures&&fill.requested_bytes==prior.requested_bytes+8);
 ggml_backend_buffer_clear(storage,0);ggml_backend_tensor_get(a,output,0,sizeof(output));for(auto f:output)m0::require(f==0);ggml_backend_synchronize(ps5);
 ggml_backend_buffer_free(storage);ggml_free(ctx);
 ctx=ggml_init(params);a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto sum=ggml_add(ctx,a,b);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,sum);
 auto cpu=ggml_backend_cpu_init();m0::require(cpu);ggml_backend_t backends[]={ps5,cpu};auto scheduler=ggml_backend_sched_new(backends,nullptr,2,128,false,false);m0::require(scheduler);
 m0::require(ggml_backend_sched_alloc_graph(scheduler,graph));ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_set(b,input,0,sizeof(input));float sentinel[16];for(auto& f:sentinel)f=-999;ggml_backend_tensor_set(sum,sentinel,0,sizeof(sentinel));
 m0::require(ggml_backend_sched_get_tensor_backend(scheduler,sum)==cpu);
 m0::require(m0::release_compute(scheduler,ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().forwarded==0);
 ggml_backend_tensor_get(sum,output,0,sizeof(output));m0::require(!std::memcmp(output,sentinel,sizeof(output)));
 m0::require(ggml_backend_graph_compute(ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().direct_graph_calls==1);
 ggml_backend_sched_free(scheduler);ggml_backend_free(cpu);ggml_free(ctx);ggml_backend_free(ps5);
 m0::require(observed.allocations==observed.frees && !observed.terminal && !observed.uncertain && observed.launches==0);
 std::puts("M0_APP STORAGE copied=16 cleared=16 synthetic_add_rejected=true cpu_compute_calls=0 hip_launches=0 client_graph_executed=false");
}

static void log_policy_controls(){
 const char* first="graph_compute: ggml_backend_sched_graph_compute_async failed with error -1\n";
 const char* second="process_ubatch: failed to compute graph, compute status: -1\n";
 m0::GateStatistics rejected{1,0,1,0,nullptr};
 {m0::ModelGateLog p;p.accept(GGML_LOG_LEVEL_ERROR,"loader: failed to read tensor\n");m0::require(p.unexpected.load());}
 {m0::ModelGateLog p;p.accept(GGML_LOG_LEVEL_ERROR,first,rejected);m0::require(p.unexpected.load());}
 {m0::ModelGateLog p;p.begin();p.accept(GGML_LOG_LEVEL_ERROR,first,rejected);p.accept(GGML_LOG_LEVEL_ERROR,second,rejected);p.finish();}
 {m0::ModelGateLog p;p.begin();p.accept(GGML_LOG_LEVEL_ERROR,second,rejected);m0::require(p.unexpected.load());}
 {m0::ModelGateLog p;p.begin();p.accept(GGML_LOG_LEVEL_ERROR,first,{});m0::require(p.unexpected.load());}
 {m0::ModelGateLog p;p.begin();p.accept(GGML_LOG_LEVEL_ERROR,first,rejected);p.accept(GGML_LOG_LEVEL_ERROR,second,rejected);p.accept(GGML_LOG_LEVEL_ERROR,first,rejected);m0::require(p.unexpected.load());}
 {m0::ModelGateLog p;p.begin();bool failed=false;try{p.finish();}catch(const BackendFailure&){failed=true;}m0::require(failed);}
 std::puts("LOG_POLICY_PASS controls=7 unexpected_loader_error_rejected=true");
}
int main(int argc,char**argv){
 m0::require(argc==2);log_policy_controls();storage_and_gate_check();
 const auto result=m0::reject_actual_model(argv[1]);
 m0::require(result.attempts==2&&result.graph_reused&&result.gate.calls==1&&result.gate.rejected==1&&result.gate.forwarded==0&&result.post_compute==0);
 m0::require(observed.allocations==observed.frees&&!observed.terminal&&!observed.uncertain&&observed.launches==0&&submits==0);
 std::puts("PRODUCTION_GATE_HOST_PASS actual_llama_process=true cpu_bypass=false storage_bounds=3 allocation_limit=67108864 model_compute=0 gpu_submits=0");
}
