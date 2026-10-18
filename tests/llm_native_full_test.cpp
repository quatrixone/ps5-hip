// Transport/lifecycle fixture: controlled zero/EOS outputs are never arithmetic evidence.
// Reuse transport mocks; replace the selected-layer scheduler rejection stubs.
#define __real_ggml_backend_sched_graph_compute_async unused_layer_scheduler_rejection
#define ggml_backend_sched_graph_compute_async unused_layer_scheduler_entry
#define hipModuleLaunchKernel unused_transport_launch
#include "llm_layer_hip_mock.h"
#undef hipModuleLaunchKernel
#undef __real_ggml_backend_sched_graph_compute_async
#undef ggml_backend_sched_graph_compute_async
#include "full_qualification.h"
#include <dlfcn.h>
#include "ggml-backend-impl.h"
#include <cmath>
static bool force_eos=false;
static std::vector<int32_t> processed_tokens;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){
 assert(uintptr_t(f)>=1&&uintptr_t(f)<=21&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);
 if(uintptr_t(f)==1){const int32_t* token=nullptr;std::memcpy(&token,args[1],sizeof(token));assert(token);processed_tokens.push_back(*token);}++launches;return hipSuccess;
}
static unsigned real_scheduler_calls=0,device_graph_calls=0,cpu_arithmetic_nodes=0;
static bool metadata(ggml_op op){return op==GGML_OP_NONE||op==GGML_OP_VIEW||op==GGML_OP_RESHAPE||op==GGML_OP_PERMUTE||op==GGML_OP_TRANSPOSE;}
static void check_backend(ggml_backend_t backend,ggml_cgraph* graph){
 const bool ours=ggml_backend_get_device(backend)==m0::device_handle();if(ours)++device_graph_calls;
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if(!(t->flags&GGML_TENSOR_FLAG_COMPUTE)||metadata(t->op))continue;
  if(!ours){++cpu_arithmetic_nodes;std::fprintf(stderr,"CPU_ARITHMETIC name=%s op=%s backend=%s\n",t->name,ggml_op_name(t->op),ggml_backend_name(backend));assert(false&&"CPU tensor arithmetic forbidden");}
 }
}
using GraphCompute=ggml_status(*)(ggml_backend_t,ggml_cgraph*);
static std::map<ggml_backend_t,GraphCompute> graph_compute_originals;
static ggml_status checked_compute(ggml_backend_t backend,ggml_cgraph* graph){check_backend(backend,graph);const auto result=graph_compute_originals.at(backend)(backend,graph);if(result==GGML_STATUS_SUCCESS)for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if((t->flags&GGML_TENSOR_FLAG_COMPUTE)&&!metadata(t->op))std::memset(t->data,0,ggml_nbytes(t));if(force_eos&&!std::strcmp(t->name,"result_output"))static_cast<float*>(t->data)[2]=1.f;}return result;}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async"));assert(fn);++real_scheduler_calls;assert(graph_compute_originals.empty());for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto* backend=ggml_backend_sched_get_backend(scheduler,i);assert(backend&&backend->iface.graph_compute);graph_compute_originals.emplace(backend,backend->iface.graph_compute);backend->iface.graph_compute=checked_compute;}const auto result=fn(scheduler,graph);for(auto& entry:graph_compute_originals)entry.first->iface.graph_compute=entry.second;graph_compute_originals.clear();return result;
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){return __wrap_ggml_backend_sched_graph_compute_async(scheduler,graph);}
#include "native_full.h"
#include <set>
static std::map<std::string,std::string> metadata_files;
static unsigned files=0;static size_t bytes=0;static std::set<std::string> names;static std::vector<std::string> snapshots;
static bool synced(){return true;}
static void snapshot(const char* phase){snapshots.emplace_back(phase);}
static void write_raw(const char* name,const void* data,size_t size,void*){
 if(!std::strcmp(name,"generation.json")||!std::strcmp(name,"steps.jsonl")){assert(metadata_files.emplace(name,std::string(static_cast<const char*>(data),size)).second);++files;bytes+=size;return;}
 assert(std::strncmp(name,"run-",4)==0&&(size==8192||size==128000)&&names.insert(name).second);
 const auto* p=static_cast<const float*>(data);for(size_t i=0;i<size/4;++i)assert(p[i]==(force_eos&&std::strstr(name,"-logits.f32le")&&i==2?1.f:0.f));++files;bytes+=size;
}
int main(int argc,char**argv){
 if(argc==3&&!std::strcmp(argv[2],"clock-regress")){m0::NativeFullClock clock;clock.previous=UINT64_MAX;clock.now();return 99;}
 assert(argc==2||(argc==3&&(!std::strcmp(argv[2],"smoke")||!std::strcmp(argv[2],"generation")||!std::strcmp(argv[2],"generation-eos"))));
 const bool smoke=argc==3&&!std::strcmp(argv[2],"smoke"),generation=argc==3&&(!std::strcmp(argv[2],"generation")||!std::strcmp(argv[2],"generation-eos"));
 force_eos=generation&&!std::strcmp(argv[2],"generation-eos");
 m0::NativeFullClock clock;const auto first_tick=clock.now();assert(first_tick>0&&clock.now()>=first_tick);
 const auto profile=generation?m0::NativeFullProfile::generation:smoke?m0::NativeFullProfile::smoke:m0::NativeFullProfile::sequence;
 const auto result=m0::run_native_full(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_raw,nullptr,synced,snapshot,profile);
 const unsigned requests=generation?(force_eos?10u:16u):smoke?3u:514u,expected_files=generation?(force_eos?242u:386u):smoke?72u:168u;
 assert(result.full.requests==requests&&result.full.launches==requests*733&&files==expected_files&&result.files==files&&result.bytes==bytes&&snapshots==std::vector<std::string>({"before_load","after_load","after_context","after_cleanup"}));
 std::set<std::string> expected;
 for(unsigned run=0;run<2;++run)for(unsigned pos:(generation?(force_eos?std::vector<unsigned>{0,1,2,3,4}:std::vector<unsigned>{0,1,2,3,4,5,6,7}):smoke?(run?std::vector<unsigned>{0}:std::vector<unsigned>{0,1}):(run?std::vector<unsigned>{0,1}:std::vector<unsigned>{0,1,255,256,511})))for(unsigned i=0;i<24;++i){std::string stage=i<22?"layer-"+std::to_string(i):i==22?"norm":"logits";expected.insert("run-"+std::to_string(run)+"-pos-"+std::to_string(pos)+"-"+stage+".f32le");}assert(names==expected);
 if(generation){
  assert(metadata_files.size()==2&&metadata_files.at("generation.json").find(force_eos?"\"generated\":[2]":"\"generated\":[0,0,0,0]")!=std::string::npos);
  const std::vector<int32_t> one=force_eos?std::vector<int32_t>{1,9038,2501,263,931}:std::vector<int32_t>{1,9038,2501,263,931,0,0,0};auto both=one;both.insert(both.end(),one.begin(),one.end());assert(processed_tokens==both);
  assert(bytes==size_t(requests)*316416+metadata_files.at("generation.json").size()+metadata_files.at("steps.jsonl").size());
 }else assert(metadata_files.empty()&&bytes==(smoke?949248u:2214912u));
 assert(allocations.empty()&&allocation_count==free_count&&loads==unloads&&cpu_arithmetic_nodes==0);
 std::printf("PASS native full helper mocked: profile=%s requests=%llu launches=%llu files=%u; GPU/numerics unproven\n",generation?"generation":smoke?"smoke":"sequence",static_cast<unsigned long long>(result.full.requests),static_cast<unsigned long long>(result.full.launches),files);
}
