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
static bool force_eos=false,vary_logits=false;
static int32_t next_mock_token(int32_t token){return 100+(token+1)%100;}
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
static ggml_status checked_compute(ggml_backend_t backend,ggml_cgraph* graph){check_backend(backend,graph);const auto result=graph_compute_originals.at(backend)(backend,graph);if(result==GGML_STATUS_SUCCESS)for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if((t->flags&GGML_TENSOR_FLAG_COMPUTE)&&!metadata(t->op))std::memset(t->data,0,ggml_nbytes(t));if((force_eos||vary_logits)&&!std::strcmp(t->name,"result_output"))static_cast<float*>(t->data)[force_eos?2:next_mock_token(processed_tokens.back())]=1.f;}return result;}
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
 const char* base=std::strchr(name,'/');base=base?base+1:name;
 if(!std::strcmp(base,"generation.json")||!std::strcmp(base,"steps.jsonl")||!std::strcmp(base,"corpus.json")){assert(metadata_files.emplace(name,std::string(static_cast<const char*>(data),size)).second);++files;bytes+=size;return;}
 assert(std::strncmp(base,"run-",4)==0&&(size==8192||size==128000)&&names.insert(name).second);
 const auto* p=static_cast<const float*>(data);for(size_t i=0;i<size/4;++i)assert(p[i]==((force_eos||vary_logits)&&std::strstr(name,"-logits.f32le")&&i==size_t(force_eos?2:next_mock_token(processed_tokens.back()))?1.f:0.f));++files;bytes+=size;
}
int main(int argc,char**argv){
 if(argc==3&&!std::strcmp(argv[2],"clock-regress")){m0::NativeFullClock clock;clock.previous=UINT64_MAX;clock.now();return 99;}
 assert(argc==2||(argc==3&&(!std::strcmp(argv[2],"smoke")||!std::strcmp(argv[2],"generation")||!std::strcmp(argv[2],"generation-eos")||!std::strcmp(argv[2],"corpus")||!std::strcmp(argv[2],"corpus-eos"))));
 const bool corpus=argc==3&&(!std::strcmp(argv[2],"corpus")||!std::strcmp(argv[2],"corpus-eos"));
 const bool smoke=argc==3&&!std::strcmp(argv[2],"smoke"),generation=argc==3&&(!std::strcmp(argv[2],"generation")||!std::strcmp(argv[2],"generation-eos"));
 vary_logits=corpus;
 force_eos=(generation&&!std::strcmp(argv[2],"generation-eos"))||(corpus&&!std::strcmp(argv[2],"corpus-eos"));
 m0::NativeFullClock clock;const auto first_tick=clock.now();assert(first_tick>0&&clock.now()>=first_tick);
 const auto profile=corpus?m0::NativeFullProfile::corpus:generation?m0::NativeFullProfile::generation:smoke?m0::NativeFullProfile::smoke:m0::NativeFullProfile::sequence;
 const auto result=m0::run_native_full(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_raw,nullptr,synced,snapshot,profile);
 if(corpus){
  assert(!force_eos&&metadata_files.size()==11);
  std::set<std::string> expected;std::vector<int32_t> expected_tokens;unsigned requests=0;size_t metadata_bytes=0;
  for(const auto& entry:metadata_files)metadata_bytes+=entry.second.size();
  for(const auto& selected:m0::native_generation_cases){
   const std::string prefix=std::string(selected.id)+"/";const auto& meta=metadata_files.at(prefix+"generation.json");
   assert(meta.find("\"max_new_tokens\":16")!=std::string::npos);
   const std::string marker="\"prompt_tokens\":[";auto offset=meta.find(marker);assert(offset!=std::string::npos);offset+=marker.size();std::vector<int32_t> prompt;
   while(meta.at(offset)!=']'){size_t used=0;prompt.push_back(std::stoi(meta.substr(offset),&used));offset+=used;if(meta.at(offset)==',')++offset;}
   assert(!prompt.empty()&&prompt.size()<=32);const unsigned count=unsigned(prompt.size())+15;
   std::vector<int32_t> generated;int32_t token=prompt.back();std::string ids="\"generated\":[";
   for(unsigned i=0;i<16;++i){token=next_mock_token(token);generated.push_back(token);if(i)ids+=',';ids+=std::to_string(token);}ids+=']';assert(meta.find(ids)!=std::string::npos);
   for(unsigned run=0;run<2;++run){
    expected_tokens.insert(expected_tokens.end(),prompt.begin(),prompt.end());expected_tokens.insert(expected_tokens.end(),generated.begin(),generated.end()-1);requests+=count;
    for(unsigned pos=0;pos<count;++pos)for(unsigned i=0;i<24;++i){const std::string stage=i<22?"layer-"+std::to_string(i):i==22?"norm":"logits";expected.insert(prefix+"run-"+std::to_string(run)+"-pos-"+std::to_string(pos)+"-"+stage+".f32le");}
   }
   const auto& steps=metadata_files.at(prefix+"steps.jsonl");assert(steps.find("{\"run\":1,\"position\":0,")!=std::string::npos);
  }
  assert(metadata_files.at("corpus.json")=="{\"schema\":1,\"profile\":\"corpus\",\"max_new_tokens\":16,\"cases\":[\"story\",\"science\",\"arithmetic\",\"structured\",\"unicode\"]}\n");
  assert(names==expected&&processed_tokens==expected_tokens&&files==requests*24+11&&bytes==size_t(requests)*316416+metadata_bytes);
  assert(result.full.requests==requests&&result.full.launches==requests*733&&result.files==files&&result.bytes==bytes);
  assert(snapshots==std::vector<std::string>({"before_load","after_load","after_context","after_cleanup"})&&result.full.instances_created==1&&result.full.instances_freed==1);
  assert(allocations.empty()&&allocation_count==free_count&&loads==unloads&&cpu_arithmetic_nodes==0);
  std::printf("PASS native corpus helper mocked: cases=5 resets=10 requests=%u files=%u; GPU/numerics unproven\n",requests,files);return 0;
 }
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
