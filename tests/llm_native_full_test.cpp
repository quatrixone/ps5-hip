// Transport/lifecycle fixture: simulated backend outputs are zero, never arithmetic evidence.
// Reuse transport mocks; replace the selected-layer scheduler rejection stubs.
#define __real_ggml_backend_sched_graph_compute_async unused_layer_scheduler_rejection
#define ggml_backend_sched_graph_compute_async unused_layer_scheduler_entry
#include "llm_layer_hip_mock.h"
#undef __real_ggml_backend_sched_graph_compute_async
#undef ggml_backend_sched_graph_compute_async
#include "full_qualification.h"
#include <dlfcn.h>
#include "ggml-backend-impl.h"
#include <cmath>
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
static ggml_status checked_compute(ggml_backend_t backend,ggml_cgraph* graph){check_backend(backend,graph);const auto result=graph_compute_originals.at(backend)(backend,graph);if(result==GGML_STATUS_SUCCESS)for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if((t->flags&GGML_TENSOR_FLAG_COMPUTE)&&!metadata(t->op))std::memset(t->data,0,ggml_nbytes(t));}return result;}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async"));assert(fn);++real_scheduler_calls;assert(graph_compute_originals.empty());for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto* backend=ggml_backend_sched_get_backend(scheduler,i);assert(backend&&backend->iface.graph_compute);graph_compute_originals.emplace(backend,backend->iface.graph_compute);backend->iface.graph_compute=checked_compute;}const auto result=fn(scheduler,graph);for(auto& entry:graph_compute_originals)entry.first->iface.graph_compute=entry.second;graph_compute_originals.clear();return result;
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){return __wrap_ggml_backend_sched_graph_compute_async(scheduler,graph);}
#include "native_full.h"
#include <set>
static unsigned files=0;static size_t bytes=0;static std::set<std::string> names;static std::vector<std::string> snapshots;
static bool synced(){return true;}
static void snapshot(const char* phase){snapshots.emplace_back(phase);}
static void write_raw(const char* name,const void* data,size_t size,void*){
 assert(std::strncmp(name,"run-",4)==0&&(size==8192||size==128000)&&names.insert(name).second);
 const auto* p=static_cast<const float*>(data);for(size_t i=0;i<size/4;++i)assert(p[i]==0.f);++files;bytes+=size;
}
int main(int argc,char**argv){assert(argc==2);const auto result=m0::run_native_full(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_raw,nullptr,synced,snapshot);
 assert(result.full.requests==514&&result.full.launches==376762&&files==168&&bytes==2214912&&snapshots==std::vector<std::string>({"before_load","after_load","after_context","after_cleanup"}));
 std::set<std::string> expected;for(unsigned run=0;run<2;++run)for(unsigned pos:(run?std::vector<unsigned>{0,1}:std::vector<unsigned>{0,1,255,256,511}))for(unsigned i=0;i<24;++i){std::string stage=i<22?"layer-"+std::to_string(i):i==22?"norm":"logits";expected.insert("run-"+std::to_string(run)+"-pos-"+std::to_string(pos)+"-"+stage+".f32le");}assert(names==expected);
 assert(allocations.empty()&&allocation_count==free_count&&loads==unloads&&cpu_arithmetic_nodes==0);
 std::puts("PASS native full helper mocked:514 real decoder commits/reset, all22layers/head, 376762 launches,168 files; GPU/numerics unproven");
}
