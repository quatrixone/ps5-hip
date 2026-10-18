#pragma once
// Host-only interpretation of production GPU kernels; never linked into native code.
// Reuse transport mocks; replace the selected-layer scheduler rejection stubs.
#define __real_ggml_backend_sched_graph_compute_async unused_layer_scheduler_rejection
#define ggml_backend_sched_graph_compute_async unused_layer_scheduler_entry
#define hipModuleLaunchKernel unused_transport_launch
#include "llm_layer_hip_mock.h"
#undef hipModuleLaunchKernel
#include "llm_layer_numeric_mock.h"
#include <fstream>
#ifdef PS5HIP_SAME_INPUT_ORACLE
#include "llm_full_projection_oracle.h"
#endif
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=21&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;layer_numeric::execute(unsigned(uintptr_t(f)),gx*bx,args);
#ifdef PS5HIP_SAME_INPUT_ORACLE
same_input(unsigned(uintptr_t(f)),args);
#endif
return hipSuccess;}
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
static ggml_status checked_compute(ggml_backend_t backend,ggml_cgraph* graph){check_backend(backend,graph);return graph_compute_originals.at(backend)(backend,graph);}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async"));assert(fn);++real_scheduler_calls;assert(graph_compute_originals.empty());for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto* backend=ggml_backend_sched_get_backend(scheduler,i);assert(backend&&backend->iface.graph_compute);graph_compute_originals.emplace(backend,backend->iface.graph_compute);backend->iface.graph_compute=checked_compute;}const auto result=fn(scheduler,graph);for(auto& entry:graph_compute_originals)entry.first->iface.graph_compute=entry.second;graph_compute_originals.clear();return result;
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t scheduler,ggml_cgraph* graph){
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if(!(t->flags&GGML_TENSOR_FLAG_COMPUTE)||metadata(t->op))continue;auto* b=ggml_backend_sched_get_tensor_backend(scheduler,t);assert(b&&ggml_backend_get_device(b)==m0::device_handle());}
 return __wrap_ggml_backend_sched_graph_compute_async(scheduler,graph);
}
