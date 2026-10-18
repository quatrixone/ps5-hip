#include "layer_dispatch.h"
#include <cassert>
#include <cstdlib>
#include <map>
#include <cstdio>
static std::map<void*,size_t> allocations;
static unsigned lookups=0,releases=0,launches=0;static int fault=0;static bool dense_launched=false;
extern "C" hipError_t hipModuleLoad(hipModule_t* p,const char*){*p=reinterpret_cast<hipModule_t>(1);return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* p,hipModule_t,const char* name){++lookups;if(!std::strcmp(name,"layer_dense_f16")){if(fault==1)return hipErrorInvalidValue;*p=reinterpret_cast<hipFunction_t>(22);}else *p=reinterpret_cast<hipFunction_t>(1);return hipSuccess;}
extern "C" hipError_t hipModuleUnload(hipModule_t){++releases;return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){*p=std::malloc(n);assert(*p);allocations[*p]=n;return hipSuccess;}
extern "C" hipError_t hipFree(void* p){++releases;assert(allocations.erase(p));std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind){std::memcpy(d,s,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return fault==3&&dense_launched?hipErrorUnknown:hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,hipStream_t,void**,void**){++launches;if(uintptr_t(f)==22){dense_launched=true;if(fault==2)return hipErrorUnknown;}return hipSuccess;}
static bool owned(const ggml_tensor* t){return t&&t->data;}
static void fail(const char*,hipError_t){throw std::runtime_error("injected HIP failure");}
int main(){
 auto* ctx=ggml_init({100*1024*1024,nullptr,false});assert(ctx);ps5hip_layer::Selection selected{};selected.extent=256;ps5hip_layer::Weights weights{};
 for(unsigned i=0;i<15;i++){auto s=ps5hip_layer::detail::spec(i,256,true);auto* t=ggml_new_tensor(ctx,s.type,4,s.ne);ggml_set_name(t,s.name);std::memset(t->data,0,ggml_nbytes(t));if(i<10){selected.weights[i]=t;weights.uploads[i]={t,t};}else selected.requests[i-10]=t;}
 auto cache=ps5hip_layer::make_cache(ctx);auto graph=ps5hip_layer::materialize(ctx,selected,weights,cache);assert(!ps5hip_layer::select(graph.graph).dense_fixture);
 bool rejected=false;try{ps5hip_layer::select_dense_fixture(graph.graph);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 for(unsigned i=0;i<256;i++){static_cast<int64_t*>(graph.uploads[3].clone->data)[i]=int64_t(i*512);static_cast<float*>(graph.uploads[4].clone->data)[i]=i? -INFINITY:0;}
 ps5hip_layer::Dispatch dispatch(owned,fail);lookups=0;dispatch.initialize({"a","b","c","d","e","f"});assert(lookups==17&&!dispatch.dense_function);assert(dispatch.preflight(graph.graph));assert(dispatch.execute_graph(graph.graph));std::fprintf(stderr,"observed default launches=%u\n",launches);assert(launches==33);dispatch.shutdown();assert(allocations.empty());ggml_free(ctx);
 std::puts("DENSE_DEFAULT_LAYER_REGRESSION_PASS Q4=true lookups=17 launches=33 dense_rejected=true");
}
