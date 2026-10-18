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
 const std::array<const char*,6> paths={"a","b","c","d","e","f"};
 auto* ctx=ggml_init({12*1024*1024,nullptr,false});assert(ctx);
 auto* w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,2048,2048);ggml_set_name(w,"blk.0.attn_q.weight");auto* x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);auto* t=ggml_mul_mat(ctx,w,x);
 for(int injected:{1,2,3}){
  ps5hip_layer::Dispatch dispatch(owned,fail);fault=injected;dense_launched=false;bool rejected=false;
  try{dispatch.initialize_dense_fixture(paths);assert(injected!=1);dispatch.execute_node(t,true);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&dispatch.base.terminal);const auto before=releases;
  try{dispatch.shutdown();assert(false);}catch(const std::runtime_error&){}assert(releases==before);
  // Host mock has no live GPU. Release retained mock storage after quarantine assertions.
  for(const auto& p:allocations)std::free(p.first);allocations.clear();
 }
 fault=0;dense_launched=false;ps5hip_layer::Dispatch dispatch(owned,fail);lookups=0;dispatch.initialize_dense_fixture(paths);assert(lookups==22&&dispatch.dense_function);dispatch.shutdown();assert(!dispatch.dense_function&&!dispatch.base.ready);
 lookups=0;dispatch.initialize(paths,true);assert(lookups==21&&!dispatch.dense_function);dispatch.shutdown();assert(allocations.empty());ggml_free(ctx);
 std::puts("DENSE_LAYER_FAILURE_PASS missing_symbol,launch,sync_quarantine,reinitialize_default");
}
