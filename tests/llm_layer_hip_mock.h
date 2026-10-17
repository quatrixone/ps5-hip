#pragma once
// Mock transport only: launches deliberately perform no tensor arithmetic.
#include "layer_qualification.h"
#include "layer_dispatch.h"
#include "ggml-impl.h"
#undef DEPRECATED
#include "llama.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
static std::map<void*,size_t> allocations;
static unsigned loads=0,unloads=0,launches=0,clears=0,allocation_count=0,free_count=0;
static size_t uploaded=0;
extern "C" hipError_t hipInit(unsigned){return hipSuccess;}
extern "C" hipError_t hipGetDeviceCount(int* n){*n=1;return hipSuccess;}
extern "C" hipError_t hipModuleLoad(hipModule_t* m,const char*){*m=reinterpret_cast<hipModule_t>(++loads);return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t,const char* name){
 for(unsigned i=0;i<5;++i)if(std::strcmp(name,ps5hip_prefix::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+1);return hipSuccess;}
 for(unsigned i=0;i<13;++i)if(std::strcmp(name,ps5hip_layer::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+6);return hipSuccess;}
 return hipErrorInvalidValue;
}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloads;return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){++allocation_count;*p=std::aligned_alloc(256,(n+255)&~size_t(255));assert(*p);std::memset(*p,0,n);assert(allocations.emplace(*p,n).second);return hipSuccess;}
extern "C" hipError_t hipFree(void* p){++free_count;assert(allocations.erase(p)==1);std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind kind){if(kind==hipMemcpyHostToDevice)uploaded+=n;std::memmove(d,s,n);return hipSuccess;}
extern "C" hipError_t hipMemset(void* p,int value,size_t n){++clears;std::memset(p,value,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void**,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=18&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;return hipSuccess;}
extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason){std::fprintf(stderr,"TEST_TERMINAL %s\n",reason);std::abort();}
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*){assert(false&&"CPU graph forwarding forbidden");return GGML_STATUS_FAILED;}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s,ggml_cgraph* g){return __wrap_ggml_backend_sched_graph_compute_async(s,g);}
