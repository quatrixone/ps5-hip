#pragma once
// Shared public-HIP mock: no GPU tensor arithmetic.
#include "prefix_dispatch.h"
#include "ggml-impl.h"
#include <cassert>
#include <map>
#include <string>
#include <vector>
using namespace ps5hip_prefix;
static std::map<void*,size_t> allocations;
static size_t uploaded_bytes=0;
static std::vector<char> scratch(2304);
static std::vector<unsigned> launches;
static unsigned loads=0,unloads=0,frees=0,syncs=0;
static std::string failure;
static Dispatch * active=nullptr;
static hipError_t status(const char * site) {
    if(active){uint64_t pending=0;for(const auto& row:active->calls)pending+=row.attempts-row.completions;assert(pending==1);}
    if(failure=="nonreturn" && std::strcmp(site,"launch")==0)throw std::runtime_error("nonreturn mock");
    return failure==site?hipErrorInvalidValue:hipSuccess;
}
extern "C" hipError_t hipModuleLoad(hipModule_t * m,const char*) { auto e=status("load"); if(e==hipSuccess)*m=reinterpret_cast<hipModule_t>(++loads);return e; }
extern "C" hipError_t hipModuleGetFunction(hipFunction_t * f,hipModule_t,const char * name) { auto e=status("function");if(e==hipSuccess){for(unsigned i=0;i<5;++i)if(std::strcmp(name,Dispatch::symbols[i])==0)*f=reinterpret_cast<hipFunction_t>(i+1);}return e; }
extern "C" hipError_t hipModuleUnload(hipModule_t) { auto e=status("unload"); if(e==hipSuccess)++unloads;return e; }
extern "C" hipError_t hipMalloc(void ** p,size_t n) {
 if(n==2304){*p=scratch.data();return hipSuccess;}
 *p=std::aligned_alloc(256,(n+255)&~size_t(255));assert(*p);std::memset(*p,0,n);allocations[*p]=n;return hipSuccess;
}
extern "C" hipError_t hipFree(void * p) {
 ++frees;if(p==scratch.data())return hipSuccess;
 assert(allocations.erase(p)==1);std::free(p);return hipSuccess;
}
extern "C" hipError_t hipMemcpy(void * dst,const void * src,size_t n,hipMemcpyKind kind) {
 if(kind==hipMemcpyHostToDevice)uploaded_bytes+=n;
 std::memmove(dst,src,n);return hipSuccess;
}
extern "C" hipError_t hipInit(unsigned){return hipSuccess;}
extern "C" hipError_t hipGetDeviceCount(int * n){*n=1;return hipSuccess;}
extern "C" hipError_t hipMemset(void * dst,int value,size_t n){std::memset(dst,value,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize() {++syncs;return status("sync");}
template<class T> T arg(void ** a,int i) {T value;std::memcpy(&value,a[i],sizeof(T));return value;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned lds,hipStream_t stream,void ** args,void ** extra) {
    auto e=status("launch");if(e!=hipSuccess)return e;
    unsigned index=reinterpret_cast<uintptr_t>(f)-1;assert(index<5 && gz==1 && bz==1 && !stream && !extra);launches.push_back(index);
    if(index==0) {assert(gx==1 && gy==4 && bx==256 && by==1 && lds==0);assert(arg<int64_t>(args,3)==2048 && arg<int64_t>(args,4)==1);assert(arg<size_t>(args,9)==1152 && arg<size_t>(args,10)==36864000 && arg<size_t>(args,14)==1);auto fd=arg<Dispatch::Uint3>(args,5);assert(fd.x==1 && fd.y==0 && fd.z==1);}
    if(index==1 || index==2) {assert(gx==1 && gy==1 && bx==1024 && by==1 && lds==128);assert(arg<int>(args,2)==2048 && arg<float>(args,8)==1e-5f);assert((arg<void*>(args,9)!=nullptr)==(index==2));assert(arg<void*>(args,17)==nullptr && arg<float>(args,25)==1.f);}
    if(index==3) {assert(gx==8 && gy==1 && bx==256 && by==1 && lds==0);assert(arg<void*>(args,1)==scratch.data() && arg<int64_t>(args,6)==2048);}
    if(index==4) {assert((gx==2048 || gx==256) && gy==1 && bx==32 && by==4 && lds==0);assert(arg<void*>(args,1)==scratch.data() && !arg<void*>(args,2));assert(arg<uint32_t>(args,5)==2048 && arg<uint32_t>(args,9)==gx && arg<uint32_t>(args,11)==gx*64 && arg<uint32_t>(args,18)==0);auto fusion=arg<Dispatch::Fusion>(args,3);for(auto b:fusion.bytes)assert(b==0);}
    return hipSuccess;
}
