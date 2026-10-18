#define PS5HIP_ENABLE_DENSE_F16 1
#include "llm_layer_numeric_mock.h"
#include "layer_dispatch.h"
#include "native_dense_layer.h"
#include <cassert>
#include <map>
#include <fstream>
#include <string>
#include <cstdlib>
static std::map<void*,size_t> allocations;
static unsigned frees=0;static int fault=0;static bool dense_started=false;
static unsigned loads=0,unloads=0,launches=0,clears=0,dense_launches=0,attention_launches=0;
static size_t uploaded=0;
extern "C" hipError_t hipInit(unsigned){return hipSuccess;}
extern "C" hipError_t hipGetDeviceCount(int* n){*n=1;return hipSuccess;}
extern "C" hipError_t hipModuleLoad(hipModule_t* m,const char*){*m=reinterpret_cast<hipModule_t>(++loads);return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t,const char* name){
 for(unsigned i=0;i<5;++i)if(std::strcmp(name,ps5hip_prefix::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+1);return hipSuccess;}
 for(unsigned i=0;i<16;++i)if(std::strcmp(name,ps5hip_layer::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+6);return hipSuccess;}
 if(!std::strcmp(name,"layer_dense_f16")){if(fault==1)return hipErrorInvalidValue;*f=reinterpret_cast<hipFunction_t>(22);return hipSuccess;}
 return hipErrorInvalidValue;
}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloads;return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){*p=std::aligned_alloc(256,(n+255)&~size_t(255));assert(*p);std::memset(*p,0,n);assert(allocations.emplace(*p,n).second);return hipSuccess;}
extern "C" hipError_t hipFree(void* p){++frees;assert(allocations.erase(p)==1);std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind kind){if(fault==5&&kind==hipMemcpyDeviceToHost)return hipErrorUnknown;if(kind==hipMemcpyHostToDevice)uploaded+=n;std::memmove(d,s,n);return hipSuccess;}
extern "C" hipError_t hipMemset(void* p,int value,size_t n){++clears;std::memset(p,value,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return fault==3&&dense_started?hipErrorUnknown:hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=22&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;if(uintptr_t(f)==11||uintptr_t(f)==13)++attention_launches;if(uintptr_t(f)==22){dense_started=true;if(fault==2)return hipErrorUnknown;++dense_launches;for(layer_numeric::lane=0;layer_numeric::lane<gx*bx;layer_numeric::lane++)layer_numeric::layer_dense_f16(layer_numeric::arg<const _Float16*>(args,0),layer_numeric::arg<const _Float16*>(args,1),layer_numeric::arg<float*>(args,2),layer_numeric::arg<unsigned>(args,3),layer_numeric::arg<unsigned>(args,4));}else layer_numeric::execute(unsigned(uintptr_t(f)),gx*bx,args);return hipSuccess;}

extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason){throw std::runtime_error(reason);}
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*){throw std::runtime_error("CPU scheduler forbidden");}
static std::ifstream reference;static uint64_t offset=0;static unsigned records=0;
static bool write(uint64_t at,const void* data,size_t bytes,void*){
 if(fault==4)return false;assert(at==offset);std::vector<unsigned char> expected(bytes);reference.read(reinterpret_cast<char*>(expected.data()),std::streamsize(bytes));assert(reference.good());
 if(std::memcmp(data,expected.data(),bytes)){std::fprintf(stderr,"packed mismatch record=%u offset=%llu size=%zu\n",records,(unsigned long long)offset,bytes);throw std::runtime_error("packed CPU mismatch");}
 offset+=bytes;++records;return true;
}
static bool sync_log(){return true;}
int main(int argc,char**argv){
 assert(argc==3);std::string mode=argv[1];if(mode=="missing-symbol")fault=1;else if(mode=="launch")fault=2;else if(mode=="sync")fault=3;else if(mode=="writer")fault=4;else if(mode=="readback")fault=5;else assert(mode=="numeric");
 if(!fault){reference.open(argv[2],std::ios::binary);assert(reference.good());}
 try{
  auto result=m0::run_native_dense_layer({"getrows","norm","weighted","q8","q4","layer"},write,nullptr,sync_log);
  assert(!fault&&result.requests==526&&result.records==11084&&result.bytes==154392576&&offset==result.bytes&&records==result.records);
  char extra;assert(!reference.read(&extra,1));assert(allocations.empty()&&loads==6&&unloads==6&&frees==16&&dense_launches==3682&&attention_launches==1052);
  assert(result.calls[1].successes==22&&result.calls[4].successes==17358&&result.calls[6].successes==1);
  assert(result.storage.calls[unsigned(m0::HipCall::allocate)].successes==14&&result.storage.calls[unsigned(m0::HipCall::free)].successes==14);
  std::printf("NATIVE_DENSE_HOST_PASS requests=%u records=%u bytes=%llu frees=%u CPU_equal=true GPU_execution=false\n",result.requests,result.records,(unsigned long long)result.bytes,frees);
 }catch(const std::runtime_error& error){
  if(!fault){std::fprintf(stderr,"unexpected: %s\n",error.what());return 1;}
  assert(frees==0);std::printf("NATIVE_DENSE_QUARANTINE_PASS mode=%s frees=0\n",mode.c_str());
  // Transport mock only: no GPU work exists; release leaked mock buffers for ASan.
  for(const auto& p:allocations)std::free(p.first);allocations.clear();
 }
}
