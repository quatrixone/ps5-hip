// Explicit host transport model: real runtime/ownership, no AMD ISA execution.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/hip_runtime_control.h"
#include <ps5platform/kernel.h>
#include <sys/mman.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
int hipify_client_main(); // Actual unedited translated int main(), renamed by flag.
extern "C" const unsigned char __ps5hip_names_begin[],__ps5hip_names_end[];
extern "C" void __hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
namespace {
struct Allocation {void *cpu=nullptr;size_t bytes=0,alignment=0;};
std::map<int64_t,Allocation> allocations;
int64_t next_id=1;uint64_t next_address=0x200040000;
unsigned allocation_calls=0,backends=0,submits=0,releases=0;
std::string mode;
std::array<uint64_t,3> current_data{};
std::vector<unsigned> previous;
void check(bool yes){if(!yes)throw std::runtime_error("HIPIFY transport invariant");}
Allocation &containing(uint64_t address,size_t bytes){
 for(auto &entry:allocations){auto &a=entry.second;auto base=reinterpret_cast<uint64_t>(a.cpu);
  if(base&&address>=base&&address-base<=a.bytes&&bytes<=a.bytes-(address-base))return a;}
 throw std::runtime_error("HIPIFY address outside live backing");
}
void dispose(Allocation &a){if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;}}
uint64_t read(const uint8_t*p,unsigned bytes){uint64_t value=0;for(unsigned i=0;i<bytes;++i)value|=uint64_t(p[i])<<(8*i);return value;}
void put(uint8_t*p,uint64_t value,unsigned bytes){for(unsigned i=0;i<bytes;++i)p[i]=uint8_t(value>>(8*i));}
bool terminal_mode(){return mode=="submit"||mode=="kick"||mode=="timeout"||mode=="throw"||mode=="unknown"||mode=="postflush"||mode=="cleanup";}
struct Audit {~Audit(){
 const auto observation=ps5hip_runtime_observation();
 std::printf("HIPIFY_MODEL_FINAL live=%zu backends=%u submits=%u releases=%u completed=%u modules=%zu functions=%zu finalized=%u model_only=true\n",allocations.size(),backends,submits,releases,observation.completed,observation.modules,observation.functions,unsigned(ps5hip_runtime_finalized()));
 // Terminal cases exit before destructors; only this explicit host model may
 // dispose its private mappings after recording nonterminal lifetime evidence.
 for(auto &entry:allocations)dispose(entry.second);
}} audit;
class Backend final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{
  if(mode=="preflush"&&!submits)throw std::runtime_error("preflush");
  if(mode=="postflush"&&submits)throw std::runtime_error("postflush");
 }
 int32_t submit(uint32_t*words,uint32_t count)override{
  ++submits;check(submits<=6);std::map<uint32_t,std::vector<uint32_t>> registers;std::array<uint32_t,3>groups{};bool dispatch=false;
  for(size_t at=0;at<count;){
   const auto header=words[at];check((header>>30)==3);const size_t length=((header>>16)&0x3fff)+1;check(length<=count-at-1);
   if((header&0xffff)==0x7600){check(length>=2&&!registers.count(words[at+1]));registers.emplace(words[at+1],std::vector<uint32_t>(words+at+2,words+at+length+1));}
   if(header==0xc0031502){check(!dispatch&&words[at+4]==0x8005);dispatch=true;groups={words[at+1],words[at+2],words[at+3]};}
   at+=length+1;
  }
  const unsigned sizes[]{1,73,257},n=sizes[(submits-1)/2],round=(submits-1)%2,guard=7,poison=0xdeadbeef;
  check(dispatch&&groups==std::array<uint32_t,3>{(n+63)/64,1,1}&&registers.at(0x207)==std::vector<uint32_t>{64,1,1});
  const auto&s=registers.at(0x240),&p=registers.at(0x20c),&resource=registers.at(0x212);
  check(s.size()==6&&p.size()==2&&resource==std::vector<uint32_t>{0x60af0040,0x8c});
  for(unsigned i=0;i<4;++i)check(s[i]==0);
  containing((uint64_t(p[0])<<8)|(uint64_t(p[1])<<40),4);
  const uint64_t address=uint64_t(s[4])|(uint64_t(s[5])<<32);containing(address,288);const auto*k=reinterpret_cast<const uint8_t*>(address);
  const uint64_t ap=read(k,8),bp=read(k+8,8),cp=read(k+16,8);check(read(k+24,4)==n);
  std::array<uint64_t,3>bases{ap-guard*4,bp-guard*4,cp-guard*4};check(bases[0]!=bases[1]&&bases[0]!=bases[2]&&bases[1]!=bases[2]);
  for(auto base:bases)check(reinterpret_cast<uint64_t>(containing(base,(n+2*guard)*4).cpu)==base);
  if(!round)current_data=bases;else check(current_data==bases);
  uint8_t expected[288]{};put(expected,ap,8);put(expected+8,bp,8);put(expected+16,cp,8);put(expected+24,n,4);
  put(expected+32,(n+63)/64,4);put(expected+36,1,4);put(expected+40,1,4);put(expected+44,64,2);put(expected+46,1,2);put(expected+48,1,2);put(expected+96,1,2);check(std::memcmp(expected,k,288)==0);
  const auto*a=reinterpret_cast<const unsigned*>(ap),*b=reinterpret_cast<const unsigned*>(bp);auto*out=reinterpret_cast<unsigned*>(cp);
  for(unsigned i=0;i<n;++i){check(a[i]==UINT32_MAX-i*17u-round*101u);check(b[i]==i*31u+round*701u+9u);check(out[i]==poison);}
  for(auto pointer:{a,b,static_cast<const unsigned*>(out)})for(unsigned i=0;i<guard;++i)check(pointer[int(i)-int(guard)]==poison&&pointer[n+i]==poison);
  const auto prior_releases=releases;unsigned dummy=0;
  check(hipDeviceSynchronize()==hipErrorNotReady&&hipFree(reinterpret_cast<void*>(bases[2]))==hipErrorNotReady&&hipMemcpy(&dummy,out,4,hipMemcpyDeviceToHost)==hipErrorNotReady&&releases==prior_releases);
  if(mode=="submit")return -7;if(mode=="throw")throw std::runtime_error("submit");if(mode=="unknown")throw 42;
  std::vector<unsigned> values(n);for(unsigned i=0;i<n;++i)values[i]=unsigned((uint64_t(a[i])+b[i])&0xffffffffu);
  if(mode=="stale-second"&&submits==2){check(previous.size()==n);std::memcpy(out,previous.data(),n*4);}
  else if(mode!="missing")std::memcpy(out,values.data(),n*4);
  previous=values;
  if(mode=="corrupt")out[0]^=1;if(mode=="input")const_cast<unsigned*>(a)[0]^=1;if(mode=="guard")out[-1]=0;
  const auto*end=words+count-8;check(end[-2]==0xc0004600&&end[-1]==0x407&&end[0]==0xc0064900&&end[1]==0x30c514);
  const uint64_t marker=uint64_t(end[3])|(uint64_t(end[4])<<32);containing(marker,4);if(mode!="timeout")*reinterpret_cast<volatile uint32_t*>(marker)=1;return 0;
 }
 int32_t kick()override{return mode=="kick"?-8:0;}
 uint64_t now_ns()override{return clock;}
 void sleep_us(unsigned us)override{clock+=uint64_t(us)*1000;}
};
void duplicate_binding(){
 // A static object ensures the name is not a stack pointer rejected earlier.
 static const char name[]="translated_add";
 const auto begin=reinterpret_cast<uintptr_t>(__ps5hip_names_begin),end=reinterpret_cast<uintptr_t>(__ps5hip_names_end),at=reinterpret_cast<uintptr_t>(name);
 check(at>=begin&&at<=end&&sizeof(name)<=end-at);const auto token=ps5hip_runtime_token_for(name);check(token);
 __hipRegisterFunction(ps5hip_runtime_handle(),token,const_cast<char*>(name),name,UINT32_MAX,nullptr,nullptr,nullptr,nullptr,nullptr);
 check(hipGetLastError()==hipErrorInvalidDeviceFunction);
 void*p=nullptr;check(hipMalloc(&p,0)==hipErrorInvalidDeviceFunction&&p==nullptr&&hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
 check(hipLaunchKernel(token,dim3(1),dim3(64),nullptr,0,nullptr)==hipErrorInvalidDeviceFunction);
 (void)hipGetLastError();
}
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;if(mode=="init")throw std::runtime_error("init");return std::make_unique<Backend>();}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t*physical){
 check(type==12);++allocation_calls;if(mode=="data-allocate"&&allocation_calls==2)return -1;*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void**pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
 check(protection==0x33&&flags==0);auto&a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);check(next_address>=0x200000000&&next_address<0x300000000&&bytes<=0x300000000-next_address);
 int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
 const char*options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
 void*cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);if(cpu==MAP_FAILED)return -1;
 a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;
}
extern "C" int32_t sceKernelMunmap(void*pointer,size_t bytes){for(auto&entry:allocations)if(entry.second.cpu==pointer){check(entry.second.bytes==bytes);if(mode=="cleanup"&&entry.first==6)return -1;dispose(entry.second);return 0;}return -1;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;}
int main(int argc,char**argv){
 try{
  check(argc==2);mode=argv[1];std::printf("HIPIFY_MODEL_BEGIN mode=%s model_only=true\n",mode.c_str());check(allocations.empty()&&!backends&&!submits&&ps5hip_runtime_registration_ready());
  check(hipLaunchKernel(reinterpret_cast<const void*>(1),dim3(1),dim3(64),nullptr,0,nullptr)==hipErrorInvalidValue);(void)hipGetLastError();check(!backends&&!allocation_calls);
  if(mode=="partial")duplicate_binding();
  const int result=hipify_client_main();
  if(terminal_mode()){
   check(result==1&&ps5hip_runtime_terminal()&&submits==1);const bool unresolved=mode!="cleanup";const auto live=allocations.size();const auto freed=releases;const auto prior_backends=backends,prior_allocations=allocation_calls;check(live==(unresolved?7u:4u)&&ps5hip_runtime_gpu_unresolved()==unresolved);
   (void)hipGetLastError();void*pointer=nullptr;unsigned dummy=0;const auto token=ps5hip_runtime_token();
   check(hipMalloc(&pointer,0)==hipErrorUnknown&&pointer==nullptr&&hipFree(nullptr)==hipErrorUnknown&&hipFree(reinterpret_cast<void*>(current_data[0]))==hipErrorUnknown);
   check(hipMemcpy(&dummy,reinterpret_cast<void*>(current_data[0]),4,hipMemcpyDeviceToHost)==hipErrorUnknown&&hipDeviceSynchronize()==hipErrorUnknown&&hipLaunchKernel(token,dim3(1),dim3(64),nullptr,0,nullptr)==hipErrorUnknown);
   check(allocations.size()==live&&releases==freed&&submits==1&&backends==prior_backends&&allocation_calls==prior_allocations);std::printf("HIPIFY_QUARANTINE live=%zu releases=%u unresolved=%u submits=%u no_retry=true\n",live,releases,unsigned(unresolved),submits);std::fflush(nullptr);std::_Exit(30);
  }
  check(allocations.empty()&&!ps5hip_runtime_terminal());
  if(mode=="late-partial"){check(result==0&&submits==6);duplicate_binding();}
  if(mode!="no-finish")ps5hip_runtime_finish();
  const bool normal=mode=="normal";if(normal){check(result==0&&submits==6&&releases==33&&ps5hip_runtime_ready());
   const auto prior_releases=releases;unsigned word=0;check(hipFree(reinterpret_cast<void*>(current_data[0]))==hipErrorInvalidDevicePointer&&hipMemcpy(&word,reinterpret_cast<void*>(current_data[0]),4,hipMemcpyDeviceToHost)==hipErrorInvalidDevicePointer&&releases==prior_releases);(void)hipGetLastError();
   std::puts("HIPIFY_RUNTIME_PASS launches=6 allocations=closed stale_rejected=true");return 0;
  }
  if(mode=="no-finish")check(result==0&&!ps5hip_runtime_ready());
  else if(mode!="late-partial")check(result==1);
  std::printf("HIPIFY_EXPECTED_REJECT mode=%s result=%d completed=%u\n",mode.c_str(),result,ps5hip_runtime_observation().completed);return 20;
 }catch(const std::exception&e){std::fprintf(stderr,"HIPIFY_MODEL_FAILURE %s\n",e.what());std::fflush(nullptr);std::_Exit(99);}
}
