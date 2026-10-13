// Synthetic transport callbacks validate real packet/ownership routing, not AMD ISA.
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
namespace {
struct Allocation {void *cpu=nullptr;size_t bytes=0,alignment=0;};
std::map<int64_t,Allocation> allocations;
int64_t next_id=1;uint64_t next_address=0x200040000;
unsigned allocation_calls=0,backends=0,submits=0,releases=0;
std::string mode;void *code_pointer=nullptr;
void check_at(bool yes,unsigned line){if(!yes){std::fprintf(stderr,"MULTI_INVARIANT line=%u\n",line);throw std::runtime_error("multi transport invariant");}}
#define check(value) check_at((value),__LINE__)
Allocation &containing(uint64_t address,size_t bytes) {
    for(auto &entry:allocations){auto &a=entry.second;auto base=reinterpret_cast<uint64_t>(a.cpu);
        if(base && address>=base && address-base<=a.bytes && bytes<=a.bytes-(address-base))return a;}
    throw std::runtime_error("generic model address outside live backing");
}
void dispose(Allocation &a){if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;}}
}

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/host_registration.h"
extern "C" const uint8_t bundle_a_begin[],bundle_a_end[],bundle_b_begin[],bundle_b_end[];
extern "C" void launch_a(const unsigned*,unsigned*,unsigned),launch_b(const unsigned*,unsigned*,unsigned);
extern "C" void **__real___hipRegisterFatBinary(const void*);
extern "C" void __real___hipUnregisterFatBinary(void**);
extern "C" void __real___hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
namespace {
void **handles[2]{};const void *wrappers[2]{};unsigned module_calls=0,retire_calls=0;
bool is(const char *s){const char *m=std::getenv("MULTI_MODE");return m&&std::strcmp(m,s)==0;}
struct Audit {~Audit(){
 const bool retained=is("skip-unregister")||is("no-finish")||is("live-finish")||is("bad-wrapper")||is("duplicate-token")||is("missing-function")||is("late-after-error")||is("submit")||is("cleanup");
 check(ps5hip_runtime_finalized()==!retained);
 if(is("submit"))check(allocations.size()==6&&releases==0&&ps5hip_runtime_gpu_unresolved());
 else if(is("cleanup"))check(allocations.size()==3&&releases==3);
 else if(!is("live-finish"))check(allocations.empty());
 std::printf("MULTI_FINAL modules=%u callbacks=%u submits=%u live=%zu releases=%u finalized=%u\n",module_calls,retire_calls,submits,allocations.size(),releases,unsigned(ps5hip_runtime_finalized()));
 for(auto &item:allocations)dispose(item.second);
}} audit;
uint64_t read(const uint8_t *p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
class Backend final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t) override {}
 int32_t submit(uint32_t *words,uint32_t count) override {
  ++submits;std::map<uint32_t,std::vector<uint32_t>> regs;
  for(size_t at=0;at<count;){unsigned h=words[at],n=((h>>16)&0x3fff)+1;check((h>>30)==3&&n<count-at);
   if((h&0xffff)==0x7600)regs.emplace(words[at+1],std::vector<uint32_t>(words+at+2,words+at+n+1));at+=n+1;}
  auto &p=regs.at(0x20c),&s=regs.at(0x240);
  uint64_t entry=(uint64_t(p[0])<<8)|(uint64_t(p[1])<<40),arg=uint64_t(s[4])|(uint64_t(s[5])<<32);
  auto &backing=containing(entry,4);code_pointer=backing.cpu;
  const bool b=submits==2;
  auto *begin=b?bundle_b_begin:bundle_a_begin,*end=b?bundle_b_end:bundle_a_end;
  auto bundle=ps5hip::read_fat_binary(begin,size_t(end-begin),"gfx1013");
  auto image=ps5hip::materialize_code_object(bundle.bytes.data(),bundle.bytes.size());
  check(image.size()<=backing.bytes&&std::memcmp(backing.cpu,image.data(),image.size())==0);
  check(entry==reinterpret_cast<uintptr_t>(backing.cpu)+bundle.object.kernels.at(0).entry_address);
  containing(arg,20);auto *k=reinterpret_cast<const uint8_t*>(arg);
  auto in=read(k,8),out=read(k+8,8);unsigned n=unsigned(read(k+16,4));check(n==73);containing(in,n*4);containing(out,n*4);
  if(mode=="submit")return -7;
  for(unsigned i=0;i<n;++i)reinterpret_cast<unsigned*>(out)[i]=reinterpret_cast<const unsigned*>(in)[i]*3u+(b?11:7);
  const auto *e=words+count-8;uint64_t marker=uint64_t(e[3])|(uint64_t(e[4])<<32);containing(marker,4);*reinterpret_cast<volatile uint32_t*>(marker)=1;return 0;
 }
 int32_t kick() override{return 0;}
 uint64_t now_ns() override{return clock+=1000;}
 void sleep_us(unsigned us) override{clock+=uint64_t(us)*1000;}
};
}
extern "C" void **__wrap___hipRegisterFatBinary(const void *p){
 auto *h=__real___hipRegisterFatBinary(is("bad-wrapper")&&module_calls==1?nullptr:p);
 check(module_calls<2);wrappers[module_calls]=p;handles[module_calls++]=h;return h;
}
extern "C" void __wrap___hipRegisterFunction(void **h,const void *host,char *function,const char *name,unsigned limit,uint3 *tid,uint3 *bid,dim3 *block,dim3 *grid,int *warp){
 static const void *first=nullptr;static unsigned count=0;
 if(!count)first=host;
 ++count;
 if(count==2&&is("missing-function"))return;
 __real___hipRegisterFunction(h,count==2&&is("duplicate-token")?first:host,function,name,limit,tid,bid,block,grid,warp);
}
extern "C" void __wrap___hipUnregisterFatBinary(void **p){
 ++retire_calls;if(is("skip-unregister")&&retire_calls==1)return;
 if(is("wrong-unregister"))__real___hipUnregisterFatBinary(reinterpret_cast<void**>(uintptr_t(1)));
 __real___hipUnregisterFatBinary(p);
 if(is("duplicate-unregister"))__real___hipUnregisterFatBinary(p);
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;return std::make_unique<Backend>();}

extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t *physical){
    check(type==12);++allocation_calls;*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void **pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
    check(protection==0x33 && flags==0);auto &a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);
    next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);check(next_address>=0x200000000&&next_address<0x300000000&&bytes<=0x300000000-next_address);
    int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
    const char *options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
    void *cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);if(cpu==MAP_FAILED)return -1;
    a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;}
extern "C" int32_t sceKernelMunmap(void *pointer,size_t bytes){for(auto &entry:allocations)if(entry.second.cpu==pointer){check(entry.second.bytes==bytes);if(mode=="cleanup"&&pointer==code_pointer)return -1;dispose(entry.second);return 0;}return -1;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;}

int main(){
 std::setvbuf(stdout,nullptr,_IONBF,0);
 mode=std::getenv("MULTI_MODE")?std::getenv("MULTI_MODE"):"normal";
 check(module_calls==2);
 if(is("bad-wrapper")||is("duplicate-token")){check(!ps5hip_runtime_registration_ready());void *p=nullptr;check(hipMalloc(&p,16)!=hipSuccess&&!p);ps5hip_runtime_finish();return 0;}
 if(is("missing-function")){check(!ps5hip_runtime_registration_ready());ps5hip_runtime_finish();check(!ps5hip_runtime_ready());return 0;}
 check(ps5hip_runtime_registration_ready());auto obs=ps5hip_runtime_observation();check(obs.modules==2&&obs.functions==2&&!obs.backend_attempted);
 check(!ps5hip_runtime_handle()&&!ps5hip_runtime_token());
 check(ps5hip_runtime_token_for("multi_a")&&ps5hip_runtime_token_for("multi_b"));
 if(is("late-after-error")){check(hipMemcpy(nullptr,nullptr,4,hipMemcpyHostToDevice)!=hipSuccess);check(!__real___hipRegisterFatBinary(wrappers[0]));check(!ps5hip_runtime_registration_ready()&&allocations.empty()&&!backends);ps5hip_runtime_finish();return 0;}
 unsigned *a=nullptr,*b=nullptr;check(hipMalloc(reinterpret_cast<void**>(&a),87*4)==hipSuccess);check(hipMalloc(reinterpret_cast<void**>(&b),87*4)==hipSuccess);
 std::array<unsigned,87> x{},y{};for(unsigned i=0;i<87;++i)x[i]=i*37+9;y.fill(0xdeadbeef);
 check(hipMemcpy(a,x.data(),sizeof(x),hipMemcpyHostToDevice)==hipSuccess&&hipMemcpy(b,y.data(),sizeof(y),hipMemcpyHostToDevice)==hipSuccess);
 for(unsigned step=0;step<3;++step){
  (step==1?launch_b:launch_a)(a+7,b+7,73);
  if(is("submit")||is("cleanup")){check(hipGetLastError()!=hipSuccess);check(ps5hip_runtime_gpu_unresolved()==is("submit"));ps5hip_runtime_finish();return 0;}
  check(hipGetLastError()==hipSuccess);check(hipMemcpy(y.data(),b,sizeof(y),hipMemcpyDeviceToHost)==hipSuccess);
  for(unsigned i=0;i<87;++i)check(y[i]==(i>=7&&i<80?x[i]*3u+(step==1?11:7):0xdeadbeefu));
  check(hipMemcpy(y.data(),a,sizeof(y),hipMemcpyDeviceToHost)==hipSuccess&&y==x);
  for(unsigned i=7;i<80;++i)x[i]+=19;check(hipMemcpy(a,x.data(),sizeof(x),hipMemcpyHostToDevice)==hipSuccess);
 }
 if(!is("live-finish")){check(hipFree(a)==hipSuccess&&hipFree(b)==hipSuccess);}
 if(!is("no-finish"))ps5hip_runtime_finish();
 check(ps5hip_runtime_ready()==(!is("no-finish")&&!is("live-finish")));
 std::puts("MULTI_PASS launches=3 sequence=A/B/A guarded=true source_preserved=true owning_code=true");return 0;
}
