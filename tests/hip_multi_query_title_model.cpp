// Synthetic transport callbacks validate real packet/ownership routing, not AMD ISA.
#include "ps5hip/hip_runtime_control.h"
#include "hip_multi_query_client.h"
#include "elevation.hpp"
#include <unistd.h>
#include <cerrno>
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
  const bool expected_image=image.size()<=backing.bytes&&std::memcmp(backing.cpu,image.data(),image.size())==0;
  if(mode=="wrong-owner"){check(!expected_image);throw std::runtime_error("injected owning module mismatch");}
  check(expected_image);
  check(entry==reinterpret_cast<uintptr_t>(backing.cpu)+bundle.object.kernels.at(0).entry_address);
  containing(arg,20);auto *k=reinterpret_cast<const uint8_t*>(arg);
  auto in=read(k,8),out=read(k+8,8);unsigned n=unsigned(read(k+16,4));check(n==73);containing(in,n*4);containing(out,n*4);
  if(mode=="submit")return -7;
  for(unsigned i=0;i<n;++i)reinterpret_cast<unsigned*>(out)[i]=reinterpret_cast<const unsigned*>(in)[i]*3u+(b?11:7);
  if(mode=="source-corrupt")reinterpret_cast<unsigned*>(in)[0]^=1;
  if(mode=="output-corrupt")reinterpret_cast<unsigned*>(out)[0]^=1;
  if(mode=="guard-corrupt")reinterpret_cast<unsigned*>(out)[-1]^=1;
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
 __real___hipUnregisterFatBinary(is("reject-unregister")?reinterpret_cast<void**>(uintptr_t(1)):p);
 if(is("duplicate-unregister"))__real___hipUnregisterFatBinary(is("reject-unregister")?reinterpret_cast<void**>(uintptr_t(1)):p);
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;return std::make_unique<Backend>();}

extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t *physical){
    check(type==12);++allocation_calls;if((mode=="alloc-first"&&allocation_calls==1)||(mode=="alloc-second"&&allocation_calls==2))return -1;*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void **pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
    check(protection==0x33 && flags==0);auto &a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);
    next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);check(next_address>=0x200000000&&next_address<0x300000000&&bytes<=0x300000000-next_address);
    int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
    const char *options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
    void *cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);if(cpu==MAP_FAILED)return -1;
    a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;}
extern "C" int32_t sceKernelMunmap(void *pointer,size_t bytes){for(auto &entry:allocations)if(entry.second.cpu==pointer){check(entry.second.bytes==bytes);if(mode=="cleanup"&&pointer==code_pointer)return -1;if(mode=="free-first"&&submits==3&&entry.first==1)return -1;dispose(entry.second);return 0;}return -1;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;}


extern "C" void __real_launch_a(const unsigned*,unsigned*,unsigned),__real_launch_b(const unsigned*,unsigned*,unsigned);
extern "C" void __wrap_launch_a(const unsigned*a,unsigned*b,unsigned n){
 static unsigned calls=0;++calls;
 if(is("skip-final-a")&&calls==2)return;
 if(is("wrong-owner")&&calls==1){__real_launch_b(a,b,n);return;}
 __real_launch_a(a,b,n);
}
extern "C" void __wrap_launch_b(const unsigned*a,unsigned*b,unsigned n){if(!is("skip-b"))__real_launch_b(a,b,n);}
extern "C" hipError_t __real_hipMemPtrGetInfo(void*,size_t*);
extern "C" hipError_t __real_hipMemGetAddressRange(hipDeviceptr_t*,size_t*,hipDeviceptr_t);
extern "C" hipError_t __wrap_hipMemPtrGetInfo(void*p,size_t*n){
 auto e=__real_hipMemPtrGetInfo(p,n);
 if(is("query-size")&&e==hipSuccess)++*n;
 if(is("info-sentinel")&&e==hipErrorInvalidValue)++*n;
 return e;
}
extern "C" hipError_t __wrap_hipMemGetAddressRange(hipDeviceptr_t*b,size_t*n,hipDeviceptr_t p){
 auto e=__real_hipMemGetAddressRange(b,n,p);
 if(is("query-base")&&e==hipSuccess)*b=reinterpret_cast<void*>(uintptr_t(1));
 if(is("range-sentinel")&&e==hipErrorNotFound)++*n;
 return e;
}

extern "C" void ps5hip_install_shell_exit_probe(),ps5hip_install_vector_title_finalizer(),catchReturnFromMain(int),__real_ps5hip_runtime_finish();
int ps5hip_multi_title_main(int,char**);
namespace {
char config[4096]{};unsigned shell_requests=0,fini_calls=0,sync_calls=0;
[[noreturn]] void failed(){std::fflush(nullptr);::_exit(90);}
__attribute__((constructor(101))) void install_early(){ps5hip_install_shell_exit_probe();ps5hip_install_vector_title_finalizer();}
}
extern "C" const char *ps5hip_title_model_config(){return config;}
namespace elevation {
Status request(Capability c,const char *helper) noexcept {
 if(c!=Capability::filesystem||std::strcmp(helper,"/app0/lapy.elf")||!allocations.empty()||backends)failed();
 return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" void __wrap_ps5hip_runtime_finish(){
 if(is("late-live")){void *p=nullptr;if(hipMalloc(&p,16)!=hipSuccess||!p||allocations.size()!=1)failed();std::puts("MULTI_TITLE PRE_FINISH live=1");}
 if(!is("no-finish"))__real_ps5hip_runtime_finish();
}
extern "C" int ps5hip_title_model_fsync(int fd){
 ++sync_calls;if((is("startup-sync")&&sync_calls==1)||(is("final-sync")&&ps5hip_runtime_finalized())){errno=EIO;return -1;}return ::fsync(fd);
}
extern "C" void ps5hip_title_model_fini(){if(!ps5hip_runtime_finalized()||!allocations.empty()||++fini_calls!=1)failed();}
extern "C" int ps5hip_title_model_shell_exit(){if(!ps5hip_runtime_finalized()||!allocations.empty()||fini_calls!=1||++shell_requests!=1)failed();return is("shell-fail")?-7:0;}
extern "C" unsigned ps5hip_title_model_sleep(unsigned){
 const bool success=is("normal")||is("post-finish")||is("shell-fail");
 if(shell_requests!=unsigned(success))failed();
 unsigned live=0,closed=2+4*submits;
 if(is("alloc-first")||is("startup-sync")||is("missing-function")||is("bad-wrapper")||is("duplicate-token"))closed=0;
 if(is("alloc-second"))closed=1;
 if(is("submit")||is("wrong-owner")){live=6;closed=0;}
 if(is("cleanup")){live=3;closed=3;}
 if(is("free-first")){live=2;closed=12;}
 if(is("late-live"))live=1;
 if(allocations.size()!=live||releases!=closed)failed();
 if((is("submit")||is("wrong-owner")||is("cleanup"))&&ps5hip_runtime_gpu_unresolved()!=!is("cleanup"))failed();
 std::printf("MULTI_TITLE STOP shell=%u live=%zu releases=%u submits=%u finalized=%u\n",shell_requests,allocations.size(),releases,submits,unsigned(ps5hip_runtime_finalized()));
 std::fflush(nullptr);::_exit(0);
}
int main(int argc,char **argv){
 if(argc!=3||std::strlen(argv[2])>=sizeof(config))return 91;
 mode=argv[1];std::strcpy(config,argv[2]);
 const int status=ps5hip_multi_title_main(0,nullptr);
 if(is("config")||is("elevation")||is("log-exists")){if(status!=2||!allocations.empty()||backends)failed();std::puts("MULTI_TITLE EARLY_REJECT untouched=true");std::fflush(nullptr);::_exit(0);}
 if(is("post-finish")){const auto n=allocations.size();void*p=reinterpret_cast<void*>(1);if(hipMalloc(&p,16)!=hipErrorDeinitialized||p||allocations.size()!=n||!ps5hip_runtime_ready())failed();std::puts("MULTI_TITLE POST_FINISH rejected=true");}
 catchReturnFromMain(status);std::exit(status);
}
