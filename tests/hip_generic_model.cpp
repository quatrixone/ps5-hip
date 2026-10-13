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
void check(bool yes){if(!yes)throw std::runtime_error("generic transport invariant");}
Allocation &containing(uint64_t address,size_t bytes) {
    for(auto &entry:allocations){auto &a=entry.second;auto base=reinterpret_cast<uint64_t>(a.cpu);
        if(base && address>=base && address-base<=a.bytes && bytes<=a.bytes-(address-base))return a;}
    throw std::runtime_error("generic model address outside live backing");
}
void dispose(Allocation &a){if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;}}
struct Audit {~Audit(){
    std::printf("GENERIC_MODEL_FINAL live=%zu backends=%u submits=%u releases=%u finalized=%u\n",allocations.size(),backends,submits,releases,unsigned(ps5hip_runtime_finalized()));
    for(auto &entry:allocations)dispose(entry.second);
}} audit;
uint64_t read(const uint8_t *p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
class Backend final:public ps5hip::SubmissionBackend {
    uint64_t clock=0;
public:
    void flush(void*,size_t) override {if(mode=="preflush" && !submits)throw std::runtime_error("preflush");
#ifdef PS5HIP_DEVICE_COPY_TITLE
        if(mode=="postflush" && submits)throw std::runtime_error("postflush");
#endif
}
    int32_t submit(uint32_t *words,uint32_t count) override {
        ++submits;std::map<uint32_t,std::vector<uint32_t>> registers;std::array<uint32_t,3> groups{};bool dispatch=false;
        for(size_t at=0;at<count;) {
            const auto header=words[at];check((header>>30)==3);const size_t length=((header>>16)&0x3fff)+1;check(length<=count-at-1);
            if((header&0xffff)==0x7600){check(length>=2 && !registers.count(words[at+1]));registers.emplace(words[at+1],std::vector<uint32_t>(words+at+2,words+at+length+1));}
            if(header==0xc0031502){check(!dispatch&&words[at+4]==0x8005);dispatch=true;groups={words[at+1],words[at+2],words[at+3]};}
            at+=length+1;
        }
        check(dispatch);
#ifdef PS5HIP_DEVICE_COPY_TITLE
        check(submits<=24);const unsigned pipeline=(submits-1)/2,phase=(submits-1)%2;
        const unsigned iteration=0,seed=pipeline%2,topology=(pipeline/2)%2;
        const unsigned copy_sizes[]{1,73,257},n=copy_sizes[pipeline/4];
#elif defined(PS5HIP_MEMORY_FILL_TITLE)
        check(submits<=6);const unsigned iteration=0,seed=(submits-1)%2;
        const unsigned fill_sizes[]{1,73,257},n=fill_sizes[(submits-1)/2];
#elif defined(PS5HIP_GENERIC_TITLE)
        check(submits<=11);const unsigned sequence=submits<=1?1:submits<=4?2:3;
        const unsigned iteration=submits-(sequence==1?1:sequence==2?2:5),seed=iteration+sequence*17;
#else
        check(submits<=7);const unsigned iteration=submits-1,seed=iteration;
#endif
#if !defined(PS5HIP_MEMORY_FILL_TITLE) && !defined(PS5HIP_DEVICE_COPY_TITLE)
        const unsigned sizes[]{1,63,64,65,127,193,257},n=sizes[iteration];
#endif
        check(groups==std::array<uint32_t,3>{(n+63)/64,1,1} && registers.at(0x207)==std::vector<uint32_t>{64,1,1});
        const auto &s=registers.at(0x240),&p=registers.at(0x20c),&r=registers.at(0x212);
        check(s.size()==6 && p.size()==2 && r.size()==2 && r[1]==(iteration%2?0x1008cu:0x8cu));
        for(unsigned i=0;i<4;++i)check(s[i]==0);
        const uint64_t entry=(uint64_t(p[0])<<8)|(uint64_t(p[1])<<40);code_pointer=containing(entry,4).cpu;
        const uint64_t arg=uint64_t(s[4])|(uint64_t(s[5])<<32);containing(arg,iteration%2?20:32);
        const auto *k=reinterpret_cast<const uint8_t*>(arg);const auto input=read(k,8),output=read(k+8,8);
        check(read(k+16,4)==n);containing(input,n*4);containing(output,n*4);
        const auto *in=reinterpret_cast<const unsigned*>(input);auto *out=reinterpret_cast<unsigned*>(output);
#if !defined(PS5HIP_MEMORY_FILL_TITLE) && !defined(PS5HIP_DEVICE_COPY_TITLE)
        for(unsigned i=0;i<n;++i)check(in[i]==((i*37u+seed*1231u)^0x951dac37u));
#endif
        // Fill variant intentionally consumes actual backing, including injected bad fills.
        // Its independent host oracle must detect a no-op or stale input, not this transport.
        if(mode=="submit")return -7;if(mode=="throw")throw std::runtime_error("submit");if(mode=="unknown")throw 42;
        if(mode!="missing"
#ifdef PS5HIP_DEVICE_COPY_TITLE
            && !(mode=="missing-consumer" && phase)
#endif
        ) {
            if(iteration%2){for(unsigned g=0;g<groups[0];++g){unsigned sum=0;for(unsigned i=g*64;i<n&&i<(g+1)*64;++i)sum+=in[i];out[g]=sum;}}
            else {check(read(k+20,4)==0);const auto salt=read(k+24,8);
#ifdef PS5HIP_DEVICE_COPY_TITLE
                check(salt==(phase?0x2468ace000000071ull:0x13579bdf00000011ull)+seed);
                static uint64_t prior_seed=0,prior_source=0;
                if(!phase){prior_seed=input;prior_source=output;check(&containing(input,n*4)!=&containing(output,n*4));}
                else {
                    check(input!=prior_source && output!=prior_source && output!=prior_seed);
                    if(topology)check(input==prior_source+(n+7)*4 && &containing(input,n*4)==&containing(prior_source,n*4));
                    else check(&containing(input,n*4)!=&containing(prior_source,n*4));
                }
#else
                check(salt==0x13579bdf00000000ull+17+seed);
#endif

                for(unsigned i=0;i<n;++i)out[i]=(in[i]+unsigned(salt))^unsigned(salt>>32);}
        }
        if(mode=="corrupt")out[0]^=1;if(mode=="input")const_cast<unsigned*>(in)[0]^=1;if(mode=="guard")out[-1]=0;
        const auto *end=words+count-8;check(end[-2]==0xc0004600 && end[0]==0xc0064900 && end[1]==0x30c514);
        const uint64_t marker=uint64_t(end[3])|(uint64_t(end[4])<<32);containing(marker,4);
        if(mode!="timeout")*reinterpret_cast<volatile uint32_t*>(marker)=1;return 0;
    }
    int32_t kick() override {return mode=="kick"?-8:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{clock+=uint64_t(us)*1000;}
};
}
extern "C" void ps5hip_generic_model_configure(const char *s){check(!allocation_calls&&!backends&&!submits);mode=s;}
extern "C" unsigned ps5hip_generic_model_live(){return unsigned(allocations.size());}
extern "C" unsigned ps5hip_generic_model_submits(){return submits;}
extern "C" unsigned ps5hip_generic_model_releases(){return releases;}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;if(mode=="init")throw std::runtime_error("init");return std::make_unique<Backend>();}
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

#ifdef PS5HIP_MEMORY_FILL_TITLE
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
extern "C" hipError_t __real_hipMemset(void*,int,size_t);
extern "C" hipError_t __wrap_hipMemset(void *p,int value,size_t bytes) {
    static unsigned calls=0;const unsigned at=calls++%8;
    if(mode=="fill-noop" && at==1)return hipSuccess;
    if(mode=="fill-stale" && at>=4 && at<=6)return hipSuccess;
    if(mode=="fill-byte" && at==1)value=0x24;
    if(mode=="fill-offset" && at==2)p=static_cast<unsigned char*>(p)+1;
    return __real_hipMemset(p,value,bytes);
}
#endif

#ifdef PS5HIP_DEVICE_COPY_TITLE
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
extern "C" hipError_t __real_hipMemcpy(void*,const void*,size_t,hipMemcpyKind);
extern "C" hipError_t __wrap_hipMemcpy(void *dst,const void *src,size_t bytes,hipMemcpyKind kind) {
    static unsigned copies=0;
    if(kind!=hipMemcpyDeviceToDevice)return __real_hipMemcpy(dst,src,bytes,kind);
    const unsigned at=copies++;
    if(mode=="copy-noop" && at==0)return hipSuccess;
    if(mode=="copy-stale" && at==1)return hipSuccess;
    if(mode=="copy-destination")dst=static_cast<unsigned char*>(dst)+4;
    if(mode=="copy-source")src=static_cast<const unsigned char*>(src)+4;
    if(mode=="copy-short")bytes-=4;
    const auto status=__real_hipMemcpy(dst,src,bytes,kind);
    if(status==hipSuccess && mode=="copy-corrupt")*static_cast<unsigned*>(dst)^=1;
    if(status==hipSuccess && mode=="copy-source-mutation")*const_cast<unsigned*>(static_cast<const unsigned*>(src))^=1;
    return status;
}
#endif
