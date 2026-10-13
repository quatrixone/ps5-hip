// Host transport model only. No AMD ISA is executed or emulated.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/submission.h"
#include <ps5platform/kernel.h>
#include <sys/mman.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

namespace {
struct Allocation { void *cpu=nullptr; size_t bytes=0, alignment=0; bool mapped_code=false; };
std::map<int64_t,Allocation> allocations;
int64_t next_id=1;
uint64_t next_code=0x200040000;
unsigned allocations_called=0, backend_calls=0, submits=0, releases=0;
std::string mode;
bool flush_failed=false, cleanup_failed=false;
unsigned cleanup_failures=0;
void check(bool v){if(!v)throw std::runtime_error("vector model invariant");}
uint64_t read64(const uint8_t*p){uint64_t v;std::memcpy(&v,p,8);return v;}
void dispose(Allocation &a){if(a.cpu){if(a.mapped_code)::munmap(a.cpu,a.bytes);else std::free(a.cpu);a.cpu=nullptr;}}
struct Audit {
    ~Audit(){
        std::printf("VECTOR_MODEL_FINAL allocations=%zu backends=%u submits=%u releases=%u model_only=true\n",allocations.size(),backend_calls,submits,releases);
        std::printf("VECTOR_MODEL_CLEANUP failures=%u\n",cleanup_failures);
        // Quarantined native destructors must not release GPU-held mappings. Only
        // this host model disposes its private storage after that audit.
        for(auto &a:allocations)dispose(a.second);
    }
} audit;
class Backend final:public ps5hip::SubmissionBackend {
    uint64_t now=0;
    unsigned flushes=0;
public:
    void flush(void*,size_t)override{
        ++flushes;
        if((mode=="preflush"||mode=="preflush-unmap"||mode=="preflush-release")&&!flush_failed){flush_failed=true;throw std::runtime_error("pre-submit flush");}
        if(mode=="postflush"&&submits)throw std::runtime_error("post-submit flush");
    }
    int32_t submit(uint32_t *words,uint32_t count)override{
        ++submits;
        check(count>20);
        const auto *end=words+count-8;
        check(end[-2]==0xc0004600&&end[-1]==0x407&&end[0]==0xc0064900&&end[1]==0x30c514);
        const auto marker=uint64_t(end[3])|uint64_t(end[4])<<32;
        uint64_t kernarg=0,entry=0;
        for(size_t i=0;i+8<count-10;++i){
            if(words[i]==0xc0067600&&words[i+1]==0x240){
                check(words[i+2]==0&&words[i+3]==0&&words[i+4]==0&&words[i+5]==0);
                kernarg=uint64_t(words[i+6])|uint64_t(words[i+7])<<32;
            }
            if(words[i]==0xc0027600&&words[i+1]==0x20c)
                entry=(uint64_t(words[i+2])<<8)|(uint64_t(words[i+3])<<40);
        }
        check(kernarg&&entry>=0x200000000&&entry<0x300000000);
        const auto *k=reinterpret_cast<const uint8_t*>(kernarg);
        const auto *a=reinterpret_cast<const float*>(read64(k));
        const auto *b=reinterpret_cast<const float*>(read64(k+8));
        auto *out=reinterpret_cast<float*>(read64(k+16));
        uint32_t n;std::memcpy(&n,k+24,4);check(n==73);
        // Independently construct all288 bytes, including all reserved implicit bytes.
        uint8_t expected[288]{};std::memcpy(expected,k,28);
        expected[32]=2;expected[36]=1;expected[40]=1;expected[44]=64;expected[46]=1;expected[48]=1;expected[96]=1;
        check(std::memcmp(k,expected,288)==0);
        check(words[count-15]==0xc0031502&&words[count-14]==2&&words[count-13]==1&&words[count-12]==1&&words[count-11]==0x8005);
        check(hipDeviceSynchronize()==hipErrorNotReady);
        check(hipFree(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(out)-64))==hipErrorNotReady);
        float dummy=0;check(hipMemcpy(&dummy,out,4,hipMemcpyDeviceToHost)==hipErrorNotReady);
        if(mode=="throw-submit")throw std::runtime_error("submit exception");
        if(mode=="unknown-submit")throw 42;
        if(mode=="submit")return -7;
        if(mode!="missing")for(unsigned i=0;i<n;++i)out[i]=a[i]+b[i];
        if(mode=="corrupt")out[17]+=1;
        if(mode=="input")const_cast<float*>(a)[4]+=1;
        if(mode=="guard")out[73]=1;
        if(mode!="timeout")*reinterpret_cast<volatile uint32_t*>(marker)=1;
        return 0;
    }
    int32_t kick()override{if(mode=="throw-kick")throw std::runtime_error("kick exception");return mode=="kick"?-8:0;}
    uint64_t now_ns()override{return now;}
    void sleep_us(unsigned us)override{now+=uint64_t(us)*1000;}
};
}
extern "C" void ps5hip_vector_model_configure(const char*value){check(!allocations_called&&!backend_calls&&!submits);mode=value;}
extern "C" unsigned ps5hip_vector_model_backends(){return backend_calls;}
extern "C" unsigned ps5hip_vector_model_live(){return unsigned(allocations.size());}
extern "C" unsigned ps5hip_vector_model_submits(){return submits;}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_vector_model_backend(){
    ++backend_calls;if(mode=="init"||mode=="init-unmap"||mode=="init-release")throw std::runtime_error("init failure");return std::make_unique<Backend>();
}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t *physical){
    check(type==12);++allocations_called;
    if(mode=="code-allocate"&&allocations_called==5)return -1;
    *physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment,false});return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **p,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
    check(protection==0x33&&flags==0);auto&a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);
    if((mode=="kernarg-map"&&physical==6)||(mode=="command-map"&&physical==7)||
       ((mode=="marker-map"||mode=="marker-map-unmap")&&physical==8))return -1;
    // Code is recognizable by ELF materialization size/alignment? NativeMemory's
    // same API carries no code flag, so reserve every mapping in the valid window.
    next_code=(next_code+alignment-1)&~(uint64_t(alignment)-1);
    check(next_code>=0x200000000&&next_code<0x300000000&&bytes<=0x300000000-next_code);
    int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
    const char *options=std::getenv("ASAN_OPTIONS");
    check(options&&std::strstr(options,"protect_shadow_gap=0"));
    fixed=MAP_FIXED;
#endif
    void *cpu=mmap(reinterpret_cast<void*>(next_code),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);
    if(cpu==MAP_FAILED)return -1;
    a.cpu=cpu;a.mapped_code=true;next_code+=bytes;std::memset(cpu,0xa5,bytes);*p=cpu;return 0;
}
extern "C" int32_t sceKernelMunmap(void*p,size_t bytes){
    for(auto &a:allocations)if(a.second.cpu==p){
        check(a.second.bytes==bytes);
        if(!cleanup_failed&&((mode=="init-unmap"&&a.first==5)||
           ((mode=="preflush-unmap"||mode=="marker-map-unmap"||mode=="completed-unmap")&&a.first==7))){
            cleanup_failed=true;++cleanup_failures;return -1;
        }
        dispose(a.second);return 0;
    }
    return -1;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);
    if(!cleanup_failed&&((mode=="init-release"&&physical==6)||
       (mode=="preflush-release"&&physical==7)||(mode=="completed-release"&&physical==8))){
        cleanup_failed=true;++cleanup_failures;return -1;
    }
    allocations.erase(physical);++releases;return 0;}
