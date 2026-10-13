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
void check(bool yes){if(!yes)throw std::runtime_error("matmul transport invariant");}
Allocation &containing(uint64_t address,size_t bytes) {
    for(auto &entry:allocations){auto &a=entry.second;auto base=reinterpret_cast<uint64_t>(a.cpu);
        if(base && address>=base && address-base<=a.bytes && bytes<=a.bytes-(address-base))return a;}
    throw std::runtime_error("matmul model address outside live backing");
}
void dispose(Allocation &a){if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;}}
struct Audit {~Audit(){
    std::printf("GEMM_MODEL_FINAL live=%zu backends=%u submits=%u releases=%u finalized=%u\n",allocations.size(),backends,submits,releases,unsigned(ps5hip_runtime_finalized()));
    for(auto &entry:allocations)dispose(entry.second);
}} audit;
uint64_t read(const uint8_t *p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
class Backend final:public ps5hip::SubmissionBackend {
    uint64_t clock=0;
public:
    void flush(void*,size_t) override {if(mode=="preflush" && !submits)throw std::runtime_error("preflush");}
    int32_t submit(uint32_t *words,uint32_t count) override {
        ++submits;std::map<uint32_t,std::vector<uint32_t>> registers;std::array<uint32_t,3> groups{};bool dispatch=false;
        for(size_t at=0;at<count;) {
            const auto header=words[at];check((header>>30)==3);const size_t length=((header>>16)&0x3fff)+1;check(length<=count-at-1);
            if((header&0xffff)==0x7600){check(length>=2 && !registers.count(words[at+1]));registers.emplace(words[at+1],std::vector<uint32_t>(words+at+2,words+at+length+1));}
            if(header==0xc0031502){check(!dispatch&&words[at+4]==0x8005);dispatch=true;groups={words[at+1],words[at+2],words[at+3]};}
            at+=length+1;
        }
        check(dispatch);
        check(registers.at(0x207)==std::vector<uint32_t>{8,8,1});
        const auto &s=registers.at(0x240),&p=registers.at(0x20c);
        check(s.size()==6&&p.size()==2);
        code_pointer=containing((uint64_t(p[0])<<8)|(uint64_t(p[1])<<40),4).cpu;
        const auto arg=uint64_t(s[4])|(uint64_t(s[5])<<32);containing(arg,328);
        auto *k=reinterpret_cast<const uint8_t*>(arg);
        auto av=read(k,8),bv=read(k+8,8),cv=read(k+16,8);
        unsigned m=read(k+24,4),n=read(k+28,4),inner=read(k+32,4);
        size_t lda=read(k+40,8),ldb=read(k+48,8),ldc=read(k+56,8);
        float alpha,beta;std::memcpy(&alpha,k+64,4);std::memcpy(&beta,k+68,4);
        check(groups==std::array<uint32_t,3>{n/8+(n%8!=0),m/8+(m%8!=0),1});
        containing(cv,((m-1)*ldc+n)*4);
        if(inner){containing(av,((m-1)*lda+inner)*4);containing(bv,((inner-1)*ldb+n)*4);}
        const auto*a=reinterpret_cast<const float*>(av);const auto*b=reinterpret_cast<const float*>(bv);auto*c=reinterpret_cast<float*>(cv);
        if(mode=="submit")return -7;if(mode=="throw")throw std::runtime_error("submit");
        // Synthetic float arithmetic tests host orchestration, never GPU ISA.
        if(mode!="missing")for(unsigned row=0;row<m;++row)for(unsigned col=0;col<n;++col){
            float sum=0;for(unsigned j=0;j<inner;++j)sum+=a[row*lda+j]*b[j*ldb+col];
            float value=alpha*sum;if(beta!=0)value+=beta*c[row*ldc+col];c[row*ldc+col]=value;
        }
        if(mode=="corrupt")c[0]+=1;
        if(mode=="input"&&inner)const_cast<float*>(a)[0]+=1;
        if(mode=="guard")c[-1]=0;
        if(mode=="padding")c[n]=0;
        const auto *end=words+count-8;check(end[-2]==0xc0004600 && end[0]==0xc0064900 && end[1]==0x30c514);
        const uint64_t marker=uint64_t(end[3])|(uint64_t(end[4])<<32);containing(marker,4);
        if(mode!="timeout")*reinterpret_cast<volatile uint32_t*>(marker)=1;return 0;
    }
    int32_t kick() override {return mode=="kick"?-8:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{clock+=uint64_t(us)*1000;}
};
}
extern "C" void ps5hip_matmul_model_configure(const char *s){check(!allocation_calls&&!backends&&!submits);mode=s;}
extern "C" unsigned ps5hip_matmul_model_live(){return unsigned(allocations.size());}
extern "C" unsigned ps5hip_matmul_model_submits(){return submits;}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;if(mode=="init")throw std::runtime_error("init");return std::make_unique<Backend>();}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t *physical){
    check(type==12);++allocation_calls;if(mode=="alloc"+std::to_string(allocation_calls))return -1;*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void **pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
    check(protection==0x33 && flags==0);if(mode=="map"+std::to_string(allocation_calls))return -1;auto &a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);
    next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);check(next_address>=0x200000000&&next_address<0x300000000&&bytes<=0x300000000-next_address);
    int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
    const char *options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
    void *cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);if(cpu==MAP_FAILED)return -1;
    a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;}
extern "C" int32_t sceKernelMunmap(void *pointer,size_t bytes){for(auto &entry:allocations)if(entry.second.cpu==pointer){check(entry.second.bytes==bytes);if(mode=="cleanup"&&pointer==code_pointer)return -1;dispose(entry.second);return 0;}return -1;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;}
