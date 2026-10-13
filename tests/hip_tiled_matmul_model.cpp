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
    std::printf("MATMUL_MODEL_FINAL live=%zu backends=%u submits=%u releases=%u finalized=%u\n",allocations.size(),backends,submits,releases,unsigned(ps5hip_runtime_finalized()));
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
        check(dispatch && submits<=8);
        const unsigned shapes[][3]={{8,8,8},{13,21,19},{1,1,17},{17,9,1}};
        const unsigned shape=(submits-1)/2,round=(submits-1)%2,m=shapes[shape][0],n=shapes[shape][1],inner=shapes[shape][2];
        check(groups==std::array<uint32_t,3>{(n+7)/8,(m+7)/8,1}&&registers.at(0x207)==std::vector<uint32_t>{8,8,1});
        const auto &s=registers.at(0x240),&p=registers.at(0x20c),&r=registers.at(0x212);
        check(s.size()==6&&p.size()==2&&r.size()==2&&r[1]==0x1098c);
        for(unsigned i=0;i<4;++i)check(s[i]==0);
        const uint64_t entry=(uint64_t(p[0])<<8)|(uint64_t(p[1])<<40);code_pointer=containing(entry,4).cpu;
        const uint64_t arg=uint64_t(s[4])|(uint64_t(s[5])<<32);containing(arg,296);
        const auto *k=reinterpret_cast<const uint8_t*>(arg);const uint64_t av=read(k,8),bv=read(k+8,8),cv=read(k+16,8);
        check(read(k+24,4)==m&&read(k+28,4)==n&&read(k+32,4)==inner&&read(k+36,4)==0);
        std::array<uint8_t,256> hidden{};
        auto put=[&](unsigned off,uint64_t v,unsigned bytes){for(unsigned i=0;i<bytes;++i)hidden.at(off+i)=uint8_t(v>>(8*i));};
        put(0,groups[0],4);put(4,groups[1],4);put(8,1,4);put(12,8,2);put(14,8,2);put(16,1,2);put(64,2,2);
        check(std::memcmp(k+40,hidden.data(),hidden.size())==0);
        auto &aa=containing(av,m*inner*4),&ba=containing(bv,inner*n*4),&ca=containing(cv,m*n*4);
        check(&aa!=&ba&&&aa!=&ca&&&ba!=&ca);
        check(av-reinterpret_cast<uint64_t>(aa.cpu)==64&&bv-reinterpret_cast<uint64_t>(ba.cpu)==64&&cv-reinterpret_cast<uint64_t>(ca.cpu)==64);
        const auto *a=reinterpret_cast<const float*>(av),*b=reinterpret_cast<const float*>(bv);auto *c=reinterpret_cast<float*>(cv);
        for(unsigned row=0;row<m;++row)for(unsigned i=0;i<inner;++i)check(a[row*inner+i]==float(int((row*3+i*5+round*11)%7)-3));
        for(unsigned i=0;i<inner;++i)for(unsigned col=0;col<n;++col)check(b[i*n+col]==float(int((i*7+col*3+round*11)%5)-2));
        for(unsigned i=0;i<m*n;++i){uint32_t value;std::memcpy(&value,c+i,4);check(value==0x7fc12345);}
        if(mode=="submit")return -7;if(mode=="throw")throw std::runtime_error("submit");if(mode=="unknown")throw 42;
        if(mode!="missing")for(unsigned index=0;index<m*n;++index) {
            int sum=0;for(unsigned i=0;i<inner;++i)sum+=int(a[(index/n)*inner+i])*int(b[i*n+index%n]);c[index]=float(sum);
        }
        if(mode=="corrupt")c[0]+=1;
        if(mode=="input")const_cast<float*>(a)[0]+=1;if(mode=="input-b")const_cast<float*>(b)[0]+=1;
        if(mode=="guard")c[-1]=0;if(mode=="tail")c[m*n]=0;
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
