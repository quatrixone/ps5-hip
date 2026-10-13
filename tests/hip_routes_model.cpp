// Host syscall/transport model: checks packets and ownership, never executes AMD ISA.
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

namespace {
struct Allocation { void *cpu=nullptr; size_t bytes=0, alignment=0; };
std::map<int64_t,Allocation> allocations;
int64_t next_id=1;
uint64_t next_address=0x200040000;
unsigned allocations_called=0, backends=0, submits=0, releases=0;
std::string mode;
bool preflush_failed=false, scratch_map_failed=false;
std::array<uint64_t,3> data_pointers{};
std::array<uint64_t,4> entry_offsets{};
std::array<uint32_t,8192> first_input{};
void check(bool value) { if(!value) throw std::runtime_error("routes model invariant"); }
uint64_t read(const uint8_t *p,size_t bytes) {
    uint64_t value=0; for(size_t i=0;i<bytes;++i)value|=uint64_t(p[i])<<(8*i); return value;
}
Allocation &containing(uint64_t address,size_t bytes) {
    for(auto &entry:allocations) {
        auto &a=entry.second;const auto base=reinterpret_cast<uint64_t>(a.cpu);
        if(base && address>=base && address-base<=a.bytes && bytes<=a.bytes-(address-base))return a;
    }
    throw std::runtime_error("routes model address outside live allocation");
}
void dispose(Allocation &a) { if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;} }
struct Audit {
    ~Audit() {
        // The compiler unregister and shared State destructor have already run.
        std::printf("ROUTES_MODEL_FINAL allocations=%zu backends=%u submits=%u releases=%u finalized=%u model_only=true\n",
            allocations.size(),backends,submits,releases,unsigned(ps5hip_runtime_finalized()));
        for(auto &entry:allocations)dispose(entry.second);
    }
} audit;
class Backend final : public ps5hip::SubmissionBackend {
    uint64_t clock=0;
public:
    void flush(void *,size_t) override {
        if(mode=="preflush"&&!preflush_failed){preflush_failed=true;throw std::runtime_error("pre-submit flush");}
        if(mode=="scratch-postflush"&&submits==3)throw std::runtime_error("scratch post-submit flush");
    }
    int32_t submit(uint32_t *words,uint32_t count) override {
        ++submits;
        const unsigned route=(submits-1)%4,round=(submits-1)/4;
        check(round<2 && count>20);
        std::map<uint32_t,std::vector<uint32_t>> registers;
        std::array<uint32_t,3> groups{};
        bool dispatch=false;
        // Decode complete type-3 packet boundaries, not accidental words in payloads.
        for(size_t i=0;i<count;) {
            const uint32_t header=words[i];check((header>>30)==3);
            const size_t payload=((header>>16)&0x3fff)+1;
            check(payload<=count-i-1);
            if((header&0xffff)==0x7600) {
                check(payload>=2 && !registers.count(words[i+1]));
                registers.emplace(words[i+1],std::vector<uint32_t>(words+i+2,words+i+1+payload));
            }
            if(header==0xc0031502) {
                check(!dispatch && words[i+4]==0x8005);dispatch=true;
                groups={words[i+1],words[i+2],words[i+3]};
            }
            i+=payload+1;
        }
        check(dispatch);
        const auto &sgpr=registers.at(0x240),&program=registers.at(0x20c),&resource=registers.at(0x212);
        check(sgpr.size()==6 && program.size()==2 && resource.size()==2);
        check(groups==(route==3?std::array<uint32_t,3>{3,2,1}:std::array<uint32_t,3>{2,1,1}));
        check(registers.at(0x207)==(route==3?std::vector<uint32_t>{8,8,1}:std::vector<uint32_t>{64,1,1}));
        check(resource[1]==std::array<uint32_t,4>{0x8c,0x1008c,0x8d,0x138c}[route]);
        const uint64_t entry=(uint64_t(program[0])<<8)|(uint64_t(program[1])<<40);
        auto &code=containing(entry,4);
        const uint64_t offset=entry-reinterpret_cast<uint64_t>(code.cpu);
        if(!round){entry_offsets[route]=offset;for(unsigned j=0;j<route;++j)check(offset!=entry_offsets[j]);}
        else check(offset==entry_offsets[route]);
        const uint64_t args=uint64_t(sgpr[4])|(uint64_t(sgpr[5])<<32);
        const size_t argument_bytes=route==0?24:route==3?296:16;
        containing(args,argument_bytes);
        const auto *k=reinterpret_cast<const uint8_t*>(args);
        const uint64_t input_address=read(k,8),output_address=read(k+8,8);
        containing(input_address,8192*4);containing(output_address,8192*4);
        if(submits==1){data_pointers[0]=input_address;data_pointers[1]=output_address;}
        check(data_pointers[0]==input_address && data_pointers[1]==output_address);
        const auto *input=reinterpret_cast<const uint32_t*>(input_address);
        auto *output=reinterpret_cast<uint32_t*>(output_address);
        for(unsigned i=0;i<8192;++i) {
            // Independent input oracle; stale sequence-two uploads must be caught.
            check(input[i]==((i*17u+round*1009u+31u)^0x531a9c7du));
            if(!round)first_input[i]=input[i];
            else check(first_input[i]!=input[i]);
        }
        if(route==2) {
            const uint64_t scratch=uint64_t(sgpr[0])|(uint64_t(sgpr[1]&0x7fffffffu)<<32);
            auto &ring=containing(scratch,sgpr[2]);
            check((sgpr[1]&0x80000000u) && sgpr[3]==0x21c14fac && scratch%4096==0);
            check(reinterpret_cast<uint64_t>(ring.cpu)==scratch && ring.bytes>=sgpr[2]);
            check(sgpr[2]==9175040 && registers.at(0x218)==std::vector<uint32_t>{0x7500});
        } else for(unsigned i=0;i<4;++i)check(sgpr[i]==0);
        uint8_t expected[296]{};
        std::memcpy(expected,k,route==0?24:route==3?40:16);
        if(route==3) {
            expected[40]=3;expected[44]=2;expected[48]=1;
            expected[52]=8;expected[54]=8;expected[56]=1;expected[104]=2;
        }
        check(std::memcmp(expected,k,argument_bytes)==0);
        check(hipDeviceSynchronize()==hipErrorNotReady);
        check(hipFree(reinterpret_cast<void*>(input_address-64))==hipErrorNotReady);
        uint32_t sample=0;check(hipMemcpy(&sample,input,4,hipMemcpyDeviceToHost)==hipErrorNotReady);
        if(route==2) {
            if(mode=="scratch-submit")return -7;
            if(mode=="scratch-throw")throw std::runtime_error("scratch submit");
            if(mode=="scratch-unknown")throw 42;
        }
        const auto *source=(mode=="stale-second-round"&&round)?first_input.data():input;
        if(mode!="missing") {
            if(route==0) {
                check(read(k+16,4)==128 && read(k+20,4)==(round?0x22u:0x11u));
                for(unsigned i=0;i<128;++i)output[i]=source[i]^uint32_t(read(k+20,4));
            } else if(route==1) {
                for(unsigned g=0;g<2;++g){uint32_t sum=0;for(unsigned lane=0;lane<64;++lane)sum+=source[g*64+lane];output[g]=sum;}
            } else if(route==2)std::memcpy(output,source,8192*4);
            else {
                check(read(k+24,4)==24 && read(k+28,4)==16 && read(k+32,4)==1 && read(k+36,4)==6144);
                const auto tail_address=read(k+16,8);containing(tail_address,384*4);
                if(!round)data_pointers[2]=tail_address;else check(data_pointers[2]==tail_address);
                auto *tail=reinterpret_cast<uint32_t*>(tail_address);
                for(unsigned i=0;i<384;++i) {
                    const unsigned x=i%24,y=i/24;
                    const uint32_t fields[]{x/8,y/8,0,x%8,y%8,0,8,8,1,3,2,1,x,y,0,source[i]};
                    std::memcpy(output+16*i,fields,sizeof(fields));
                    if(x<21&&y<13)tail[i]=source[i]^(x*17u+y*257u+0x53a971bdu);
                }
            }
        }
        if(mode=="corrupt")output[0]^=1;
        if(mode=="input")const_cast<uint32_t*>(input)[0]^=1;
        if(mode=="guard")output[-1]=0;
        if(mode=="wrong-route-output"&&route==1)output[0]=input[0]^0x11;
        const auto *end=words+count-8;
        check(end[-2]==0xc0004600&&end[-1]==0x407&&end[0]==0xc0064900&&end[1]==0x30c514);
        const auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);containing(marker,4);
        if(!(route==2&&mode=="scratch-timeout"))*reinterpret_cast<volatile uint32_t*>(marker)=1;
        return 0;
    }
    int32_t kick() override {return mode=="scratch-kick"&&submits==3?-8:0;}
    uint64_t now_ns() override {return clock;}
    void sleep_us(unsigned us) override {clock+=uint64_t(us)*1000;}
};
}
extern "C" void ps5hip_routes_model_configure(const char *value){check(!allocations_called&&!backends&&!submits);mode=value;}
extern "C" unsigned ps5hip_routes_model_backends(){return backends;}
extern "C" unsigned ps5hip_routes_model_live(){return unsigned(allocations.size());}
extern "C" unsigned ps5hip_routes_model_submits(){return submits;}
extern "C" unsigned ps5hip_routes_model_releases(){return releases;}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){
    ++backends;if(mode=="init")throw std::runtime_error("backend init");return std::make_unique<Backend>();
}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t *physical){
    check(type==12);++allocations_called;*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
    check(protection==0x33&&flags==0);auto &a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);
    if(mode=="scratch-map"&&bytes>1024*1024&&!scratch_map_failed){scratch_map_failed=true;return -1;}
    next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);
    check(next_address>=0x200000000&&next_address<0x300000000&&bytes<=0x300000000-next_address);
    int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
    const char *options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
    void *cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);
    if(cpu==MAP_FAILED)return -1;
    a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;
}
extern "C" int32_t sceKernelMunmap(void *pointer,size_t bytes){
    for(auto &entry:allocations)if(entry.second.cpu==pointer){
        check(entry.second.bytes==bytes);
        // Keep failing through destructor retry so the retained-backing case is real.
        if(mode=="scratch-cleanup"&&submits==3&&bytes>1024*1024)return -1;
        dispose(entry.second);return 0;
    }
    return -1;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){
    check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;
}
