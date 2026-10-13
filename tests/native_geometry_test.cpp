// Host routing/ownership model, not ISA emulation or hardware proof.
// The callback supplies valid and deliberately invalid geometry observations.
#include "../tools/native_runtime.h"
extern "C" unsigned mock_quarantine_sleep(unsigned);
#define sleep mock_quarantine_sleep
#define main native_title_main_not_called
#include "../tools/native_canary.cpp"
#undef main
#undef sleep
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <sys/wait.h>

namespace {
struct Allocation { void *cpu; uint64_t bytes; bool code; unsigned gpu_uses = 0; };
std::map<int64_t, Allocation> allocations;
int64_t next_id = 1;
unsigned attempted_owned_close = 0;
void require(bool value) { if (!value) throw ps5hip::ObjectError("geometry model assertion"); }
uint64_t address(int64_t id) { return 0x200000000ull + uint64_t(id) * 0x100000; }
void *translate(uint64_t gpu, size_t bytes) {
    for (const auto &[id, a] : allocations)
        if (gpu >= address(id) && gpu - address(id) <= a.bytes && bytes <= a.bytes - (gpu - address(id)))
            return static_cast<char *>(a.cpu) + gpu - address(id);
    throw ps5hip::ObjectError("geometry model address outside live allocation");
}
void corrupt_guard(void *payload, unsigned which) {
    for (const auto &[id,a] : allocations) {
        (void)id;
        if (static_cast<char *>(a.cpu)+64==payload) {
            static_cast<uint8_t *>(a.cpu)[which==0?0:a.bytes-1]^=1; return;
        }
    }
    throw ps5hip::ObjectError("guard allocation missing");
}
struct Model : ps5hip::SubmissionBackend {
    const ps5hip::CodeObject &object;const std::vector<uint8_t> &image;
    unsigned mode,bad_round=1,failure,submits=0;uint64_t clock=0;
    Model(const ps5hip::CodeObject &o,const std::vector<uint8_t> &i,unsigned m,unsigned f=0):object(o),image(i),mode(m),failure(f){}
    void flush(void *,size_t) override {}
    int32_t submit(uint32_t *words,uint32_t length) override {
        require(submits++==0 && allocations.size()==7);for(const auto &[id,a]:allocations){(void)id;require(a.gpu_uses==1);}
        std::map<uint32_t,uint32_t> regs;size_t pos=10;
        while(pos<length && ((words[pos]>>8)&255)==0x76){unsigned n=(words[pos]>>16)&0x3fff;require(n && pos+n+2<=length);for(unsigned i=0;i<n;++i)require(regs.emplace(0xb000+words[pos+1]*4+i*4,words[pos+2+i]).second);pos+=n+2;}
#ifdef PS5HIP_CANARY_GEOMETRY_3D
        const unsigned nx=16,ny=12,nz=8,txs=8,tys=4,tzs=2,gsx=2,gsy=3,gsz=4,logicalx=13,logicaly=9,logicalz=7;
        const char *name="geometry_3d";
#else
        const unsigned nx=24,ny=16,nz=1,txs=8,tys=8,tzs=1,gsx=3,gsy=2,gsz=1,logicalx=21,logicaly=13,logicalz=1;
        const char *name="geometry_2d";
#endif
        const unsigned total=nx*ny*nz;
        require(pos+15==length && words[pos]==0xc0031502 && words[pos+1]==gsx && words[pos+2]==gsy && words[pos+3]==gsz && words[pos+4]==0x8005);
        require(regs.at(0xb81c)==txs && regs.at(0xb820)==tys && regs.at(0xb824)==tzs && regs.at(0xb84c)==0x138c);
        const auto k=std::find_if(object.kernels.begin(),object.kernels.end(),[&](const auto &v){return v.name==name;});require(k!=object.kernels.end());
        const uint64_t entry=uint64_t(regs.at(0xb830))<<8 | uint64_t(regs.at(0xb834))<<40;
        require(std::memcmp(translate(entry-k->entry_address,image.size()),image.data(),image.size())==0);
        auto *raw=static_cast<uint8_t *>(translate(regs.at(0xb910)|uint64_t(regs.at(0xb914))<<32,296));
        uint64_t ptrs[3];std::memcpy(ptrs,raw,24);require(ptrs[0]!=ptrs[1] && ptrs[1]!=ptrs[2] && ptrs[0]!=ptrs[2]);
        const uint32_t scalars[4]={nx,ny,nz,total*16};require(std::memcmp(raw+24,scalars,16)==0);
        auto *input=static_cast<uint32_t *>(translate(ptrs[0],total*4));auto *record=static_cast<uint32_t *>(translate(ptrs[1],total*64));auto *tail=static_cast<uint32_t *>(translate(ptrs[2],total*4));
        for(unsigned i=0;i<total;++i)require(input[i]==(0x71530000u^(i*131u+17u)) && tail[i]==0xa5a5a5a5);
        for(unsigned i=0;i<total*16;++i)require(record[i]==0xa5a5a5a5);
        if(failure==1)return -1;if(failure==4)throw ps5hip::ObjectError("callback error");if(failure==5)throw 7;
        // Axis-explicit loops independently construct callback records; never use
        // native oracle helpers. This is supplied observation, not ISA emulation.
        for(unsigned bz=0;bz<gsz;++bz)for(unsigned by=0;by<gsy;++by)for(unsigned bx=0;bx<gsx;++bx)
        for(unsigned tz=0;tz<tzs;++tz)for(unsigned ty=0;ty<tys;++ty)for(unsigned tx=0;tx<txs;++tx) {
            const unsigned x=bx*txs+tx,y=by*tys+ty,z=bz*tzs+tz,index=(z*ny+y)*nx+x;
            if(mode==1 && index==total-1)continue;
            unsigned values[16]={bx,by,bz,tx,ty,tz,txs,tys,tzs,gsx,gsy,gsz,x,y,z,input[index]};
            if(mode>=2 && mode<=17 && index==total-1)values[mode-2]^=1;
            if(mode==29)std::swap(values[3],values[4]);
            if(mode==30)std::swap(values[9],values[10]);
            std::copy(values,values+16,record+16*index);
            if(x<logicalx && y<logicaly && z<logicalz)tail[index]=input[index]^(17*x+257*y+65537*z+0x53a971bdu);
            else if(mode==18)tail[index]=0;
        }
        if(mode==19)input[total-1]^=1;
        if(mode==20)tail[0]^=1;
        if(mode==21)std::copy(record,record+16,record+16*(total-1));
        if(mode>=22 && mode<=27)corrupt_guard(mode<24?static_cast<void*>(input):mode<26?static_cast<void*>(record):static_cast<void*>(tail),mode%2);
        if(mode==28)record[total*16]^=1;
        if(mode==31)std::copy(record,record+32*16,record+32*16);
        auto *end=words+length-8;require(end[-2]==0xc0004600 && end[-1]==0x407 && end[0]==0xc0064900 && end[5]==1);
        auto *marker=static_cast<uint32_t *>(translate(uint64_t(end[3])|uint64_t(end[4])<<32,4));require(*marker==0);
        if(!failure)*marker=1;
        return 0;
    }
    int32_t kick() override{return failure==2?-1:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{require(failure==3);clock+=uint64_t(us)*1000;}
};
Model *quarantined_model=nullptr;ps5hip::SubmissionQueue *quarantined_queue=nullptr;
}
namespace ps5hip {
NativeMemory::NativeMemory(uint64_t bytes, uint64_t, bool code) {
    bytes_ = (bytes + 0x3fff) & ~uint64_t(0x3fff);
    cpu_ = std::calloc(1, bytes_);
    if (!cpu_) throw std::bad_alloc();
    physical_ = next_id++;
    allocations.emplace(physical_, Allocation{cpu_, bytes_, code});
}
NativeMemory::~NativeMemory() { if (!close()) std::abort(); }
GpuRange NativeMemory::range() const { require(cpu_ != nullptr); return {address(physical_), bytes_}; }
void NativeMemory::acquire_gpu_use() { ++gpu_uses_; ++allocations.at(physical_).gpu_uses; }
void NativeMemory::complete_gpu_use() { require(gpu_uses_ != 0); --gpu_uses_; --allocations.at(physical_).gpu_uses; }
bool NativeMemory::close() {
    if (gpu_uses_) { ++attempted_owned_close; return false; }
    if (cpu_) { std::free(cpu_); allocations.erase(physical_); cpu_ = nullptr; }
    return true;
}
}
extern "C" int32_t sceAgcInit(uint32_t) { std::abort(); }
extern "C" int32_t sceAgcDriverSubmitDcb(void *) { std::abort(); }
extern "C" int32_t sceAgcSuspendPoint() { std::abort(); }
extern "C" unsigned mock_quarantine_sleep(unsigned seconds) {
    bool ok=seconds==1 && quarantined_queue && quarantined_queue->poisoned() && quarantined_model &&
        quarantined_model->submits==quarantined_model->bad_round && allocations.size()==7 && attempted_owned_close==0;
    for(const auto &[id,a]:allocations){(void)id;ok=ok && a.gpu_uses==1;}
    ::_exit(ok?0:7);
}
int main(int argc,char **argv) {
 try{
    require(argc==2);std::ifstream f(argv[1],std::ios::binary);const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),{});
    auto object=ps5hip::inspect_code_object(bytes.data(),bytes.size());auto image=ps5hip::materialize_code_object(bytes.data(),bytes.size());require(object.kernels.size()==2);
    for(const auto &k:object.kernels)require(!ps5hip::initial_launch_rejection(k).empty() && ps5hip::geometry_launch_rejection(k).empty());
    for(unsigned mode=0;mode<=31;++mode){
        Model m(object,image,mode);ps5hip::SubmissionQueue q(m);bool rejected=false;
        try{run(q,object,image);}catch(const ps5hip::ObjectError &){rejected=true;}
        require(rejected==(mode!=0) && !q.poisoned() && allocations.empty() && attempted_owned_close==0 && m.submits==1);
    }
    for(unsigned failure=1;failure<=5;++failure){
        const pid_t child=::fork();require(child>=0);
        if(child==0){Model m(object,image,0,failure);ps5hip::SubmissionQueue q(m);quarantined_model=&m;quarantined_queue=&q;run(q,object,image);::_exit(3);}
        int status=0;require(::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    }
    std::puts("geometry native model: exact coordinate/tail records,31 corruptions,5 retained failures; no GPU execution");
 }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
