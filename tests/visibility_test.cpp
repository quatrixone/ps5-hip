// Host routing/ownership model, not ISA emulation or hardware proof.
// The callback supplies valid and deliberately invalid visibility observations.
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
void require(bool value) { if (!value) throw ps5hip::ObjectError("visibility model assertion"); }
uint64_t address(int64_t id) { return 0x200000000ull + uint64_t(id) * 0x100000; }
void *translate(uint64_t gpu, size_t bytes) {
    for (const auto &[id, a] : allocations)
        if (gpu >= address(id) && gpu - address(id) <= a.bytes && bytes <= a.bytes - (gpu - address(id)))
            return static_cast<char *>(a.cpu) + gpu - address(id);
    throw ps5hip::ObjectError("visibility model address outside live allocation");
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
    const ps5hip::CodeObject &object; const std::vector<uint8_t> &image;
    unsigned mode, bad_round, failure, submits=0; uint64_t clock=0;
    std::set<int64_t> persistent; uint64_t input_gpu=0,middle_gpu=0,output_gpu=0;
    Model(const ps5hip::CodeObject &o,const std::vector<uint8_t> &i,unsigned m,unsigned b,unsigned f=0)
        :object(o),image(i),mode(m),bad_round(b),failure(f) {}
    void flush(void *,size_t) override {}
    int32_t submit(uint32_t *words,uint32_t length) override {
        require(submits<6 && allocations.size()==7);const unsigned phase=submits%2,round=submits/2;++submits;
        std::set<int64_t> ids;for(const auto &[id,a]:allocations){require(a.gpu_uses==1);if(ids.size()<5)ids.insert(id);}
        if(persistent.empty())persistent=ids;else require(persistent==ids);
        std::map<uint32_t,uint32_t> regs;size_t pos=10;
        while(pos<length && ((words[pos]>>8)&255)==0x76){unsigned n=(words[pos]>>16)&0x3fff;require(n && pos+n+2<=length);for(unsigned i=0;i<n;++i)regs[0xb000+words[pos+1]*4+i*4]=words[pos+2+i];pos+=n+2;}
        require(pos+15==length && words[pos]==0xc0031502 && words[pos+1]==5 && words[pos+2]==1 && words[pos+3]==1);
        require(regs.at(0xb81c)==64 && regs.at(0xb820)==1 && regs.at(0xb824)==1);
        const char *name=phase?"visibility_consumer":"visibility_producer";
        const auto k=std::find_if(object.kernels.begin(),object.kernels.end(),[&](const auto &v){return v.name==name;});require(k!=object.kernels.end());
        const uint64_t entry=uint64_t(regs.at(0xb830))<<8 | uint64_t(regs.at(0xb834))<<40;
        require(k->kernarg_size==16 && k->group_bytes==0 && k->private_bytes==0 && k->rsrc2==0x8c && k->properties==0x409);
        require(std::memcmp(translate(entry-k->entry_address,image.size()),image.data(),image.size())==0);
        uint64_t args[2];std::memcpy(args,translate(regs.at(0xb910)|uint64_t(regs.at(0xb914))<<32,16),16);
        if(!phase){if(!input_gpu){input_gpu=args[0];middle_gpu=args[1];}require(args[0]==input_gpu && args[1]==middle_gpu);}
        else {if(!output_gpu)output_gpu=args[1];require(args[0]==middle_gpu && args[1]==output_gpu && input_gpu!=output_gpu && output_gpu!=middle_gpu);}
        if(submits==bad_round && failure==1)return -1;
        if(submits==bad_round && failure==4)throw ps5hip::ObjectError("callback error");
        if(submits==bad_round && failure==5)throw 7;
        auto *src=static_cast<uint32_t *>(translate(args[0],257*4));auto *dst=static_cast<uint32_t *>(translate(args[1],257*4));
        if(!phase){
            for(unsigned i=0;i<257;++i){
                const uint32_t want=(0xfedcba98u+round*0x31415927u)^(i*0x1020305u);require(src[i]==want);
                // Old intermediate must persist unchanged until the next producer.
                if(round==0)require(dst[i]==0xa5a5a5a5u);
                else {const uint32_t old=(0xfedcba98u+(round-1)*0x31415927u)^(i*0x1020305u);require(dst[i]==old*17u+i*31u+0x13579bdfu);}
            }
            if(!(mode==1 || (mode==4 && round>0)))for(unsigned i=0;i<257;++i)dst[i]=src[i]*17u+i*31u+0x13579bdfu;
            if(mode==2)dst[128]^=1;
        }else{
            for(unsigned i=0;i<257;++i)require(dst[i]==0xa5a5a5a5u);
            if(mode==3)src=static_cast<uint32_t *>(translate(input_gpu,257*4));
            for(unsigned i=0;i<257;++i)dst[i]=(src[i]^0xa39d71b5u)+src[(i+113)%257]*3u;
            if(mode==5)dst[256]^=1;
            if(mode==6)static_cast<uint32_t *>(translate(input_gpu,257*4))[64]^=1;
            if(mode==7)src[256]^=1;
            if(mode==8)corrupt_guard(dst,0);
            if(mode==9)dst[257]^=1;
            if(mode==10)corrupt_guard(dst,1);
        }
        auto *end=words+length-8;require(end[-2]==0xc0004600 && end[-1]==0x407 && end[0]==0xc0064900 && end[5]==1);
        auto *marker=static_cast<uint32_t *>(translate(uint64_t(end[3])|uint64_t(end[4])<<32,4));require(*marker==0);
        if(!(submits==bad_round && failure))*marker=1;
        return 0;
    }
    int32_t kick() override{return submits==bad_round && failure==2?-1:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{require(failure==3 && submits==bad_round);clock+=uint64_t(us)*1000;}
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
    // Actual compiled metadata must remain inside the existing scratch-free route.
    for(const auto &kernel:object.kernels){
        require(ps5hip::initial_launch_rejection(kernel).empty());
        for(unsigned kind=0;kind<5;++kind){auto bad=kernel;
            if(kind==0)bad.private_bytes=4;else if(kind==1)bad.group_bytes=256;
            else if(kind==2)bad.wave_size=64;else if(kind==3)bad.rsrc3=1;else bad.preload=1;
            require(!ps5hip::initial_launch_rejection(bad).empty());
        }
    }
    auto broken=bytes;broken[0]=0;bool parser_rejected=false;
    try{ps5hip::inspect_code_object(broken.data(),broken.size());}catch(const ps5hip::ObjectError &){parser_rejected=true;}
    require(parser_rejected);
    for(unsigned mode=0;mode<=10;++mode){
        Model m(object,image,mode,0);ps5hip::SubmissionQueue q(m);bool rejected=false;
        try{run(q,object,image);}catch(const ps5hip::ObjectError &){rejected=true;}
        require(rejected==(mode!=0) && !q.poisoned() && allocations.empty() && attempted_owned_close==0);
        require(m.submits==(mode==0?6:mode==4?4:2));
    }
    for(unsigned step=1;step<=6;++step)for(unsigned failure=1;failure<=5;++failure){
        const pid_t child=::fork();require(child>=0);
        if(child==0){Model m(object,image,0,step,failure);ps5hip::SubmissionQueue q(m);quarantined_model=&m;quarantined_queue=&q;run(q,object,image);::_exit(3);}
        int status=0;require(::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    }
    std::puts("visibility model: three persistent rounds, ten corruption paths, thirty quarantined failures; no GPU ISA execution");
 }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
