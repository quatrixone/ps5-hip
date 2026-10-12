// Host-only routing model: actual harness, parser, launch encoder and queue.
// Synthetic GPU addresses translate to ASan-covered host storage. No GPU or
// native allocator behavior is emulated as hardware evidence.
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
void require(bool value) { if (!value) throw ps5hip::ObjectError("suite model assertion"); }
uint64_t address(int64_t id) { return 0x200000000ull + uint64_t(id) * 0x10000; }
void *translate(uint64_t gpu, size_t bytes) {
    for (const auto &[id, allocation] : allocations)
        if (gpu >= address(id) && gpu - address(id) <= allocation.bytes &&
            bytes <= allocation.bytes - (gpu - address(id)))
            return static_cast<char *>(allocation.cpu) + gpu - address(id);
    throw ps5hip::ObjectError("model address outside live allocation");
}
struct Model : ps5hip::SubmissionBackend {
    const ps5hip::CodeObject &object;
    const std::vector<uint8_t> &image;
    unsigned mode, failure, submits=0, polls=0;
    uint64_t clock=0;
    Model(const ps5hip::CodeObject &o,const std::vector<uint8_t> &i,unsigned m,unsigned f=0)
        : object(o),image(i),mode(m),failure(f) {}
    void flush(void *, size_t) override {}
    int32_t submit(uint32_t *words,uint32_t length) override {
        require(++submits==1 && allocations.size()==5);
        std::map<uint32_t,uint32_t> registers;
        size_t pos=10;
        while(pos<length && ((words[pos]>>8)&255)==0x76) {
            unsigned n=(words[pos]>>16)&0x3fff;
            require(n && pos+n+2<=length);
            for(unsigned i=0;i<n;++i) registers[0xb000+words[pos+1]*4+i*4]=words[pos+2+i];
            pos+=n+2;
        }
        require(pos+15==length && words[pos]==0xc0031502 && words[pos+1]==1 &&
                words[pos+2]==1 && words[pos+3]==1 && words[pos+4]==0x8005);
        require(registers.at(0xb81c)==64 && registers.at(0xb820)==1 && registers.at(0xb824)==1);
        require(object.kernels.size()==1);
        const auto &kernel=object.kernels.front();
        require(kernel.name=="private_aperture" && kernel.kernarg_size==8 && kernel.arguments.size()==1);
        require(kernel.private_bytes==0 && kernel.group_bytes==0 && kernel.properties==0x409 && kernel.rsrc2==0x8c);
        const uint64_t entry=uint64_t(registers.at(0xb830))<<8 | uint64_t(registers.at(0xb834))<<40;
        require(entry>=kernel.entry_address);
        require(std::memcmp(translate(entry-kernel.entry_address,image.size()),image.data(),image.size())==0);
        require(registers.at(0xb848)==kernel.rsrc1 && registers.at(0xb84c)==kernel.rsrc2);
        for(unsigned i=0;i<4;++i) require(registers.at(0xb900+4*i)==0);
        const uint64_t args_address=registers.at(0xb910) | uint64_t(registers.at(0xb914))<<32;
        uint64_t output_address=0;
        std::memcpy(&output_address,translate(args_address,8),8);
        auto *output=static_cast<uint32_t *>(translate(output_address,384*4));
        if(failure==1)return -10;
        if(failure==4)throw ps5hip::ObjectError("model post-submit exception");
        if(failure==5)throw 7;
        for(unsigned lane=0;lane<64;++lane) {
            auto *record=output+6*lane;
            for(unsigned j=0;j<6;++j)require(record[j]==(j<4?aperture::raw_poison(lane,j):poison));
            if(mode!=2) {
                // These deliberately arbitrary values include zero, high bits,
                // base>limit and distinct waves. They are observations only.
                record[0]=0xffffffff;record[1]=0x81234567;
                record[2]=0;record[3]=0x12345678;
                if(mode==1 && lane>=32) {record[0]=0;record[1]=0;record[2]=0xffffffff;record[3]=0xffffffff;}
                if(mode==19 || mode==20)
                    for(unsigned j=0;j<4;++j)record[j]=mode==19?0:poison;
                record[4]=lane;record[5]=lane^0x6b18d3a5;
                for(unsigned j=0;j<4;++j)
                    if(mode==13 || mode==14+j || (mode==18 && lane>=48))
                        record[j]=aperture::raw_poison(lane,j);
            }
        }
        if(mode==3)output[63*6+4]=62;
        if(mode==4)output[32*6+5]^=1;
        if(mode==5)output[31*6]^=1;
        if(mode==6)output[63*6+3]^=1;
        if(mode==7)output[-16]^=1;
        if(mode==8)output[384]^=1;
        if(mode==9) {
            for(auto &[id,a]:allocations)if(translate(address(id),1)==static_cast<void *>(output-16))
                static_cast<uint32_t *>(a.cpu)[a.bytes/4-1]^=1;
        }
        if(mode==11)output[31*6+1]^=1;
        if(mode==12)output[63*6+2]^=1;
        if(mode==10)for(unsigned lane=1;lane<64;++lane)std::memcpy(output+6*lane,output,24);
        const auto *end=words+length-8;
        require(end[-2]==0xc0004600 && end[-1]==0x407 && end[0]==0xc0064900 && end[5]==1);
        auto *marker=static_cast<uint32_t *>(translate(uint64_t(end[3]) | uint64_t(end[4])<<32,4));
        require(*marker==0);if(!failure)*marker=1;
        return 0;
    }
    int32_t kick() override{return failure==2?-20:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{require(failure==3 && us==100);++polls;clock+=uint64_t(us)*1000;}
};
Model *quarantined_model=nullptr;
ps5hip::SubmissionQueue *quarantined_queue=nullptr;
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
    bool ok=seconds==1 && quarantined_model && quarantined_queue && quarantined_queue->poisoned() &&
        quarantined_model->submits==1 && allocations.size()==5 && attempted_owned_close==0 &&
        quarantined_model->polls==(quarantined_model->failure==3?20000u:0u);
    for(const auto &[id,a]:allocations){(void)id;ok=ok && a.gpu_uses==1;}
    ::_exit(ok?0:7);
}
int main(int argc,char **argv) {
    try {
        require(argc==2 || argc==3);
        std::ifstream file(argv[1],std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});
        const auto object=ps5hip::inspect_code_object(bytes.data(),bytes.size());
        const auto image=ps5hip::materialize_code_object(bytes.data(),bytes.size());
        for(unsigned mode=0;mode<=20;++mode) {
            Model model(object,image,mode);ps5hip::SubmissionQueue queue(model);
            FILE *log=std::tmpfile();require(log!=nullptr);std::fflush(stdout);
            int saved=::dup(STDOUT_FILENO);require(saved>=0 && ::dup2(::fileno(log),STDOUT_FILENO)>=0);
            bool rejected=false;
            try{run_experiment(queue,object,image);}catch(const ps5hip::ObjectError &){rejected=true;}
            std::fflush(stdout);require(::dup2(saved,STDOUT_FILENO)>=0);::close(saved);
            std::rewind(log);char text[16384]{};size_t n=std::fread(text,1,sizeof(text),log);std::fclose(log);
            std::string output(text,n);
            require(rejected==(mode>=2 && mode<=18) && !queue.poisoned() && allocations.empty() && model.submits==1);
            require((output.find("PASS kernel=private_aperture exact_values=128 raw_words=256")!=std::string::npos)==!rejected);
            require((output.find("within_wave=consistent")!=std::string::npos)==!rejected);
            require(output.find("PS5HIP APERTURE lane=63 ")!=std::string::npos);
            if(mode==19)require(output.find("base=0x0000000000000000 limit=0x0000000000000000")!=std::string::npos);
            if(mode==20)require(output.find("base=0xa5a5a5a5a5a5a5a5 limit=0xa5a5a5a5a5a5a5a5")!=std::string::npos);
            if(mode<2) {
                require(output.find("base=0x81234567ffffffff limit=0x1234567800000000")!=std::string::npos);
                require(output.find(mode==1?"across_waves=different":"across_waves=equal")!=std::string::npos);
                if(argc==3) {
                    std::ofstream saved(std::string(argv[2])+(mode==0?"/equal.log":"/different.log"));
                    saved << output;
                    require(bool(saved));
                }
            }
        }
        for(unsigned failure=1;failure<=5;++failure) {
            FILE *log=std::tmpfile();require(log!=nullptr);std::fflush(nullptr);
            pid_t child=::fork();
            if(child==0) {
                if(::dup2(::fileno(log),STDOUT_FILENO)<0 || ::dup2(::fileno(log),STDERR_FILENO)<0)::_exit(2);
                Model model(object,image,0,failure);ps5hip::SubmissionQueue queue(model);
                quarantined_model=&model;quarantined_queue=&queue;run_guarded_experiment(queue,object,image);::_exit(3);
            }
            int status=0;require(child>0 && ::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
            std::rewind(log);char text[2048]{};size_t n=std::fread(text,1,sizeof(text),log);std::fclose(log);
            std::string output(text,n);
            require(output.find("PS5HIP QUARANTINED")!=std::string::npos && output.find("PS5HIP PASS")==std::string::npos);
            require(output.find("PS5HIP APERTURE lane=")==std::string::npos);
        }
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
