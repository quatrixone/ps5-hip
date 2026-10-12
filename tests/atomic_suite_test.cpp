// Host routing/ownership model, not ISA emulation or hardware proof.
// The callback supplies valid and deliberately invalid atomic observations.
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
void require(bool value) { if (!value) throw ps5hip::ObjectError("atomic model assertion"); }
uint64_t address(int64_t id) { return 0x200000000ull + uint64_t(id) * 0x100000; }
void *translate(uint64_t gpu, size_t bytes) {
    for (const auto &[id, a] : allocations)
        if (gpu >= address(id) && gpu - address(id) <= a.bytes && bytes <= a.bytes - (gpu - address(id)))
            return static_cast<char *>(a.cpu) + gpu - address(id);
    throw ps5hip::ObjectError("atomic model address outside live allocation");
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
    const ps5hip::CodeObject &object;
    const std::vector<uint8_t> &image;
    unsigned bad_round, mode, failure, submits=0, polls=0;
    uint64_t clock=0;
    int64_t previous_max=0;
    std::set<int64_t> current_ids;
    Model(const ps5hip::CodeObject &o,const std::vector<uint8_t> &i,unsigned round,unsigned m,unsigned f=0)
        : object(o),image(i),bad_round(round),mode(m),failure(f) {}
    void flush(void *, size_t) override {}
    template<class T> void observations(T *counter,T *output,uint32_t *offsets,unsigned index) {
        constexpr unsigned n=16448;
        const bool cas=index==2 || index==3;
        const T seed=T(T(~T(0))-T(4095));
        require(*counter==(cas?T(0):seed));
        const auto *raw=reinterpret_cast<const uint8_t *>(output);
        for(size_t i=0;i<n*sizeof(T);++i)require(raw[i]==0xa5);
        if(offsets)for(unsigned i=0;i<n;++i)require(offsets[i]==0);
        if(cas) {
            // First and last winners are both legal; no scheduler-order assumption.
            unsigned winner=mode==1?n-1:0;
            T final=T((sizeof(T)==8?0x13579bdf00000000ull:0ull) | uint64_t(winner+1));
            *counter=final;
            for(unsigned i=0;i<n;++i)output[i]=i==winner?T(0):final;
        } else {
            *counter=T(seed+T(n));
            for(unsigned i=0;i<n;++i)output[i]=T(seed+T(mode==1?(i+417)%n:n-1-i));
        }
        if(submits!=bad_round)return;
        switch(mode) {
        case 2:*counter^=1;break;
        case 3:output[1]=output[0];break;
        case 4:std::memset(output+n-1,0xa5,sizeof(T));break;
        case 5:
            if constexpr(sizeof(T)==8) {
                *counter=T(uint32_t(*counter));
                for(unsigned i=0;i<n;++i)output[i]=T(uint32_t(output[i]));
            }
            break;
        case 6:output[17]=T(42);break;
        case 7:output[1]=0;break;
        case 8:std::swap(output[0],output[1]);break;
        case 9:corrupt_guard(counter,0);break;
        case 10:reinterpret_cast<uint8_t *>(counter)[sizeof(T)]^=1;break;
        case 11:corrupt_guard(counter,1);break;
        case 12:corrupt_guard(output,0);break;
        case 13:reinterpret_cast<uint8_t *>(output)[n*sizeof(T)]^=1;break;
        case 14:corrupt_guard(output,1);break;
        case 15:require(offsets);offsets[n-1]=1;break;
        case 16:require(offsets);corrupt_guard(offsets,0);break;
        case 17:require(offsets);offsets[n]^=1;break;
        case 18:require(offsets);corrupt_guard(offsets,1);break;
        case 19:std::memset(output,0xa5,n*sizeof(T));break;
        default:break;
        }
    }
    int32_t submit(uint32_t *words,uint32_t length) override {
        require(submits<6);const unsigned index=submits++;
        require(allocations.size()==(index>=4?7:6));
        current_ids.clear();
        for(const auto &[id,a]:allocations){require(id>previous_max && a.gpu_uses==1);current_ids.insert(id);}
        previous_max=*current_ids.rbegin();
        std::map<uint32_t,uint32_t> registers;size_t pos=10;
        while(pos<length && ((words[pos]>>8)&255)==0x76) {
            unsigned n=(words[pos]>>16)&0x3fff;require(n && pos+n+2<=length);
            for(unsigned i=0;i<n;++i)registers[0xb000+words[pos+1]*4+i*4]=words[pos+2+i];
            pos+=n+2;
        }
        require(pos+15==length && words[pos]==0xc0031502 && words[pos+1]==257 && words[pos+2]==1 && words[pos+3]==1 && words[pos+4]==0x8005);
        require(registers.at(0xb81c)==64 && registers.at(0xb820)==1 && registers.at(0xb824)==1);
        const auto k=std::find_if(object.kernels.begin(),object.kernels.end(),[&](const auto &v){return v.name==atomics::kernels[index];});
        require(k!=object.kernels.end() && k->kernarg_size==(index>=4?24:16) && k->private_bytes==0 && k->group_bytes==0 && k->properties==0x409 && k->rsrc2==0x8c);
        const uint64_t entry=uint64_t(registers.at(0xb830))<<8 | uint64_t(registers.at(0xb834))<<40;
        require(entry>=k->entry_address && std::memcmp(translate(entry-k->entry_address,image.size()),image.data(),image.size())==0);
        require(registers.at(0xb848)==k->rsrc1 && registers.at(0xb84c)==k->rsrc2);
        for(unsigned i=0;i<4;++i)require(registers.at(0xb900+4*i)==0);
        const uint64_t arg_address=registers.at(0xb910) | uint64_t(registers.at(0xb914))<<32;
        uint64_t args[3]{};std::memcpy(args,translate(arg_address,k->kernarg_size),k->kernarg_size);
        uint32_t *offsets=index>=4?static_cast<uint32_t *>(translate(args[2],16448*4)):nullptr;
        if(submits==bad_round && failure==1)return -10;
        if(submits==bad_round && failure==4)throw ps5hip::ObjectError("atomic callback exception");
        if(submits==bad_round && failure==5)throw 7;
        if(index%2)observations(static_cast<uint64_t *>(translate(args[0],8)),static_cast<uint64_t *>(translate(args[1],16448*8)),offsets,index);
        else observations(static_cast<uint32_t *>(translate(args[0],4)),static_cast<uint32_t *>(translate(args[1],16448*4)),offsets,index);
        const auto *end=words+length-8;
        require(end[-2]==0xc0004600 && end[-1]==0x407 && end[0]==0xc0064900 && end[5]==1);
        auto *marker=static_cast<uint32_t *>(translate(uint64_t(end[3])|uint64_t(end[4])<<32,4));require(*marker==0);
        if(!(submits==bad_round && failure))*marker=1;
        return 0;
    }
    int32_t kick() override{return submits==bad_round && failure==2?-20:0;}
    uint64_t now_ns() override{return clock;}
    void sleep_us(unsigned us) override{require(failure==3 && submits==bad_round && us==100);++polls;clock+=uint64_t(us)*1000;}
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
    auto *m=quarantined_model;
    bool ok=seconds==1 && m && quarantined_queue && quarantined_queue->poisoned() &&
        m->submits==m->bad_round && allocations.size()==(m->bad_round>=5?7:6) && attempted_owned_close==0 &&
        m->polls==(m->failure==3?20000u:0u);
    for(const auto &[id,a]:allocations)ok=ok && m->current_ids.count(id)==1 && a.gpu_uses==1;
    ::_exit(ok?0:7);
}
namespace {
struct Capture {
    FILE *file=std::tmpfile();int saved;
    Capture(){require(file!=nullptr);std::fflush(nullptr);saved=::dup(STDOUT_FILENO);require(saved>=0 && ::dup2(::fileno(file),STDOUT_FILENO)>=0);}
    std::string finish(){std::fflush(stdout);require(::dup2(saved,STDOUT_FILENO)>=0);::close(saved);std::rewind(file);char text[16384]{};size_t n=std::fread(text,1,sizeof(text),file);std::fclose(file);return {text,n};}
};
unsigned pass_count(const std::string &s){unsigned n=0;size_t p=0;while((p=s.find("PS5HIP PASS kernel=atomic_",p))!=std::string::npos){++n;++p;}return n;}
}
int main(int argc,char **argv) {
 try {
    require(argc==2);std::ifstream file(argv[1],std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});
    const auto object=ps5hip::inspect_code_object(bytes.data(),bytes.size());
    const auto image=ps5hip::materialize_code_object(bytes.data(),bytes.size());require(object.kernels.size()==6);
    unsigned rejected_cases=0;
    for(unsigned round=1;round<=6;++round)for(unsigned mode=0;mode<=19;++mode) {
        bool cas=round==3 || round==4;
        if((mode==5 && round%2) || ((mode==6 || mode==7 || mode==8) && !cas) || (mode>=15 && mode<=18 && round<5))continue;
        // Swapping ticket observations is legal; CAS winner identity is not.
        Model model(object,image,round,mode);ps5hip::SubmissionQueue queue(model);Capture capture;bool rejected=false;
        try{run_guarded_experiment(queue,object,image);}catch(const ps5hip::ObjectError &){rejected=true;}
        const auto log=capture.finish();require(rejected==(mode>=2));
        require(!queue.poisoned() && allocations.empty() && attempted_owned_close==0);
        require(model.submits==(rejected?round:6));require(pass_count(log)==(rejected?round-1:7));
        require((log.find("kernel=atomic_suite dispatches=6")!=std::string::npos)==!rejected);
        if(rejected)++rejected_cases;
    }
    for(unsigned round=1;round<=6;++round)for(unsigned failure=1;failure<=5;++failure) {
        FILE *log=std::tmpfile();require(log!=nullptr);std::fflush(nullptr);pid_t child=::fork();
        if(child==0) {
            if(::dup2(::fileno(log),STDOUT_FILENO)<0 || ::dup2(::fileno(log),STDERR_FILENO)<0)::_exit(2);
            Model model(object,image,round,0,failure);ps5hip::SubmissionQueue queue(model);
            quarantined_model=&model;quarantined_queue=&queue;run_guarded_experiment(queue,object,image);::_exit(3);
        }
        int status=0;require(child>0 && ::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
        std::rewind(log);char text[8192]{};size_t n=std::fread(text,1,sizeof(text),log);std::fclose(log);std::string output(text,n);
        require(pass_count(output)==round-1 && output.find("PS5HIP QUARANTINED")!=std::string::npos && output.find("kernel=atomic_suite dispatches=6")==std::string::npos);
    }
    std::printf("atomic host model: 12 valid suites, %u corruption cases, 30 retained-ownership failures; no GPU execution\n",rejected_cases);
 }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
