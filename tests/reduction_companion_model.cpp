// Runs the unmodified sample through the real HIP runtime with synthetic transport.
// The direct selected-index sum is a CPU oracle, not GPU ISA emulation.
#include "ps5hip/hip_runtime_control.h"
#include <ps5platform/kernel.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iterator>
#include "ps5hip/code_object.h"
namespace {
struct Allocation {void* cpu=nullptr;size_t bytes=0,alignment=0;};
std::map<int64_t,Allocation> allocations;
int64_t next_id=1;uint64_t next_address=0x200040000;
unsigned backends=0,submits=0,releases=0;std::string mode;
std::vector<uint8_t> image,fixture;uint64_t entry_offset=0;int previous[16]{};
void check(bool yes){if(!yes)throw std::runtime_error("reduction transport invariant");}
void containing(uint64_t p,size_t n){for(auto &e:allocations){auto b=reinterpret_cast<uint64_t>(e.second.cpu);if(b && p>=b && p-b<=e.second.bytes && n<=e.second.bytes-(p-b))return;}throw std::runtime_error("unowned reduction span");}
void dispose(Allocation&a){if(a.cpu){check(munmap(a.cpu,a.bytes)==0);a.cpu=nullptr;}}
struct Audit {~Audit(){std::printf("REDUCTION_COMPANION_MODEL_FINAL live=%zu backends=%u submits=%u releases=%u finalized=%u\n",allocations.size(),backends,submits,releases,unsigned(ps5hip_runtime_finalized()));for(auto&e:allocations)dispose(e.second);}}audit;
uint64_t read(const uint8_t*p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
class Backend final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{}
 int32_t submit(uint32_t*w,uint32_t count)override{
  ++submits;check(submits<=3);std::map<uint32_t,std::vector<uint32_t>>regs;bool dispatch=false;
  for(size_t at=0;at<count;){auto h=w[at];check((h>>30)==3);size_t n=((h>>16)&0x3fff)+1;check(n<count-at);
   if((h&0xffff)==0x7600){check(n>=2&&!regs.count(w[at+1]));regs[w[at+1]]={w+at+2,w+at+n+1};}
   if(h==0xc0031502){check(!dispatch&&w[at+1]==16&&w[at+2]==1&&w[at+3]==1&&w[at+4]==0x8005);dispatch=true;}at+=n+1;
  }
  check(dispatch&&regs.at(0x207)==std::vector<uint32_t>({1024,1,1})&&regs.at(0x215)==std::vector<uint32_t>({0x400000}));
  check(regs.at(0x212)==std::vector<uint32_t>({0x60af0080,0x4008c}));auto&s=regs.at(0x240);check(s.size()==6);for(unsigned i=0;i<4;++i)check(!s[i]);
  const auto &program=regs.at(0x20c);check(program.size()==2);const uint64_t entry=((uint64_t(program[0])<<8)|(uint64_t(program[1])<<40))+(mode=="entry"?256:0);check(entry>=entry_offset);const uint64_t base=entry-entry_offset;
  bool image_owned=false;for(const auto&e:allocations)if(reinterpret_cast<uint64_t>(e.second.cpu)==base&&e.second.bytes>=image.size())image_owned=true;
  if(mode=="code-image"&&image_owned)reinterpret_cast<uint8_t*>(base)[entry_offset]^=1;
  check(image_owned&&!image.empty()&&std::memcmp(reinterpret_cast<const void*>(base),image.data(),image.size())==0);
  auto a=uint64_t(s[4])|(uint64_t(s[5])<<32);containing(a,288);auto*k=reinterpret_cast<const uint8_t*>(a);
  const unsigned round=submits-1;const size_t logical=round==2?16377:16384;check(read(k+24,8)==logical&&read(k+44,2)==1024&&read(k+152,4)==4096);
  auto ip=read(k,8),mp=read(k+8,8),op=read(k+16,8);containing(ip,16384*4);containing(mp,512*8);containing(op,16*4);
  auto*input=reinterpret_cast<const int*>(ip);auto*masks=reinterpret_cast<const uint64_t*>(mp);auto*output=reinterpret_cast<int*>(op);
  constexpr size_t fixture_stride=16384*4+512*8;check(fixture.size()==3*fixture_stride);
  check(std::memcmp(input,fixture.data()+round*fixture_stride,16384*4)==0&&std::memcmp(masks,fixture.data()+round*fixture_stride+16384*4,512*8)==0);
  if(mode=="submit")return -7;
  if(mode!="missing")for(unsigned b=0;b<16;++b){int sum=0;for(unsigned i=b*1024;i<(b+1)*1024;++i)if((i<logical||mode=="tail")&&(((masks[i/32]|(mode=="highbits"?(masks[i/32]>>32):0))>>(i%32))&1))sum+=input[i];output[b]=sum;}
  if(mode=="stale"&&round)std::memcpy(output,previous,sizeof(previous));
  std::memcpy(previous,output,sizeof(previous));
  if(mode=="corrupt")output[0]^=1;
  if(mode=="input-guard")const_cast<int*>(input)[-1]^=1;
  if(mode=="mask-guard")const_cast<uint64_t*>(masks)[-1]^=1;
  if(mode=="output-guard")output[-1]^=1;
  auto*end=w+count-8;check(end[-2]==0xc0004600&&end[0]==0xc0064900&&end[1]==0x30c514);auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);containing(marker,4);*reinterpret_cast<volatile uint32_t*>(marker)=1;return 0;
 }
 int32_t kick()override{return 0;}uint64_t now_ns()override{return clock;}void sleep_us(unsigned us)override{clock+=uint64_t(us)*1000;}
};
}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;return std::make_unique<Backend>();}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t bytes,size_t alignment,int type,int64_t*physical){check(type==12);*physical=next_id++;allocations.emplace(*physical,Allocation{nullptr,bytes,alignment});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void**pointer,size_t bytes,int protection,int flags,int64_t physical,size_t alignment){
 check(protection==0x33&&flags==0);auto&a=allocations.at(physical);check(a.bytes==bytes&&a.alignment==alignment);next_address=(next_address+alignment-1)&~(uint64_t(alignment)-1);check(bytes<=0x300000000-next_address);
 int fixed=MAP_FIXED_NOREPLACE;
#if __has_feature(address_sanitizer)
 const char*options=std::getenv("ASAN_OPTIONS");check(options&&std::strstr(options,"protect_shadow_gap=0"));fixed=MAP_FIXED;
#endif
 void*cpu=mmap(reinterpret_cast<void*>(next_address),bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|fixed,-1,0);if(cpu==MAP_FAILED)return -1;a.cpu=cpu;next_address+=bytes;std::memset(cpu,0xa5,bytes);*pointer=cpu;return 0;
}
extern "C" int32_t sceKernelMunmap(void*pointer,size_t bytes){for(auto&e:allocations)if(e.second.cpu==pointer){check(e.second.bytes==bytes);dispose(e.second);return 0;}return -1;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical,size_t bytes){check(allocations.at(physical).bytes==bytes);allocations.erase(physical);++releases;return 0;}
extern "C" int ps5hip_reduction_companion_main();
extern "C" int __wrap_main(int argc,char**argv){
 check(argc==4);mode=argv[1];check(mode=="normal"||mode=="corrupt"||mode=="missing"||mode=="stale"||mode=="highbits"||mode=="tail"||mode=="input-guard"||mode=="mask-guard"||mode=="output-guard"||mode=="submit"||mode=="code-image"||mode=="entry");
 std::ifstream codefile(argv[2],std::ios::binary),fixturefile(argv[3],std::ios::binary);check(bool(codefile)&&bool(fixturefile));
 const std::vector<uint8_t> code{std::istreambuf_iterator<char>(codefile),{}};fixture={std::istreambuf_iterator<char>(fixturefile),{}};
 auto object=ps5hip::inspect_code_object(code.data(),code.size());check(object.kernels.size()==1&&object.kernels[0].name=="_Z12block_reducePiPmS_m");entry_offset=object.kernels[0].entry_address;image=ps5hip::materialize_code_object(code.data(),code.size());
 const int result=ps5hip_reduction_companion_main();
 if(ps5hip_runtime_terminal()){
  check(result==30&&ps5hip_runtime_gpu_unresolved()&&allocations.size()==7&&releases==0&&submits==1);std::puts("REDUCTION_COMPANION_TERMINAL retained=7 releases=0 submits=1");std::fflush(nullptr);std::_Exit(30);
 }
 const unsigned expected=mode=="normal"||mode=="tail"?3:mode=="stale"?2:1;
 check(submits==expected&&ps5hip_runtime_ready()&&ps5hip_runtime_observation().completed==expected);
 check(result==(mode=="normal"?0:20));return result;
}
