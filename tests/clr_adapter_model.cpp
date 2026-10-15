// Actual ROCclr/PS5 adapter, modeled direct memory and packet transport only.
// This model's arithmetic is NOT GPU execution or an ISA emulator.
#include "ps5-device.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/context.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "device/device.hpp"
#include "utils/flags.hpp"
#include <ps5platform/kernel.h>
#include <sys/mman.h>
#include <fstream>
#include <map>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
static void checked(bool ok,int line){if(!ok){std::fprintf(stderr,"CLR_ADAPTER_MODEL assertion line=%d\n",line);std::abort();}}
#define check(x) checked(bool(x),__LINE__)
struct Allocation{void* p;size_t n;};static std::map<int64_t,Allocation> allocations;
static int64_t serial=1;static uintptr_t next_address=0x200040000;
static unsigned releases=0,submits=0,backend_calls=0;static const char* mode="normal";static void* code=nullptr;
static std::mutex mutex;static std::condition_variable changed;static bool entered=false,resume=false;
static constexpr int model_map_flags=MAP_PRIVATE|MAP_ANONYMOUS|
#if defined(__clang__) && __has_feature(address_sanitizer)
 // ASan reserves this fixed native shader window even with gap protection off.
 // Replace only our bounded model pages; production never uses this provider.
 MAP_FIXED;
#else
 MAP_FIXED_NOREPLACE;
#endif
static unsigned joined_queues=0;
class CheckedQueue final:public amd::HostQueue {
 bool terminate()override{if(!amd::HostQueue::terminate()){std::fprintf(stderr,"CLR_MODEL_JOIN_FAILED ownership retained\n");std::_Exit(70);}++joined_queues;return true;}
public:using amd::HostQueue::HostQueue;
};
int ps5hip_clr_model_destroy(pthread_mutex_t* gate)noexcept{return !std::strcmp(mode,"destroy")?EBUSY:pthread_mutex_destroy(gate);}
static void live(uint64_t p,size_t n){for(const auto&a:allocations){auto b=uint64_t(a.second.p);if(p>=b&&p-b<=a.second.n&&n<=a.second.n-(p-b))return;}check(false);}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t n,size_t,int,int64_t* p){if(!std::strcmp(mode,"svm-alloc-fail"))return -12;*p=serial++;allocations.emplace(*p,Allocation{nullptr,n});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void**p,size_t n,int,int,int64_t id,size_t align){next_address=(next_address+align-1)&~(align-1);check(next_address>=0x200040000&&next_address<0x300000000&&n<=0x300000000-next_address);void*v=mmap(reinterpret_cast<void*>(next_address),n,PROT_READ|PROT_WRITE,model_map_flags,-1,0);if(v==MAP_FAILED){std::fprintf(stderr,"MODEL_MAP_FAILED address=%llx errno=%d\n",static_cast<unsigned long long>(next_address),errno);return -1;}next_address+=n;allocations.at(id).p=v;*p=v;return 0;}
extern "C" int32_t sceKernelMunmap(void* p,size_t n){if((!std::strcmp(mode,"cleanup")&&p==code)||!std::strcmp(mode,"svm-free-fail"))return -1;return munmap(p,n);}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t id,size_t){allocations.erase(id);++releases;return 0;}
class Transport final:public ps5hip::SubmissionBackend{
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{if(!std::strcmp(mode,"postflush")&&submits)throw std::runtime_error("injected postflush");}
 int32_t submit(uint32_t*w,uint32_t count)override{
  ++submits;
  {std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();check(changed.wait_for(lock,std::chrono::seconds(5),[]{return resume;}));}
  std::map<unsigned,std::vector<unsigned>> registers;
  for(size_t at=0;at<count;){auto h=w[at];size_t n=((h>>16)&0x3fff)+1;check(h>>30==3&&n<count-at);if((h&0xffff)==0x7600)registers.emplace(w[at+1],std::vector<unsigned>(w+at+2,w+at+n+1));at+=n+1;}
  auto&s=registers.at(0x240);auto&p=registers.at(0x20c);auto arg=uint64_t(s.at(4))|(uint64_t(s.at(5))<<32);live(arg,16);
  uint64_t output;unsigned n,salt;std::memcpy(&output,reinterpret_cast<void*>(arg),8);std::memcpy(&n,reinterpret_cast<void*>(arg+8),4);std::memcpy(&salt,reinterpret_cast<void*>(arg+12),4);live(output,n*4);
  auto entry=(uint64_t(p.at(0))<<8)|(uint64_t(p.at(1))<<40);for(const auto&a:allocations)if(entry>=uint64_t(a.second.p)&&entry<uint64_t(a.second.p)+a.second.n)code=a.second.p;
#if defined(PS5HIP_MODEL_PAIRED_SALT)
  if(!std::strncmp(mode,"replay",6))check(code&&n==73&&submits<=(!std::strcmp(mode,"replay-combined")?84u:80u)&&salt>=19&&(salt-19)/101<20&&((salt-19)%101==0||(salt-19)%101==37));
  else check(code&&n==73&&submits<=4&&salt==19+((submits-1)/2)*24);
#elif defined(PS5HIP_MODEL_CHANGING_SALT)
  check(code&&n==73&&submits<=2&&salt==19+(submits-1)*24);
#else
  check(code&&n==73&&salt==19);
#endif
  if(!std::strcmp(mode,"submit"))return -7;if(!std::strcmp(mode,"throw"))throw std::runtime_error("injected submit");
  if(std::strcmp(mode,"noop"))for(unsigned i=0;i<n;++i)reinterpret_cast<unsigned*>(output)[i]=i*17+salt+unsigned(!std::strcmp(mode,"corrupt"));
  const auto*end=w+count-8;check(end[0]==0xc0064900);auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);
  if(std::strcmp(mode,"timeout"))*reinterpret_cast<volatile unsigned*>(marker)=1;return 0;
 }
 int32_t kick()override{return !std::strcmp(mode,"kick")?-8:0;}
 uint64_t now_ns()override{return clock;}
 void sleep_us(unsigned n)override{clock+=uint64_t(n)*1000;}
};
static std::unique_ptr<ps5hip::SubmissionBackend> backend(){++backend_calls;return std::make_unique<Transport>();}
static void CL_CALLBACK callback(cl_event,int32_t status,void* opaque){
 check(status==CL_COMPLETE);auto& ctx=*static_cast<amd::Context*>(opaque);
 auto* buffer=new(ctx)amd::Buffer(ctx,CL_MEM_READ_WRITE,16);check(buffer->create());
 check(buffer->getDeviceMemory(*ctx.devices()[0]));buffer->release();
}
int main(int argc,char**argv){
 check(argc==3);mode=argv[2];amd::IS_HIP=true;AMD_DIRECT_DISPATCH=false;
 ps5hip::clr::Observation observed;
 std::unique_ptr<amd::Device> device(ps5hip::clr::create_device(backend,observed));check(bool(device));
 amd::Context::Info info{};check(amd::Context::checkProperties(nullptr,&info)==CL_SUCCESS);
 auto* context=new amd::Context({device.get()},info);check(context->create(nullptr)==CL_SUCCESS);
 if(!std::strncmp(mode,"svm-",4)&&std::strcmp(mode,"svm-launch")){
  check(context->svmDevices().size()==1);
  for(auto flags:{cl_svm_mem_flags(CL_MEM_SVM_FINE_GRAIN_BUFFER),cl_svm_mem_flags(CL_MEM_SVM_ATOMICS),cl_svm_mem_flags(CL_MEM_USE_HOST_PTR)})
   check(!amd::SvmBuffer::malloc(*context,flags,256,65536));
  check(!amd::SvmBuffer::malloc(*context,0,0,65536));
  check(!amd::SvmBuffer::malloc(*context,0,64u*1024u*1024u+1,65536));
  check(!amd::SvmBuffer::malloc(*context,0,256,3));
  check(!amd::SvmBuffer::malloc(*context,0,256,131072));
  check(allocations.empty()&&observed.allocations==0);
  auto* pointer=amd::SvmBuffer::malloc(*context,0,256,65536);
  if(!std::strcmp(mode,"svm-alloc-fail")){check(!pointer&&allocations.empty()&&!observed.terminal);context->release();device.reset();check(observed.gate_destroyed);}
  else {
   check(pointer&&reinterpret_cast<uintptr_t>(pointer)%65536==0&&amd::SvmBuffer::malloced(pointer));
   auto* owner=amd::MemObjMap::FindMemObj(pointer);check(owner&&owner->getSvmPtr()==pointer&&owner->getSize()==256);
   check(!device->svmAlloc(*context,256,65536,0,pointer));
   device->svmFree(static_cast<char*>(pointer)+1);check(observed.frees==0&&amd::MemObjMap::FindMemObj(pointer)==owner);
   owner->retain();amd::SvmBuffer::free(*context,pointer);check(observed.frees==0&&amd::SvmBuffer::malloced(pointer));owner->release();
   amd::SvmBuffer::free(*context,pointer);
   if(!std::strcmp(mode,"svm-free-fail")){
    check(observed.terminal&&observed.frees==0&&allocations.size()==1&&releases==0);
    check(amd::SvmBuffer::malloced(pointer)&&amd::MemObjMap::FindMemObj(pointer)==owner);
    check(!amd::SvmBuffer::malloc(*context,0,128,65536));device.release();
   }else{
    check(!amd::SvmBuffer::malloced(pointer)&&!amd::MemObjMap::FindMemObj(pointer));
    check(observed.frees==1&&allocations.empty());amd::SvmBuffer::free(*context,pointer);check(observed.frees==1);
    context->release();device.reset();check(observed.gate_destroyed);
   }
  }
  std::printf("CLR_PS5_ADAPTER_MODEL_PASS mode=%s gpu_execution=false actual_upstream_svm=true\n",mode);return 0;
 }
 std::ifstream in(argv[1],std::ios::binary);std::vector<char> binary((std::istreambuf_iterator<char>(in)),{});check(!binary.empty());
 auto* program=new amd::Program(*context);
 if(!std::strcmp(mode,"truncated")){
  for(size_t length: {size_t(0),size_t(1),size_t(2),size_t(3),size_t(63),size_t(64)}){
   std::unique_ptr<char[]> tiny(new char[length?length:1]);std::memcpy(tiny.get(),binary.data(),length);
   check(program->addDeviceProgram(*device,tiny.get(),length,true)==CL_INVALID_BINARY);
  }
  check(program->addDeviceProgram(*device,nullptr,0,true)==CL_INVALID_BINARY);
  check(!backend_calls&&!submits&&allocations.empty());program->release();context->release();device.reset();
  std::puts("CLR_PS5_ADAPTER_MODEL_PASS mode=truncated gpu_execution=false rejected=true backend_calls=0");return 0;
 }
 if(!std::strcmp(mode,"badbinary")){
  binary[18]=0;int result=program->addDeviceProgram(*device,binary.data(),binary.size(),true);
  check(result!=CL_SUCCESS||!program->load());check(!backend_calls&&!submits&&allocations.empty());
  program->release();context->release();device.reset();std::puts("CLR_PS5_ADAPTER_MODEL_PASS mode=badbinary gpu_execution=false rejected=true backend_calls=0");return 0;
 }
 check(program->addDeviceProgram(*device,binary.data(),binary.size(),true)==CL_SUCCESS);
 check(program->load());auto* symbol=program->findSymbol("thread_words");check(symbol);
 check(program->load()&&program->findSymbol("thread_words")==symbol);
 check(program->addDeviceProgram(*device,binary.data(),binary.size(),true)==CL_INVALID_OPERATION);
 auto* kernel=new amd::Kernel(*program,*symbol,"thread_words");program->release();
 auto* a=new CheckedQueue(*context,*device,0);auto*b=new CheckedQueue(*context,*device,0);check(a->create()&&b->create());
 constexpr size_t words=81;unsigned input[words],output[words];for(unsigned i=0;i<words;++i){input[i]=0xa5000000u+i;output[i]=0xdeadbeef;}
 const bool svmLaunch=!std::strcmp(mode,"svm-launch");void*svmX=nullptr;void*svmY=nullptr;
 amd::Buffer*x=nullptr;amd::Buffer*y=nullptr;
 if(svmLaunch){
  svmX=amd::SvmBuffer::malloc(*context,0,sizeof(input),65536);svmY=amd::SvmBuffer::malloc(*context,0,sizeof(input),65536);check(svmX&&svmY);
  x=dynamic_cast<amd::Buffer*>(amd::MemObjMap::FindMemObj(svmX));y=dynamic_cast<amd::Buffer*>(amd::MemObjMap::FindMemObj(svmY));check(x&&y);
 }else{x=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(input));y=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(input));check(x->create()&&y->create());}
 if(!std::strcmp(mode,"unsupported")){
  unsigned pattern=7;auto* fill=new amd::FillMemoryCommand(*a,CL_COMMAND_FILL_IMAGE,{},*x,&pattern,4,amd::Coord3D(0),amd::Coord3D(sizeof(input)),amd::Coord3D(sizeof(input)));
  auto* denied=new amd::ReadMemoryCommand(*b,CL_COMMAND_READ_BUFFER,{fill},*x,amd::Coord3D(0),amd::Coord3D(sizeof(output)),output);
  check(fill->validateMemory()&&denied->validateMemory());fill->enqueue();denied->enqueue();check(!denied->awaitCompletion());a->finish(true);b->finish(true);
  check(fill->status()==CL_INVALID_OPERATION&&denied->status()==CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
  fill->release();denied->release();x->release();y->release();kernel->release();a->release();b->release();context->release();device.reset();
  std::fprintf(stderr,"UNSUPPORTED_COUNTS backend=%u submits=%u terminal=%u retained=%zu frees=%u joined=%u\n",backend_calls,submits,unsigned(observed.terminal.load()),allocations.size(),observed.frees.load(),joined_queues);
  check(!backend_calls&&!submits&&!observed.terminal&&allocations.empty()&&observed.frees==2);
  std::puts("CLR_PS5_ADAPTER_MODEL_PASS mode=unsupported gpu_execution=false rejected=true backend_calls=0");return 0;
 }
 cl_mem handle=as_cl(static_cast<amd::Memory*>(y));unsigned count=73,salt=19;
 kernel->parameters().set(0,sizeof(handle),&handle);kernel->parameters().set(1,4,&count);kernel->parameters().set(2,4,&salt);
 auto*write=new amd::WriteMemoryCommand(*a,CL_COMMAND_WRITE_BUFFER,{},*x,amd::Coord3D(0),amd::Coord3D(sizeof(input)),input);
 auto*copy=new amd::CopyMemoryCommand(*b,CL_COMMAND_COPY_BUFFER,{write},*x,*y,amd::Coord3D(0),amd::Coord3D(0),amd::Coord3D(sizeof(input)));
 size_t offset=0,global=128,local=64;amd::NDRangeContainer shape(1,&offset,&global,&local);
 auto*launch=new amd::NDRangeKernelCommand(*a,{copy},*kernel,shape);check(launch->captureAndValidate()==CL_SUCCESS);
 if(!std::strcmp(mode,"callback"))check(launch->setCallback(CL_COMPLETE,callback,context));
 auto*read=new amd::ReadMemoryCommand(*b,CL_COMMAND_READ_BUFFER,{launch},*y,amd::Coord3D(0),amd::Coord3D(sizeof(output)),output);
 check(write->validateMemory()&&copy->validateMemory()&&read->validateMemory());
 salt=999;kernel->parameters().set(2,4,&salt);kernel->release();
 if(svmLaunch){amd::SvmBuffer::free(*context,svmX);amd::SvmBuffer::free(*context,svmY);check(amd::SvmBuffer::malloced(svmX)&&amd::SvmBuffer::malloced(svmY));}
 else{x->release();y->release();}
 check(observed.allocations==2&&observed.frees==0);
 write->enqueue();copy->enqueue();launch->enqueue();read->enqueue();
 {std::unique_lock<std::mutex> lock(mutex);check(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}));check(observed.frees==0&&observed.launches==0);for(auto v:output)check(v==0xdeadbeef);resume=true;changed.notify_all();}
 const bool success=svmLaunch||!std::strcmp(mode,"normal")||!std::strcmp(mode,"callback")||!std::strcmp(mode,"noop")||!std::strcmp(mode,"corrupt")||!std::strcmp(mode,"destroy");check(read->awaitCompletion()==success);a->finish(true);b->finish(true);
 if(success){for(unsigned i=0;i<words;++i)if(output[i]!=(i<count?i*17+19:input[i])){std::fprintf(stderr,"CLR_OUTPUT_ORACLE_REJECT mode=%s index=%u\n",mode,i);std::abort();}check(observed.launches==1);}
 else {check(launch->status()<CL_COMPLETE&&read->status()==CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);for(auto v:output)check(v==0xdeadbeef);check(observed.terminal);}
 write->release();copy->release();launch->release();read->release();a->release();b->release();check(joined_queues==2);
 if(svmLaunch){check(observed.frees==0);amd::SvmBuffer::free(*context,svmX);amd::SvmBuffer::free(*context,svmY);check(!amd::SvmBuffer::malloced(svmX)&&!amd::SvmBuffer::malloced(svmY));check(!amd::MemObjMap::FindMemObj(svmX)&&!amd::MemObjMap::FindMemObj(svmY));}
 context->release();device.reset();
 if(success){unsigned expected=!std::strcmp(mode,"callback")?3:2;check(observed.allocations==expected&&observed.frees==expected&&allocations.empty());check(observed.gate_destroyed==bool(std::strcmp(mode,"destroy")));if(!std::strcmp(mode,"destroy"))check(observed.terminal);}
 else {check(observed.frees==0);const bool uncertain=std::strcmp(mode,"cleanup");check(observed.uncertain==uncertain);check(allocations.size()==(uncertain?6:3)&&releases==(uncertain?0:3));}
 check(observed.queues==2&&observed.retired_queues==2);
 std::printf("CLR_PS5_ADAPTER_MODEL_PASS mode=%s gpu_execution=false dispatches=%u allocations=%u frees=%u retained=%zu releases=%u uncertain=%u\n",mode,submits,observed.allocations.load(),observed.frees.load(),allocations.size(),releases,unsigned(observed.uncertain.load()));
}
