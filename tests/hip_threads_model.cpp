// Real compiler stubs/runtime, synthetic packet transport. Ordering uses gates, never sleeps.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include <ps5platform/kernel.h>
#include <sys/mman.h>
#include <pthread.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <map>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <stdexcept>
extern "C" void threads_launch(unsigned*,unsigned,unsigned);
extern "C" void __real___hipUnregisterFatBinary(void**);
extern "C" void** __hipRegisterFatBinary(const void*);
extern "C" const unsigned char __ps5hip_wrappers_begin[];
namespace {
void check(bool ok){if(!ok){std::fprintf(stderr,"THREAD_MODEL assertion failed\n");std::abort();}}
std::atomic<unsigned> fault{0},locks{0},keys{0},sets{0},contexts{0},state_attempts{0};
std::mutex coordination;std::condition_variable changed;unsigned phase=0;
bool pause_submit=false,entered=false,resume=false,waiter_attempted=false;
thread_local bool waiter=false;
void step(unsigned n){std::lock_guard<std::mutex> l(coordination);phase=n;changed.notify_all();}
void until(unsigned n){std::unique_lock<std::mutex> l(coordination);changed.wait(l,[&]{return phase>=n;});}
bool consume(unsigned n){unsigned expected=n;return fault.compare_exchange_strong(expected,0);}
struct Allocation{void*cpu;size_t bytes;};std::map<int64_t,Allocation> allocations;
int64_t next_id=1;uint64_t address=0x200040000;unsigned submits=0,backends=0,releases=0;
const char*mode="";bool expect_final=true;void*code_pointer=nullptr;
uint64_t read(const uint8_t*p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
void live(uint64_t p,size_t n){for(auto&a:allocations){auto b=uint64_t(a.second.cpu);if(p>=b&&p-b<=a.second.bytes&&n<=a.second.bytes-(p-b))return;}check(false);}
class Backend final:public ps5hip::SubmissionBackend{
 uint64_t clock=0;
 public:
 void flush(void*,size_t)override{if(!std::strcmp(mode,"postflush")&&submits)throw std::runtime_error("flush");}
 int32_t submit(uint32_t*w,uint32_t count)override{
  ++submits;
  if(pause_submit){std::unique_lock<std::mutex> l(coordination);entered=true;changed.notify_all();changed.wait(l,[]{return resume;});}
  std::map<unsigned,std::vector<unsigned>> r;
  for(size_t at=0;at<count;){auto h=w[at];size_t n=((h>>16)&0x3fff)+1;check(h>>30==3&&n<count-at);if((h&0xffff)==0x7600)r.emplace(w[at+1],std::vector<unsigned>(w+at+2,w+at+n+1));at+=n+1;}
  auto&s=r.at(0x240);auto&p=r.at(0x20c);auto arg=uint64_t(s.at(4))|(uint64_t(s.at(5))<<32);live(arg,16);
  auto*k=reinterpret_cast<const uint8_t*>(arg);auto out=read(k,8);auto n=unsigned(read(k+8,4)),salt=unsigned(read(k+12,4));live(out,n*4);
  auto entry=(uint64_t(p.at(0))<<8)|(uint64_t(p.at(1))<<40);for(auto&a:allocations)if(entry>=uint64_t(a.second.cpu)&&entry<uint64_t(a.second.cpu)+a.second.bytes)code_pointer=a.second.cpu;
  if(!std::strcmp(mode,"submit"))return -7;if(!std::strcmp(mode,"throw"))throw std::runtime_error("submit");if(!std::strcmp(mode,"unknown"))throw 7;
  for(unsigned i=0;i<n;++i)reinterpret_cast<unsigned*>(out)[i]=i*17u+salt;
  const auto*end=w+count-8;check(end[0]==0xc0064900);auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);
  if(std::strcmp(mode,"timeout"))*reinterpret_cast<volatile unsigned*>(marker)=1;return 0;
 }
 int32_t kick()override{return !std::strcmp(mode,"kick")?-8:0;}
 uint64_t now_ns()override{return clock;}void sleep_us(unsigned us)override{clock+=uint64_t(us)*1000;}
};
void discovery_error(hipError_t expected){int value=713;
 check(hipInit(UINT_MAX)==expected&&hipSetDevice(-1)==expected&&hipGetDeviceCount(nullptr)==expected&&hipGetDevice(nullptr)==expected&&hipDeviceGetAttribute(nullptr,hipDeviceAttributeWarpSize,-1)==expected);
 check(hipInit(0)==expected&&hipSetDevice(0)==expected);
 check(hipGetDeviceCount(&value)==expected&&value==713);
 check(hipGetDevice(&value)==expected&&value==713);
 check(hipDeviceGetAttribute(&value,hipDeviceAttributeWarpSize,0)==expected&&value==713);
}
void discovery_ok(){int value=713;
 check(hipInit(0)==hipSuccess&&hipInit(0)==hipSuccess);
 check(hipGetDeviceCount(&value)==hipSuccess&&value==1);
 check(hipSetDevice(0)==hipSuccess&&hipGetDevice(&value)==hipSuccess&&value==0);
 check(hipDeviceGetAttribute(&value,hipDeviceAttributeWarpSize,0)==hipSuccess&&value==32);
}
void closed(){const auto a=allocations.size();const auto b=backends,s=submits,r=releases;void*p=nullptr;unsigned v=0;
 size_t bytes=713;void *base=reinterpret_cast<void*>(713);check(hipMemPtrGetInfo(nullptr,&bytes)==hipErrorDeinitialized&&bytes==713);check(hipMemGetAddressRange(&base,&bytes,nullptr)==hipErrorDeinitialized&&base==reinterpret_cast<void*>(713)&&bytes==713);
 discovery_error(hipErrorDeinitialized);
 check(hipMemset(nullptr,0,0)==hipErrorDeinitialized);
 check(hipMemcpy(nullptr,nullptr,0,hipMemcpyDeviceToDevice)==hipErrorDeinitialized);
 check(hipMalloc(&p,4)==hipErrorDeinitialized&&!p&&hipFree(nullptr)==hipErrorDeinitialized&&hipMemcpy(&v,&v,0,hipMemcpyHostToHost)==hipErrorDeinitialized&&hipDeviceSynchronize()==hipErrorDeinitialized&&hipPeekAtLastError()==hipErrorDeinitialized&&hipGetLastError()==hipErrorDeinitialized);
 check(a==allocations.size()&&b==backends&&s==submits&&r==releases);
}
void poison(){const auto a=allocations.size();const auto b=backends,s=submits,r=releases,c=contexts.load(),st=state_attempts.load();void*p=nullptr;check(hipMalloc(&p,0)==hipErrorUnknown&&!p&&hipDeviceSynchronize()==hipErrorUnknown&&hipFree(nullptr)==hipErrorUnknown);
 size_t bytes=713;void *base=reinterpret_cast<void*>(713);check(hipMemPtrGetInfo(nullptr,&bytes)==hipErrorUnknown&&bytes==713);check(hipMemGetAddressRange(&base,&bytes,nullptr)==hipErrorUnknown&&base==reinterpret_cast<void*>(713)&&bytes==713);
 discovery_error(hipErrorUnknown);
 check(hipMemset(nullptr,0,0)==hipErrorUnknown);
 check(hipMemcpy(nullptr,nullptr,0,hipMemcpyDeviceToDevice)==hipErrorUnknown);
 // Transport poison may create a virgin caller's error context. Infrastructure
 // and abandoned-lifecycle faults reject before any context or State access.
 const bool transport=!std::strcmp(mode,"submit")||!std::strcmp(mode,"kick")||!std::strcmp(mode,"timeout")||!std::strcmp(mode,"throw")||!std::strcmp(mode,"unknown")||!std::strcmp(mode,"postflush")||!std::strcmp(mode,"cleanup");
 check(a==allocations.size()&&b==backends&&s==submits&&r==releases&&(transport||contexts==c)&&state_attempts==st);}
struct Audit{~Audit(){check(ps5hip_runtime_finalized()==expect_final);std::printf("THREAD_MODEL_FINAL finalized=%u live=%zu\n",unsigned(ps5hip_runtime_finalized()),allocations.size());}}audit;
}
extern "C" bool ps5hip_thread_model_fail(unsigned n)noexcept{if(n==1){++contexts;if(std::getenv("PS5HIP_CONTEXT_FAULT"))return true;}if(n==3){++state_attempts;if(std::getenv("PS5HIP_STATE_FAULT"))return true;}return consume(n);}
extern "C" int thread_mutex_lock(pthread_mutex_t*m)noexcept{
 ++locks;if(waiter){waiter=false;check(pthread_mutex_trylock(m)==EBUSY);std::lock_guard<std::mutex> l(coordination);waiter_attempted=true;changed.notify_all();}
 if(consume(3))return EINVAL;return pthread_mutex_lock(m);
}
extern "C" int thread_mutex_unlock(pthread_mutex_t*m)noexcept{int r=pthread_mutex_unlock(m);return consume(4)?EINVAL:r;}
extern "C" int thread_key_create(pthread_key_t*k,void(*d)(void*))noexcept{++keys;if(consume(5)||(std::getenv("PS5HIP_KEY_FAULT")!=nullptr))return EAGAIN;return pthread_key_create(k,d);}
extern "C" int thread_setspecific(pthread_key_t k,const void*p)noexcept{++sets;if(consume(6))return EINVAL;const bool uncertain=consume(7);const int r=pthread_setspecific(k,p);return uncertain?EINVAL:r;}
extern "C" void __wrap___hipUnregisterFatBinary(void**p){__real___hipUnregisterFatBinary(!std::strcmp(mode,"reject-unregister")?reinterpret_cast<void**>(1):p);}
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){++backends;return std::make_unique<Backend>();}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t,int64_t,size_t n,size_t,int type,int64_t*p){check(type==12);*p=next_id++;allocations.emplace(*p,Allocation{nullptr,n});return 0;}
extern "C" int32_t sceKernelMapDirectMemory(void**p,size_t n,int,int,int64_t id,size_t alignment){address=(address+alignment-1)&~(uint64_t(alignment)-1);void*v=mmap(reinterpret_cast<void*>(address),n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED,-1,0);if(v==MAP_FAILED)return -1;address+=n;allocations.at(id).cpu=v;*p=v;return 0;}
extern "C" int32_t sceKernelMunmap(void*p,size_t n){if(!std::strcmp(mode,"cleanup")&&p==code_pointer)return -1;return munmap(p,n);}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t id,size_t){allocations.erase(id);++releases;return 0;}
__attribute__((constructor(101))) static void discovery_before_registration(){
 if(!std::getenv("PS5HIP_DISCOVERY_EARLY"))return;
 check(state_attempts==0&&backends==0&&submits==0);
 std::thread worker([]{int count=713;check(hipGetDeviceCount(&count)==hipSuccess&&count==1);discovery_ok();});worker.join();
 discovery_ok();
 check(state_attempts==0&&backends==0&&submits==0);
 std::puts("DISCOVERY_EARLY state_attempts=0 backend_attempts=0 worker_joined=true");
}
int main(int argc,char**argv){
 check(argc==2);mode=argv[1];
 if(!std::strcmp(mode,"key")){expect_final=false;poison();check(!ps5hip_runtime_registration_ready());std::puts("THREAD_MODEL_PASS key");return 0;}
 if(!std::strcmp(mode,"state-oom")){
  expect_final=false;check(state_attempts==1&&!ps5hip_runtime_registration_ready());const auto c=contexts.load();
  for(unsigned round=0;round<2;++round){
   discovery_error(hipErrorInvalidDeviceFunction);
   void*p=nullptr;check(hipMalloc(&p,4)==hipErrorInvalidDeviceFunction&&!p&&hipFree(nullptr)==hipErrorInvalidDeviceFunction&&hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
   check(__hipRegisterFatBinary(__ps5hip_wrappers_begin)==nullptr&&!ps5hip_runtime_registration_ready());(void)hipGetLastError();
   check(state_attempts==1&&contexts==c&&allocations.empty()&&!backends&&!submits&&!releases);unsetenv("PS5HIP_STATE_FAULT");
  }
  std::puts("THREAD_MODEL_PASS state-oom");return 0;
 }
 if(!std::strcmp(mode,"registration-oom")){expect_final=false;unsetenv("PS5HIP_CONTEXT_FAULT");check(!ps5hip_runtime_registration_ready());(void)hipGetLastError();void*p=nullptr;check(hipMalloc(&p,0)==hipErrorInvalidDeviceFunction&&!p&&allocations.empty()&&!backends);check(!ps5hip_runtime_registration_ready());std::puts("THREAD_MODEL_PASS registration-oom");return 0;}
 check(ps5hip_runtime_registration_ready());
 if(!std::strcmp(mode,"duplicate")){
  expect_final=false;check(__hipRegisterFatBinary(__ps5hip_wrappers_begin)==nullptr);check(!ps5hip_runtime_registration_ready());(void)hipGetLastError();
  std::thread t([]{(void)hipGetLastError();check(!ps5hip_runtime_registration_ready());void*p=nullptr;check(hipMalloc(&p,0)==hipErrorInvalidDeviceFunction&&!p);});t.join();check(!ps5hip_runtime_registration_ready()&&allocations.empty()&&backends==0);ps5hip_runtime_finish();closed();std::puts("THREAD_MODEL_PASS duplicate");return 0;
 }else if(!std::strcmp(mode,"destructor-lock")||!std::strcmp(mode,"destructor-unlock")){
  expect_final=false;std::thread t([]{check(hipFree(nullptr)==hipSuccess);fault=!std::strcmp(mode,"destructor-lock")?3:4;});t.join();poison();check(!ps5hip_runtime_ready()&&allocations.empty());std::printf("THREAD_MODEL_PASS %s\n",mode);return 0;
 }else if(!std::strcmp(mode,"finish-clear")){
  expect_final=false;check(hipFree(nullptr)==hipSuccess);fault=6;ps5hip_runtime_finish();check(!ps5hip_runtime_ready());poison();check(allocations.empty()&&releases==0);std::puts("THREAD_MODEL_PASS finish-clear");return 0;
 }else if(!std::strcmp(mode,"discovery")){
  check(std::getenv("PS5HIP_DISCOVERY_EARLY")&&state_attempts==1);
  discovery_ok();
  for(unsigned flags:{1u,UINT_MAX})check(hipInit(flags)==hipErrorInvalidValue);
  check(hipGetDeviceCount(nullptr)==hipErrorInvalidValue&&hipGetDevice(nullptr)==hipErrorInvalidValue);
  for(int device:{-1,1,INT_MIN,INT_MAX}){
   int value=713;check(hipSetDevice(device)==hipErrorInvalidDevice);
   check(hipDeviceGetAttribute(&value,hipDeviceAttributeWarpSize,device)==hipErrorInvalidDevice&&value==713);
   check(hipGetDevice(&value)==hipSuccess&&value==0);
  }
  for(auto attr:{hipDeviceAttributeMaxThreadsPerBlock,static_cast<hipDeviceAttribute_t>(30000)}){
   int value=713;check(hipDeviceGetAttribute(&value,attr,0)==hipErrorInvalidValue&&value==713);
  }
  check(hipDeviceGetAttribute(nullptr,hipDeviceAttributeWarpSize,1)==hipErrorInvalidValue);
  check(hipInit(1)==hipErrorInvalidValue);
  std::thread worker([]{discovery_ok();check(hipSetDevice(1)==hipErrorInvalidDevice&&hipPeekAtLastError()==hipErrorInvalidDevice);check(hipGetLastError()==hipErrorInvalidDevice&&hipPeekAtLastError()==hipSuccess);});worker.join();
  check(hipPeekAtLastError()==hipErrorInvalidValue);discovery_ok();
  std::thread recovering([]{int value=713;fault=1;check(hipGetDeviceCount(&value)==hipErrorOutOfMemory&&value==713);check(hipPeekAtLastError()==hipErrorOutOfMemory);check(hipGetDeviceCount(&value)==hipSuccess&&value==1);discovery_ok();});recovering.join();
  check(!backends&&!submits&&allocations.empty());
 }else if(!std::strcmp(mode,"discovery-unlock")){
  expect_final=false;int value=713;fault=4;
  check(hipGetDeviceCount(&value)==hipErrorUnknown&&value==713);poison();
  std::puts("THREAD_MODEL_PASS discovery-unlock");return 0;
 }else if(!std::strcmp(mode,"memory-query")){
  check(!std::strcmp(hipGetErrorString(hipErrorNotFound),"allocation not found"));
  unsigned char *p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),37)==hipSuccess);
  auto valid=[&](void *address){void *base=nullptr;size_t size=0;check(hipMemGetAddressRange(&base,&size,address)==hipSuccess&&base==p&&size==37);size=0;check(hipMemPtrGetInfo(address,&size)==hipSuccess&&size==37);};
  valid(p);valid(p+1);valid(p+36);
  void *base=reinterpret_cast<void*>(713);size_t size=713;
  for(void *address:{static_cast<void*>(nullptr),static_cast<void*>(p+37),static_cast<void*>(p+38),static_cast<void*>(&size),reinterpret_cast<void*>(UINTPTR_MAX)}){
   check(hipMemGetAddressRange(&base,&size,address)==hipErrorNotFound&&base==reinterpret_cast<void*>(713)&&size==713);
   check(hipMemPtrGetInfo(address,&size)==hipErrorInvalidValue&&size==713);
  }
  check(hipMemGetAddressRange(nullptr,&size,p)==hipErrorInvalidValue&&size==713);
  check(hipMemGetAddressRange(&base,nullptr,p)==hipErrorInvalidValue&&base==reinterpret_cast<void*>(713));
  check(hipMemPtrGetInfo(p,nullptr)==hipErrorInvalidValue);
  std::thread worker([&]{check(hipPeekAtLastError()==hipSuccess);valid(p+17);});worker.join();
  check(hipPeekAtLastError()==hipErrorInvalidValue);
  check(hipFree(p)==hipSuccess);
  check(hipMemGetAddressRange(&base,&size,p)==hipErrorNotFound&&base==reinterpret_cast<void*>(713)&&size==713);
  check(hipMemPtrGetInfo(p,&size)==hipErrorInvalidValue&&size==713);
  check(!backends&&!submits);
 }else if(!std::strcmp(mode,"memory-query-unlock")){
  unsigned char *p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),37)==hipSuccess);
  expect_final=false;void *base=reinterpret_cast<void*>(713);size_t bytes=713;fault=4;
  check(hipMemGetAddressRange(&base,&bytes,p)==hipErrorUnknown&&base==reinterpret_cast<void*>(713)&&bytes==713);
  poison();std::puts("THREAD_MODEL_PASS memory-query-unlock");return 0;
 }else if(!std::strcmp(mode,"device-copy")){
  unsigned char *p=nullptr,*q=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),37)==hipSuccess&&hipMalloc(reinterpret_cast<void**>(&q),41)==hipSuccess);
  unsigned char source[37],expected[41],after[41];for(unsigned i=0;i<37;++i)source[i]=static_cast<unsigned char>(i*7+3);
  std::memset(expected,0xa5,sizeof(expected));std::memcpy(expected+5,source+3,29);
  check(hipMemcpy(p,source,37,hipMemcpyHostToDevice)==hipSuccess&&hipMemset(q,0xa5,41)==hipSuccess);
  check(hipMemcpy(q+40,p,2,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
  std::thread worker([&]{
   check(hipPeekAtLastError()==hipSuccess&&hipMemcpy(q+5,p+3,29,hipMemcpyDeviceToDevice)==hipSuccess);
   check(hipMemcpy(q,p+36,2,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(q+41,p,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
   check(hipMemcpy(q,p+37,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
   check(hipMemcpy(q,p,SIZE_MAX,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(q,nullptr,1,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(nullptr,p,1,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(q,source,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
   check(hipMemcpy(source,p,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
   check(hipMemcpy(q,q+1,10,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(q+1,q,10,hipMemcpyDeviceToDevice)==hipErrorInvalidValue);
   check(hipMemcpy(q,q,1,hipMemcpyDeviceToDevice)==hipErrorInvalidValue&&hipGetLastError()==hipErrorInvalidValue);
   check(hipMemcpy(nullptr,nullptr,0,hipMemcpyDeviceToDevice)==hipSuccess);
   check(hipMemcpy(reinterpret_cast<void*>(1),reinterpret_cast<void*>(2),0,hipMemcpyDeviceToDevice)==hipSuccess);
  });worker.join();
  check(hipPeekAtLastError()==hipErrorInvalidValue);
  check(hipMemcpy(after,q,41,hipMemcpyDeviceToHost)==hipSuccess&&std::memcmp(after,expected,41)==0);
  unsigned char unchanged[37];check(hipMemcpy(unchanged,p,37,hipMemcpyDeviceToHost)==hipSuccess&&std::memcmp(unchanged,source,37)==0);
  check(hipMemcpy(q+34,q+5,7,hipMemcpyDeviceToDevice)==hipSuccess);std::memcpy(expected+34,expected+5,7);
  check(hipMemcpy(after,q,41,hipMemcpyDeviceToHost)==hipSuccess&&std::memcmp(after,expected,41)==0);
  check(hipFree(p)==hipSuccess&&hipMemcpy(q,p,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
  check(hipMemcpy(p,q,1,hipMemcpyDeviceToDevice)==hipErrorInvalidDevicePointer);
  check(hipMemcpy(p,q,0,hipMemcpyDeviceToDevice)==hipSuccess&&hipFree(q)==hipSuccess);
  check(!backends&&!submits);
 }else if(!std::strcmp(mode,"memset")){
  unsigned char*p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),17)==hipSuccess);
  check(hipMemset(p,0xa5,17)==hipSuccess);
  std::thread t([&]{
   check(hipMemset(p+1,0x123,16)==hipSuccess&&hipMemset(p+2,-1,3)==hipSuccess);
   check(hipMemset(p+16,0,2)==hipErrorInvalidValue);
   check(hipMemset(p+17,0,1)==hipErrorInvalidValue);
   check(hipMemset(p,0,SIZE_MAX)==hipErrorInvalidValue);
   check(hipMemset(nullptr,0,1)==hipErrorInvalidValue&&hipGetLastError()==hipErrorInvalidValue);
   unsigned local=7;check(hipMemset(&local,0,1)==hipErrorInvalidValue&&local==7);
   check(hipMemset(p+17,0,0)==hipSuccess&&hipMemset(reinterpret_cast<void*>(1),0,0)==hipSuccess);
   check(hipMemset(nullptr,0,0)==hipSuccess&&hipPeekAtLastError()==hipSuccess);
  });t.join();
  unsigned char out[17]{};check(hipMemcpy(out,p,17,hipMemcpyDeviceToHost)==hipSuccess);
  for(unsigned i=0;i<17;++i)check(out[i]==(i==0?0xa5:i>=2&&i<5?0xff:0x23));
  check(!backends&&!submits&&hipPeekAtLastError()==hipSuccess);
  check(hipFree(p)==hipSuccess&&hipMemset(p,0,1)==hipErrorInvalidValue);
  check(hipMemset(p,0,0)==hipSuccess);
 }else if(!std::strcmp(mode,"context-recover")){
  std::thread t([]{void*p=nullptr;fault=1;check(hipMalloc(&p,4)==hipErrorOutOfMemory&&!p);fault=1;check(hipMalloc(&p,4)==hipErrorOutOfMemory&&!p);check(hipMalloc(&p,4)==hipSuccess&&p&&hipPeekAtLastError()==hipSuccess);check(hipFree(p)==hipSuccess);});t.join();
 }else if(!std::strcmp(mode,"marker-exit")){
  std::thread t([]{fault=1;void*p=nullptr;check(hipMalloc(&p,4)==hipErrorOutOfMemory&&!p);check(hipPeekAtLastError()==hipErrorOutOfMemory);});t.join();check(allocations.empty()&&!backends);
 }else if(!std::strcmp(mode,"context")){
  std::thread t([]{auto n=contexts.load();const auto k=keys.load();check(hipPeekAtLastError()==hipSuccess&&hipGetLastError()==hipSuccess&&contexts==n&&keys==k&&allocations.empty());fault=1;void*p=nullptr;check(hipMalloc(&p,4)==hipErrorOutOfMemory&&!p);n=contexts;check(hipPeekAtLastError()==hipErrorOutOfMemory&&contexts==n);check(hipGetLastError()==hipErrorOutOfMemory&&hipPeekAtLastError()==hipSuccess&&contexts==n);check(hipMalloc(&p,4)==hipSuccess&&p);check(hipFree(p)==hipSuccess);});t.join();
 }else if(!std::strcmp(mode,"stacks")){
  std::thread a([]{check(__hipPushCallConfiguration(dim3(3,2,1),dim3(8,8),11,nullptr)==hipSuccess);step(1);until(2);dim3 g,b;size_t s;hipStream_t stream;check(__hipPopCallConfiguration(&g,&b,&s,&stream)==hipSuccess&&g.x==3&&g.y==2&&b.x==8&&s==11);check(__hipPushCallConfiguration(dim3(7),dim3(64),19,nullptr)==hipSuccess);check(__hipPushCallConfiguration(dim3(9),dim3(64),23,nullptr)==hipSuccess);check(__hipPopCallConfiguration(&g,&b,&s,&stream)==hipSuccess&&g.x==9&&s==23);check(__hipPopCallConfiguration(&g,&b,&s,&stream)==hipSuccess&&g.x==7&&s==19);check(hipFree(reinterpret_cast<void*>(1))==hipErrorInvalidDevicePointer);step(3);until(4);check(hipGetLastError()==hipErrorInvalidDevicePointer&&hipPeekAtLastError()==hipSuccess);});
  std::thread b([]{until(1);dim3 g,b;size_t s;hipStream_t stream;check(__hipPopCallConfiguration(&g,&b,&s,&stream)!=hipSuccess);check(__hipPushCallConfiguration(dim3(5),dim3(64),13,nullptr)==hipSuccess);step(2);until(3);check(__hipPopCallConfiguration(&g,&b,&s,&stream)==hipSuccess&&g.x==5&&s==13);check(hipPeekAtLastError()==hipSuccess);step(4);});a.join();b.join();check(hipPeekAtLastError()==hipSuccess);
 }else if(!std::strcmp(mode,"vector")){
  fault=2;check(__hipPushCallConfiguration(dim3(1),dim3(64),0,nullptr)==hipErrorOutOfMemory);dim3 g,b;size_t s;hipStream_t stream;check(__hipPopCallConfiguration(&g,&b,&s,&stream)!=hipSuccess);check(__hipPushCallConfiguration(dim3(2),dim3(64),0,nullptr)==hipSuccess);fault=2;check(__hipPushCallConfiguration(dim3(7),dim3(64),0,nullptr)==hipErrorOutOfMemory);check(__hipPopCallConfiguration(&g,&b,&s,&stream)==hipSuccess&&g.x==2);
 }else if(!std::strcmp(mode,"lock")||!std::strcmp(mode,"unlock")||!std::strcmp(mode,"setspecific")||!std::strcmp(mode,"uncertain-install")){
  expect_final=false;std::thread t([]{fault=!std::strcmp(mode,"lock")?3:!std::strcmp(mode,"unlock")?4:!std::strcmp(mode,"uncertain-install")?7:6;void*p=nullptr;check(hipMalloc(&p,4)==hipErrorUnknown);check(bool(p)==!std::strcmp(mode,"unlock"));});t.join();check(allocations.size()==size_t(!std::strcmp(mode,"unlock"))&&backends==0&&submits==0&&releases==0);poison();check(hipGetLastError()==hipErrorUnknown&&hipPeekAtLastError()==hipErrorUnknown);std::printf("THREAD_MODEL_PASS %s\n",mode);return 0;
 }else if(!std::strcmp(mode,"clear")){
  expect_final=false;std::thread t([]{fault=1;void*p=nullptr;check(hipMalloc(&p,4)==hipErrorOutOfMemory);fault=6;check(hipGetLastError()==hipErrorUnknown);});t.join();poison();std::printf("THREAD_MODEL_PASS %s\n",mode);return 0;
 }else if(!std::strcmp(mode,"abandoned")||!std::strcmp(mode,"live-finish")){
  expect_final=false;std::thread t([]{check(__hipPushCallConfiguration(dim3(1),dim3(64),0,nullptr)==hipSuccess);if(!std::strcmp(mode,"live-finish")){step(1);until(2);}});
  if(!std::strcmp(mode,"live-finish")){until(1);ps5hip_runtime_finish();step(2);t.join();}else{
   t.join();const auto c=contexts.load(),st=state_attempts.load();const auto allocated=next_id;const auto n=allocations.size();
   auto abandoned_queries=[] {poison();for(unsigned i=0;i<3;++i)check(hipPeekAtLastError()==hipErrorUnknown&&hipGetLastError()==hipErrorUnknown);};
   abandoned_queries();std::thread peer(abandoned_queries);peer.join();
   check(contexts==c&&state_attempts==st&&next_id==allocated&&allocations.size()==n&&!submits&&!backends&&!releases);ps5hip_runtime_finish();
  }check(!ps5hip_runtime_ready());closed();std::printf("THREAD_MODEL_PASS %s\n",mode);return 0;
 }else if(!std::strcmp(mode,"finish-wait")){
  expect_final=false;unsigned*p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),257*4)==hipSuccess);pause_submit=true;
  std::thread a([&]{threads_launch(p,257,31);step(1);until(2);});
  std::unique_lock<std::mutex> l(coordination);changed.wait(l,[]{return entered;});l.unlock();
  std::thread b([]{waiter=true;ps5hip_runtime_finish();check(!ps5hip_runtime_ready());step(2);});
  l.lock();changed.wait(l,[]{return waiter_attempted;});check(submits==1&&releases==0);resume=true;changed.notify_all();l.unlock();a.join();b.join();closed();check(allocations.size()==1&&releases==4);std::puts("THREAD_MODEL_PASS finish-wait");return 0;
 }else{
  unsigned*p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),257*4)==hipSuccess);
  unsigned*q=nullptr;const bool device_copy=!std::strcmp(mode,"blocked-device-copy");
  if(device_copy){check(hipMalloc(reinterpret_cast<void**>(&q),259*4)==hipSuccess&&hipMemset(q,0xa5,259*4)==hipSuccess);}
  const bool blocked=!std::strncmp(mode,"blocked-",8);pause_submit=blocked;
  const bool ordinary=!std::strcmp(mode,"ownership")||!std::strcmp(mode,"no-finish")||!std::strcmp(mode,"reject-unregister");
  std::thread a([&]{threads_launch(p,257,31);if(ordinary||blocked)check(hipGetLastError()==hipSuccess);});
  if(blocked){std::unique_lock<std::mutex> l(coordination);changed.wait(l,[]{return entered;});l.unlock();
   std::thread b([&]{waiter=true;if(!std::strcmp(mode,"blocked-free"))check(hipFree(p)==hipSuccess);else if(!std::strcmp(mode,"blocked-sync"))check(hipDeviceSynchronize()==hipSuccess);else if(!std::strcmp(mode,"blocked-memset"))check(hipMemset(p,0x5a,257*4)==hipSuccess);else if(device_copy)check(hipMemcpy(q+1,p,257*4,hipMemcpyDeviceToDevice)==hipSuccess);else if(!std::strcmp(mode,"blocked-copy")){unsigned values[257];check(hipMemcpy(values,p,sizeof(values),hipMemcpyDeviceToHost)==hipSuccess);for(unsigned i=0;i<257;++i)check(values[i]==i*17u+31);}else threads_launch(p,73,71);});
   l.lock();changed.wait(l,[]{return waiter_attempted;});check(submits==1&&backends==1&&releases==0);resume=true;changed.notify_all();l.unlock();a.join();b.join();
  }else a.join();
  if(!ordinary&&!blocked){expect_final=false;check(ps5hip_runtime_terminal());const bool cleanup=!std::strcmp(mode,"cleanup");check(allocations.size()==size_t(cleanup?2:5)&&releases==unsigned(cleanup?3:0)&&submits==1&&backends==1&&ps5hip_runtime_gpu_unresolved()!=cleanup);std::thread b([]{poison();});b.join();std::printf("THREAD_MODEL_PASS %s retained=true\n",mode);std::fflush(nullptr);std::_Exit(0);}
  if(std::strcmp(mode,"blocked-free")){std::thread b([&]{unsigned out[257];check(hipMemcpy(out,p,sizeof(out),hipMemcpyDeviceToHost)==hipSuccess);const unsigned n=!std::strcmp(mode,"blocked-launch")?73:257;for(unsigned i=0;i<n;++i)check(out[i]==(!std::strcmp(mode,"blocked-memset")?0x5a5a5a5au:i*17u+(n==73?71:31)));check(hipFree(p)==hipSuccess);check(hipFree(p)==hipErrorInvalidDevicePointer);});b.join();}
  if(device_copy){unsigned out[259];check(hipMemcpy(out,q,sizeof(out),hipMemcpyDeviceToHost)==hipSuccess&&out[0]==0xa5a5a5a5u&&out[258]==0xa5a5a5a5u);for(unsigned i=0;i<257;++i)check(out[i+1]==i*17u+31);check(hipFree(q)==hipSuccess);}
  check(backends==1&&submits==unsigned(!std::strcmp(mode,"blocked-launch")?2:1));
 }
 if(!std::strcmp(mode,"no-finish")){expect_final=false;check(!ps5hip_runtime_ready());std::puts("THREAD_MODEL_PASS no-finish");return 0;}
 if(!std::strcmp(mode,"reject-unregister"))expect_final=false;
 ps5hip_runtime_finish();check(ps5hip_runtime_ready());closed();std::printf("THREAD_MODEL_PASS %s\n",mode);return 0;
}
