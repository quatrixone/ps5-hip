#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "platform/runtime.hpp"
#include "utils/flags.hpp"
#include <atomic>
#include <thread>
#include <cerrno>
#include <dirent.h>
static ps5hip::clr::Observation public_observed;
static bool init_failure=false;
static unsigned final_kind=0;
struct FinalObserver {
 ~FinalObserver(){
  if(final_kind==0)return;
  const bool retained=final_kind==4||final_kind==5;
  checked(public_observed.gate_destroyed==!retained,__LINE__);
  const unsigned expected=(final_kind==2||final_kind==6||final_kind==7)?1:(final_kind==3?4:3);
  checked(public_observed.queues==expected,__LINE__);
  checked(public_observed.retired_queues==(final_kind==5?2:expected),__LINE__);
  const unsigned expected_allocations=final_kind==6?1:0;
  checked(public_observed.allocations==expected_allocations&&public_observed.frees==expected_allocations,__LINE__);
  std::printf("HIP_PUBLIC_FINAL queues=%u retired=%u gate_destroyed=%d retained=%d\n",unsigned(public_observed.queues),unsigned(public_observed.retired_queues),bool(public_observed.gate_destroyed),retained);
 }
};
__attribute__((init_priority(101))) static FinalObserver final_observer;
static std::atomic<bool> fail_create{false},fail_join{false};
static std::atomic<unsigned> rejected_join{0};
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){if(fail_create.exchange(false))return EAGAIN;return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){if(fail_join.exchange(false)){++rejected_join;return EINVAL;}return __real_pthread_join(p,v);}
extern "C" bool ps5hip_clr_admit_device(){return !init_failure&&ps5hip::clr::register_runtime_device(backend,public_observed);}
static std::atomic<bool> callback_entered{false},callback_resume{false};
static std::atomic<int> self_result{-1};
static void hold(void*){
 callback_entered.store(true,std::memory_order_release);
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(!callback_resume.load(std::memory_order_acquire)){
  if(std::chrono::steady_clock::now()>=deadline){std::fputs("HIP_CALLBACK_TIMEOUT\n",stderr);std::abort();}
  std::this_thread::yield();
 }
}
static void self_destroy(void* p){self_result.store(int(hipStreamDestroy(static_cast<hipStream_t>(p))),std::memory_order_release);}
static void ok(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"HIP_ADMISSION_ERROR %d\n",int(e));std::abort();}}
static unsigned descriptor_count(){
 auto* d=opendir("/proc/self/fd");checked(d!=nullptr,__LINE__);unsigned n=0;
 while(readdir(d))++n;checked(closedir(d)==0,__LINE__);return n;
}
int main(int argc,char** argv){
 const std::string mode=argc>1?argv[1]:"normal";
 const bool reset_case=mode=="reset"||mode=="reset-join-fail";
 if(mode=="init-fail"){
  init_failure=true;checked(hipInit(0)==hipErrorInvalidDevice,__LINE__);checked(hipInit(0)==hipErrorInvalidDevice,__LINE__);
  std::puts("HIP_PUBLIC_PASS mode=init-fail gpu_execution=false");return 0;
 }
 final_kind=mode=="create-fail"?2:((mode=="tls"||mode=="default")?3:((mode=="join-fail"||mode=="reset-join-fail")?4:(mode=="self-destroy"?5:(mode=="memory"?6:1))));
 const bool module_case=mode=="module"||mode=="module-unbundler";
 if(module_case)final_kind=7;
 ok(hipInit(0));int count=0;ok(hipGetDeviceCount(&count));checked(count==1,__LINE__);
 hipMemPool_t pool=nullptr;ok(hipDeviceGetDefaultMemPool(&pool,0));checked(pool!=nullptr,__LINE__);
 // Warm the genuine auxiliary thread setup before narrowly arming a queue fault.
 hipStream_t warm=nullptr;ok(hipStreamCreate(&warm));ok(hipStreamDestroy(warm));
 if(module_case){
  checked(argc==4,__LINE__);const std::string path=argv[2];
  checked(HIP_USE_RUNTIME_UNBUNDLER==(mode=="module-unbundler"),__LINE__);
  std::ifstream input(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(input)),{});
  checked(bytes.size()>64,__LINE__);input.close();
  const unsigned fds=descriptor_count();
  hipModule_t module=nullptr;
  checked(hipModuleLoad(nullptr,path.c_str())==hipErrorInvalidValue,__LINE__);
  checked(hipModuleLoadData(&module,reinterpret_cast<void*>(1))==hipErrorNotSupported&&module==nullptr,__LINE__);
  checked(hipModuleLoad(&module,(path+".missing").c_str())==hipErrorFileNotFound&&module==nullptr,__LINE__);
  checked(hipModuleLoad(&module,argv[3])==hipErrorNoBinaryForGpu&&module==nullptr,__LINE__);
  const std::string invalid=path+".invalid";
  for(size_t size:{size_t(0),size_t(63),size_t(128),size_t(64*1024*1024+1)}){
   {std::ofstream out(invalid,std::ios::binary);if(size){out.seekp(size-1);out.put(0);}checked(out.good(),__LINE__);}
   checked(hipModuleLoad(&module,invalid.c_str())==hipErrorInvalidImage&&module==nullptr,__LINE__);
   checked(std::remove(invalid.c_str())==0,__LINE__);
  }
  for(unsigned cycle=0;cycle<8;++cycle){
   const std::string owned=path+".owned";
   {std::ofstream out(owned,std::ios::binary);out.write(bytes.data(),bytes.size());checked(out.good(),__LINE__);}
   ok(hipModuleLoad(&module,owned.c_str()));checked(module!=nullptr,__LINE__);
   hipModule_t independent=nullptr;ok(hipModuleLoad(&independent,owned.c_str()));
   checked(independent&&independent!=module,__LINE__);
   checked(std::remove(owned.c_str())==0,__LINE__);
   hipFunction_t first=nullptr,second=nullptr,missing=nullptr;
   ok(hipModuleGetFunction(&first,module,"indexed_store"));ok(hipModuleGetFunction(&second,module,"vector_add"));
   checked(first&&second&&first!=second,__LINE__);
   hipFunction_t independent_function=nullptr;
   ok(hipModuleGetFunction(&independent_function,independent,"indexed_store"));
   checked(independent_function&&independent_function!=first&&independent_function!=second,__LINE__);
   checked(hipModuleGetFunction(&missing,module,"absent_kernel")==hipErrorNotFound&&missing==nullptr,__LINE__);
   ok(hipModuleUnload(module));checked(hipModuleUnload(module)==hipErrorNotFound,__LINE__);module=nullptr;
   hipFunction_t retained=nullptr;ok(hipModuleGetFunction(&retained,independent,"indexed_store"));
   checked(retained==independent_function,__LINE__);ok(hipModuleUnload(independent));
  }
  checked(descriptor_count()==fds&&!backend_calls&&!submits&&allocations.empty(),__LINE__);
  std::printf("HIP_PUBLIC_PASS mode=%s gpu_execution=false\n",mode.c_str());return 0;
 }
 if(mode=="memory"){
  void* pointer=nullptr;
  checked(hipMalloc(nullptr,256)==hipErrorInvalidValue,__LINE__);
  ok(hipMalloc(&pointer,256));
  checked(pointer&&public_observed.allocations==1&&allocations.size()==1,__LINE__);
  checked(hipFree(static_cast<char*>(pointer)+1)==hipErrorInvalidValue,__LINE__);
  checked(public_observed.frees==0&&allocations.size()==1,__LINE__);
  ok(hipFree(pointer));
  checked(public_observed.frees==1&&allocations.empty(),__LINE__);
  checked(hipFree(pointer)==hipErrorInvalidValue,__LINE__);
  checked(!backend_calls&&!submits,__LINE__);
  std::puts("HIP_PUBLIC_PASS mode=memory gpu_execution=false");return 0;
 }
 if(mode=="create-fail"){
  fail_create=true;hipStream_t failed=nullptr;checked(hipStreamCreate(&failed)!=hipSuccess&&failed==nullptr,__LINE__);checked(!fail_create,__LINE__);
  std::puts("HIP_PUBLIC_PASS mode=create-fail gpu_execution=false");return 0;
 }
 hipStream_t a=nullptr,b=nullptr;ok(hipStreamCreate(&a));ok(hipStreamCreateWithFlags(&b,hipStreamNonBlocking));
 hipEvent_t producer=nullptr,consumer=nullptr;ok(hipEventCreateWithFlags(&producer,hipEventDisableTiming));ok(hipEventCreateWithFlags(&consumer,hipEventDisableTiming));
 ok(hipLaunchHostFunc(a,hold,nullptr));ok(hipEventRecord(producer,a));ok(hipStreamWaitEvent(b,producer,0));ok(hipEventRecord(consumer,b));
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(!callback_entered.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
 checked(callback_entered&&hipEventQuery(consumer)==hipErrorNotReady,__LINE__);
 if(reset_case){
  const auto queues=public_observed.queues.load(),retired=public_observed.retired_queues.load();
  fail_join=true;
  checked(hipDeviceReset()==hipErrorNotSupported,__LINE__);
  checked(hipDeviceReset()==hipErrorNotSupported,__LINE__);
  checked(fail_join&&rejected_join==0&&!amd::Runtime::quarantined(),__LINE__);
  hipMemPool_t same=nullptr;ok(hipDeviceGetDefaultMemPool(&same,0));checked(same==pool,__LINE__);
  checked(public_observed.queues==queues&&public_observed.retired_queues==retired,__LINE__);
  checked(hipEventQuery(consumer)==hipErrorNotReady,__LINE__);
  fail_join=false;
 }
 callback_resume.store(true,std::memory_order_release);ok(hipEventSynchronize(consumer));ok(hipStreamSynchronize(a));ok(hipStreamSynchronize(b));
 ok(hipEventDestroy(producer));ok(hipEventDestroy(consumer));ok(hipStreamDestroy(b));
 if(mode=="join-fail"||mode=="reset-join-fail"){
  fail_join=true;checked(hipStreamDestroy(a)==hipErrorUnknown,__LINE__);checked(rejected_join==1&&amd::Runtime::quarantined(),__LINE__);
  checked(hipStreamSynchronize(a)==hipErrorUnknown,__LINE__);checked(hipStreamDestroy(a)==hipErrorUnknown&&rejected_join==1,__LINE__);
 }else if(mode=="self-destroy"){
  ok(hipLaunchHostFunc(a,self_destroy,a));deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(self_result.load(std::memory_order_acquire)<0&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
  checked(self_result==hipErrorUnknown&&amd::Runtime::quarantined(),__LINE__);
 }else{
  ok(hipStreamDestroy(a));
  if(mode=="default")ok(hipStreamSynchronize(nullptr));
  if(mode=="tls"){
   std::thread t([]{hipEvent_t e=nullptr;ok(hipEventCreateWithFlags(&e,hipEventDisableTiming));ok(hipEventRecord(e,hipStreamPerThread));ok(hipEventSynchronize(e));ok(hipEventDestroy(e));});t.join();
  }
 }
 checked(!backend_calls&&!submits&&allocations.empty(),__LINE__);
 std::printf("HIP_PUBLIC_PASS mode=%s gpu_execution=false quarantine=%d\n",mode.c_str(),amd::Runtime::quarantined());
}
