// Hold completed fill work before batch retirement; compare the finish barrier with a bypass.
#define main adapter_unused_main
#include "clr_adapter_model.cpp"
#undef main
#undef check
#include <hip/hip_runtime.h>
#include <thread>
#include "hip_internal.hpp"
static ps5hip::clr::Observation observed;
static bool completed=false;
struct Final {
 ~Final(){
  checked(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain,__LINE__);
  checked(observed.allocations==1&&observed.frees==1&&observed.launches==0&&backend_calls==0,__LINE__);
  std::printf("RETIREMENT_COUNTS queues=%u retired=%u\n",unsigned(observed.queues),unsigned(observed.retired_queues));
  checked(observed.queues==1&&observed.retired_queues==1&&allocations.empty(),__LINE__);
  std::puts("RETIREMENT_FINAL queues=1 retired=1 ownership=clean gpu_execution=false");
 }
};
__attribute__((init_priority(101))) static Final final;
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* p,const pthread_attr_t* a,void*(*f)(void*),void* d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void** v){return __real_pthread_join(p,v);}
static void ok(hipError_t status,int line){if(status!=hipSuccess){std::fprintf(stderr,"PUBLIC_FILL_ERROR line=%d status=%d\n",line,int(status));std::abort();}}
#define HIP_OK(x) ok(x,__LINE__)

static std::mutex hold_mutex; static std::condition_variable hold_cv;
static bool held=false,released=false,sync_started=false,sync_done=false;
extern "C" void qualification_hold_after_fill(){std::unique_lock<std::mutex> l(hold_mutex);held=true;hold_cv.notify_all();checked(hold_cv.wait_for(l,std::chrono::seconds(10),[]{return released;}),__LINE__);}
int main(int argc,char**argv){
 checked(argc==2&&(!std::strcmp(argv[1],"old")||!std::strcmp(argv[1],"new")),__LINE__);
 bool old=std::strcmp(argv[1],"old")==0;
 HIP_OK(hipInit(0));hipStream_t stream=nullptr;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 void* p=nullptr;HIP_OK(hipMalloc(&p,128));HIP_OK(hipMemsetAsync(p,0x39,128,stream));
 {std::unique_lock<std::mutex> l(hold_mutex);checked(hold_cv.wait_for(l,std::chrono::seconds(5),[]{return held;}),__LINE__);}
 std::thread waiter([&]{ {std::lock_guard<std::mutex> l(hold_mutex);sync_started=true;hold_cv.notify_all();} HIP_OK(hipStreamSynchronize(stream)); {std::lock_guard<std::mutex> l(hold_mutex);sync_done=true;hold_cv.notify_all();}});
 {std::unique_lock<std::mutex> l(hold_mutex);checked(hold_cv.wait_for(l,std::chrono::seconds(5),[]{return sync_started;}),__LINE__);bool early=hold_cv.wait_for(l,std::chrono::milliseconds(500),[]{return sync_done;});checked(early==old,__LINE__);released=true;hold_cv.notify_all();}
 waiter.join();
 if(old)HIP_OK(hipStreamDestroy(stream));
 HIP_OK(hipFree(p));
 if(!old)HIP_OK(hipStreamDestroy(stream));
 completed=true;std::puts(old?"RETIREMENT_OLD early_sync_observed=true":"RETIREMENT_NEW held_until_retirement=true free_once=true");
}
