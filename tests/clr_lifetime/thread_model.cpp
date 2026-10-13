// Focused fault tests of actual patched ROCclr Thread/HostQueue, not a scheduler model.
// Reuse existing explicit synthetic provider definitions; no native submission occurs.
#define main unused_adapter_transport_main
#include "../clr_adapter_model.cpp"
#undef main
#include <atomic>
#include <sched.h>
static const char* lifetime_mode=nullptr;
static std::atomic<unsigned> create_calls{0},join_calls{0},queue_destructors{0};
static std::atomic<bool> armed{false},self_released{false};
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t* t,const pthread_attr_t* a,void*(*f)(void*),void* p){
 if(armed){++create_calls;if(!std::strcmp(lifetime_mode,"thread-create-failure")||!std::strcmp(lifetime_mode,"queue-create-failure"))return EAGAIN;}
 return __real_pthread_create(t,a,f,p);
}
extern "C" int __wrap_pthread_join(pthread_t t,void** p){
 if(armed){++join_calls;if(!std::strcmp(lifetime_mode,"queue-join-failure"))return EINVAL;}
 return __real_pthread_join(t,p);
}
static unsigned aligned_calls=0,aligned_frees=0,aligned_fail=0;
static bool allocation_fault_active=false;
extern "C" void* __real__ZN3amd13AlignedMemory8allocateEmm(size_t,size_t);
extern "C" void __real__ZN3amd13AlignedMemory10deallocateEPv(void*);
extern "C" void* __wrap__ZN3amd13AlignedMemory8allocateEmm(size_t n,size_t align){
 if(allocation_fault_active&&++aligned_calls==aligned_fail)return nullptr;
 auto* p=__real__ZN3amd13AlignedMemory8allocateEmm(n,align);if(allocation_fault_active)check(p&&reinterpret_cast<uintptr_t>(p)%64==0);return p;
}
extern "C" void __wrap__ZN3amd13AlignedMemory10deallocateEPv(void* p){if(allocation_fault_active&&p)++aligned_frees;__real__ZN3amd13AlignedMemory10deallocateEPv(p);}
static bool monitor_failed=false;
extern "C" void* __real_malloc(size_t);
extern "C" void* __wrap_malloc(size_t n){
 if(allocation_fault_active&&aligned_fail==4&&aligned_calls==3&&!monitor_failed){monitor_failed=true;return nullptr;}
 return __real_malloc(n);
}
struct TestThread final:amd::Thread {
 std::atomic<bool> ran{false};bool self_result=true;unsigned published=0;
 explicit TestThread(size_t stack=0):amd::Thread("lifetime-test",stack){}
 void run(void* p)override{
  check(amd::Thread::current()==this&&p==&published&&published==0x12345678);
  if(!std::strcmp(lifetime_mode,"thread-self-join"))self_result=join();
  ran.store(true,std::memory_order_release);
 }
};
struct TestQueue final:amd::HostQueue {
 TestQueue(amd::Context& c,amd::Device& d):HostQueue(c,d,0){}
 ~TestQueue()override{++queue_destructors;}
};
static void CL_CALLBACK self_release(cl_event,int32_t status,void* opaque){
 auto* q=static_cast<TestQueue*>(opaque);check(status==CL_COMPLETE&&q->referenceCount()==1);
 check(&q->thread()==amd::Thread::current());check(q->release()==0);
 check(q->referenceCount()==0&&queue_destructors==0&&join_calls==0);
 self_released.store(true,std::memory_order_release);
}
static void finish(const char* detail){std::printf("CLR_LIFETIME_PASS mode=%s %s\n",lifetime_mode,detail);std::fflush(nullptr);}
int main(int argc,char** argv){
 check(argc==2);lifetime_mode=argv[1];amd::IS_HIP=true;AMD_DIRECT_DISPATCH=false;armed=true;
 if(!std::strncmp(lifetime_mode,"allocation-",11)){
  aligned_fail=unsigned(std::atoi(lifetime_mode+11));check(aligned_fail>=1&&aligned_fail<=4);allocation_fault_active=true;
  bool caught=false;try{TestThread t;}catch(const std::bad_alloc&){caught=true;}
  allocation_fault_active=false;check(caught&&aligned_calls==(aligned_fail==4?3:aligned_fail)&&aligned_frees==aligned_fail-1&&(aligned_fail!=4||monitor_failed)&&create_calls==0&&join_calls==0);
  finish("bad_alloc=true prior_semaphores_freed=true os_threads_created=0");return 0;
 }
 if(!std::strncmp(lifetime_mode,"thread-",7)){
  TestThread t;
  if(!std::strcmp(lifetime_mode,"thread-create-failure")){
   check(t.state()==amd::Thread::FAILED&&!t.handle()&&!t.start());check(t.join());check(create_calls==1&&join_calls==0);
   finish("create_failed=true no_wait=true joins=0");return 0;
  }
  check(t.state()==amd::Thread::INITIALIZED&&t.handle()&&!t.ran);t.published=0x12345678;check(t.start(&t.published));
  while(t.state()!=amd::Thread::FINISHED)sched_yield();
  check(t.ran.load(std::memory_order_acquire)&&t.handle());
  if(!std::strcmp(lifetime_mode,"thread-self-join"))check(!t.self_result&&join_calls==0);
  check(t.join()&&!t.handle()&&join_calls==1);check(t.join()&&join_calls==1);
  finish("startup_published=true finished_joined=true consumed_handle=true joins=1");return 0;
 }
 // All manual owners are deliberately retained on uncertain queue retirement.
 armed=false;
 { TestThread warmup(512*1024);warmup.published=0x12345678;check(warmup.start(&warmup.published)&&warmup.join()); }
 auto* observed=new ps5hip::clr::Observation;
 auto* device=ps5hip::clr::create_device(backend,*observed);check(device);
 amd::Context::Info info{};check(amd::Context::checkProperties(nullptr,&info)==CL_SUCCESS);
 auto* context=new amd::Context({device},info);check(context->create(nullptr)==CL_SUCCESS);
 create_calls=0;join_calls=0;armed=true;
 auto* q=new TestQueue(*context,*device);
 if(!std::strcmp(lifetime_mode,"queue-create-failure")){
  check(!q->create()&&q->thread().state()==amd::Thread::FAILED&&observed->queues==0);q->release();
  check(queue_destructors==1&&join_calls==0);context->release();delete device;check(observed->gate_destroyed);delete observed;
  finish("create_failed=true queue_deleted=true owners_retired=true joins=0");return 0;
 }
 check(q->create()&&observed->queues==1);
 if(!std::strcmp(lifetime_mode,"queue-self-release")){
  auto* marker=new amd::Marker(*q,true);check(marker->setCallback(CL_COMPLETE,self_release,q));marker->enqueue();
  while(!self_released.load(std::memory_order_acquire))sched_yield();
  check(q->referenceCount()==0&&queue_destructors==0&&join_calls==0&&observed->retired_queues==0&&!observed->gate_destroyed&&context->referenceCount()>0&&!backend_calls);
  finish("queue_refcount=0 queue_retained=true owners_retained=true worker_alive=true joins=0");std::_Exit(0);
 }
 check(!std::strcmp(lifetime_mode,"queue-join-failure")||!std::strcmp(lifetime_mode,"queue-normal"));
 q->release();check(join_calls==1);
 if(!std::strcmp(lifetime_mode,"queue-join-failure")){
  check(q->referenceCount()==0&&q->thread().handle()&&q->thread().state()==amd::Thread::FINISHED);
  check(queue_destructors==0&&observed->retired_queues==1&&!observed->gate_destroyed&&context->referenceCount()>0&&!backend_calls);
  finish("queue_refcount=0 queue_retained=true owners_retained=true virtual_retired=1 join_failed=true");std::_Exit(0);
 }
 check(queue_destructors==1&&observed->retired_queues==1);context->release();delete device;check(observed->gate_destroyed&&!backend_calls);delete observed;
 finish("queue_deleted=true owners_retired=true joins=1");return 0;
}
