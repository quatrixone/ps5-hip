// Genuine compiler -> public HIP -> CLR -> kernarg packer. Only GPU transport is modeled.
#define main unused_adapter_main
#define Transport UnusedTransport
#define backend unused_backend
#include "clr_adapter_model.cpp"
#undef backend
#undef Transport
#undef main
#undef check
#include "hip_aggregate_fixture.h"
#include <array>
extern "C" const void* aggregate_shapes;
static ps5hip::clr::Observation observed;
static std::array<unsigned char,164> expected_bytes;
static bool completed=false;
#define require(value) checked(bool(value),__LINE__)
class AggregateTransport final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{}
 int32_t submit(uint32_t*w,uint32_t count)override{
  ++submits;
  {std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();require(changed.wait_for(lock,std::chrono::seconds(5),[]{return resume;}));}
  std::map<unsigned,std::vector<unsigned>> regs;
  for(size_t at=0;at<count;){auto h=w[at];size_t n=((h>>16)&0x3fff)+1;require(h>>30==3&&n<count-at);if((h&0xffff)==0x7600)regs.emplace(w[at+1],std::vector<unsigned>(w+at+2,w+at+n+1));at+=n+1;}
  auto&s=regs.at(0x240);uint64_t arg=uint64_t(s.at(4))|(uint64_t(s.at(5))<<32);live(arg,180);
  auto*bytes=reinterpret_cast<const unsigned char*>(arg);uint64_t output;std::memcpy(&output,bytes,8);live(output,172);
  // These are compiler metadata offsets, including its inter-argument padding.
  for(size_t j=24;j<32;++j)require(bytes[j]==0);
  const size_t offsets[]={8,12,32,48,176},dest[]={0,4,16,32,160},sizes[]={4,12,16,128,4};
  for(unsigned i=0;i<5;++i)for(size_t j=0;j<sizes[i];++j){
   if(i==3&&j>=12&&j<16)continue; // C++ source padding is not a portable member value.
   if(bytes[offsets[i]+j]!=expected_bytes[dest[i]+j]){std::fprintf(stderr,"AGGREGATE_CAPTURE_MISMATCH field=%u byte=%zu\n",i,j);std::abort();}
  }
  if(std::strcmp(mode,"noop"))for(unsigned i=0;i<5;++i)std::memcpy(reinterpret_cast<void*>(output+dest[i]),bytes+offsets[i],sizes[i]);
  if(!std::strcmp(mode,"corrupt"))*reinterpret_cast<unsigned char*>(output)^=1;
  const auto*end=w+count-8;require(end[0]==0xc0064900);uint64_t marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);*reinterpret_cast<volatile unsigned*>(marker)=1;return 0;
 }
 int32_t kick()override{return 0;}
 uint64_t now_ns()override{return clock;}
 void sleep_us(unsigned n)override{clock+=uint64_t(n)*1000;}
};
static std::unique_ptr<ps5hip::SubmissionBackend> aggregate_backend(){++backend_calls;return std::make_unique<AggregateTransport>();}
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(aggregate_backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t*p,const pthread_attr_t*a,void*(*f)(void*),void*d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void**v){return __real_pthread_join(p,v);}
static void ok(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"AGGREGATE_HIP_ERROR status=%d\n",int(e));std::abort();}}
struct Final {
 ~Final(){require(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain);require(observed.allocations==1&&observed.frees==1&&observed.launches==2);require(observed.queues==observed.retired_queues&&allocations.empty());std::puts("PUBLIC_AGGREGATE_FINAL ownership=clean gpu_execution=false");}
};
__attribute__((init_priority(101))) static Final final;
int main(int argc,char**argv){
 require(argc==2);mode=argv[1];ok(hipInit(0));unsigned char*buffer=nullptr;ok(hipMalloc(&buffer,172));
 for(unsigned pass=0;pass<2;++pass){
  std::array<unsigned char,172> initial,output;initial.fill(0xa7);output.fill(0xde);ok(hipMemcpy(buffer,initial.data(),initial.size(),hipMemcpyHostToDevice));
  uint32_t before=0x12345678+pass,after=0xfedcba98-pass;uint3 v{0x11223344+pass,0x55667788,0x99aabbcc};Aggregate16 a{0x1020304050607080ull+pass,0xa1b2c3d4,0xe5f6,7,24};
  Softmax128 p;std::memset(&p,0,sizeof(p));p.nheads=17;p.n_head_log2=4;p.ncols=512;p.nrows_x=7;p.nrows_y=8;p.ne00=9;p.ne01=10;p.ne02=11;p.ne03=12;p.nb11=13;p.nb12=14;p.nb13=15;p.ne12=16;p.ne13=18;p.scale=0.5f+pass;p.max_bias=1.25f;p.m0=-2.5f;p.m1=3.75f;
  std::memcpy(expected_bytes.data(),&before,4);std::memcpy(expected_bytes.data()+4,&v,12);std::memcpy(expected_bytes.data()+16,&a,16);std::memcpy(expected_bytes.data()+32,&p,128);std::memcpy(expected_bytes.data()+160,&after,4);
  // Unsupported initial-route geometry/shared memory must reject synchronously.
  void*args[]={&buffer,&before,&v,&a,&p,&after};
  require(hipLaunchKernel(&aggregate_shapes,dim3(1),dim3(pass?64:32),args,pass?4:0,nullptr)==hipErrorInvalidValue);
  (void)hipGetLastError();require(submits==pass);
  {std::lock_guard<std::mutex> lock(mutex);entered=false;resume=false;}
  launch_aggregate_fixture(buffer,before,v,a,p,after);ok(hipGetLastError());
  {std::unique_lock<std::mutex> lock(mutex);require(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}));
   before=after=0;std::memset(&v,0,sizeof(v));std::memset(&a,0,sizeof(a));std::memset(&p,0xff,sizeof(p));
   require(observed.frees==0);resume=true;changed.notify_all();}
  ok(hipDeviceSynchronize());ok(hipMemcpy(output.data(),buffer,output.size(),hipMemcpyDeviceToHost));
  for(size_t i=0;i<output.size();++i){if(i>=44&&i<48)continue;unsigned char want=i<164?expected_bytes[i]:0xa7;
   if(output[i]!=want){std::fprintf(stderr,"PUBLIC_AGGREGATE_OUTPUT_MISMATCH index=%zu\n",i);std::abort();}}
 }
 require(submits==2&&backend_calls==1);ok(hipFree(buffer));completed=true;
 std::puts("PUBLIC_AGGREGATE_PASS compiler_stubs=true captures=2 invalid_calls=2 gap_zero=true caller_mutation=true guards=8 gpu_execution=false");
}
