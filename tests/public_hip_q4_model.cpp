// Genuine public HIP/CLR; modeled memory and packet completion, no numerical GPU emulation.
#define main unused_adapter_main
#define Transport UnusedTransport
#define backend unused_backend
#include "clr_adapter_model.cpp"
#undef backend
#undef Transport
#undef main
#undef check
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "ps5hip/code_object.h"
#include <array>
static ps5hip::clr::Observation observed;
static std::vector<uint8_t> expected,expected_image;
static uint64_t expected_entry=0;
static unsigned expected_rsrc1=0;
static std::array<unsigned,3> expected_threads;
static unsigned expected_groups=0,current_kind=0;
static bool completed=false;
#define require(v) checked(bool(v),__LINE__)
class QTransport final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{}
 int32_t submit(uint32_t* words,uint32_t count)override{
  ++submits;
  {std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();require(changed.wait_for(lock,std::chrono::seconds(5),[]{return resume;}));}
  require(allocations.size()==8); // Four user allocations, code, kernargs, command buffer and completion marker.
  live(reinterpret_cast<uint64_t>(words),size_t(count)*4);
  std::map<unsigned,std::vector<unsigned>> regs;bool dispatched=false;
  for(size_t at=0;at<count;){const auto h=words[at];const size_t n=((h>>16)&0x3fff)+1;require(h>>30==3&&n<count-at);
   if((h&0xffff)==0x7600)require(regs.emplace(words[at+1],std::vector<unsigned>(words+at+2,words+at+n+1)).second);
   if((h&0xffff)==0x1502){require(!dispatched&&n==4&&words[at+1]==expected_groups&&words[at+2]==1&&words[at+3]==1&&words[at+4]==0x8005);dispatched=true;}
   at+=n+1;
  }
  require(dispatched&&regs.at(0x212)==std::vector<unsigned>({expected_rsrc1,current_kind?0x10b8cu:0x38cu}));
  require(regs.at(0x207)==std::vector<unsigned>(expected_threads.begin(),expected_threads.end()));
  require(regs.at(0x215)==std::vector<unsigned>({0x400000}));
  const auto& program=regs.at(0x20c);require(program.size()==2);
  const auto entry=(uint64_t(program[0])<<8)|(uint64_t(program[1])<<40);require(entry>=expected_entry);
  const auto base=entry-expected_entry;live(base,expected_image.size());
  require(std::any_of(allocations.begin(),allocations.end(),[&](const auto& a){return uint64_t(a.second.p)==base;}));
  require(std::equal(expected_image.begin(),expected_image.end(),reinterpret_cast<const uint8_t*>(base)));
  const auto& user=regs.at(0x240);require(user.size()==6&&std::all_of(user.begin(),user.begin()+4,[](unsigned v){return v==0;}));
  const auto arg=uint64_t(user[4])|(uint64_t(user[5])<<32);live(arg,expected.size());
  auto oracle=expected;if(!std::strcmp(mode,"mismatch")&&current_kind)oracle[24]^=1;
  if(!std::equal(oracle.begin(),oracle.end(),reinterpret_cast<const uint8_t*>(arg))){std::fputs("PUBLIC_Q4_PACKET_MISMATCH\n",stderr);std::abort();}
  const auto* end=words+count-8;require(end[0]==0xc0064900);auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);*reinterpret_cast<volatile unsigned*>(marker)=1;
  return 0;
 }
 int32_t kick()override{return 0;}uint64_t now_ns()override{return clock;}void sleep_us(unsigned n)override{clock+=uint64_t(n)*1000;}
};
static std::unique_ptr<ps5hip::SubmissionBackend> q_backend(){++backend_calls;return std::make_unique<QTransport>();}
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(q_backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t*p,const pthread_attr_t*a,void*(*f)(void*),void*d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void**v){return __real_pthread_join(p,v);}
static void hip_ok(hipError_t e,int line){if(e!=hipSuccess){std::fprintf(stderr,"Q4_HIP_ERROR status=%d line=%d\n",int(e),line);std::abort();}}
#define ok(e) hip_ok((e),__LINE__)
struct Final{~Final(){require(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain);require(observed.allocations==4&&observed.frees==4&&observed.launches==4&&observed.queues==observed.retired_queues&&allocations.empty());std::puts("PUBLIC_Q4_FINAL ownership=clean gpu_execution=false");}};
__attribute__((init_priority(101))) static Final final;
int main(int argc,char**argv){
 require(argc==4);mode=argv[3];ok(hipInit(0));
 const char* names[]={"_ZL13quantize_q8_1PKfPvlllllj15HIP_vector_typeIjLj3EE","_ZL13mul_mat_vec_qIL9ggml_type2ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj"};
 std::array<std::vector<uint8_t>,2> images;std::array<ps5hip::CodeObject,2> objects;
 hipModule_t modules[2]{};hipFunction_t functions[2]{};
 for(unsigned kind=0;kind<2;++kind){
  std::ifstream in(argv[kind+1],std::ios::binary);require(bool(in));images[kind]={std::istreambuf_iterator<char>(in),{}};
  objects[kind]=ps5hip::inspect_code_object(images[kind].data(),images[kind].size());require(objects[kind].kernels.size()==1&&objects[kind].kernels[0].name==names[kind]);
  ok(hipModuleLoad(&modules[kind],argv[kind+1]));ok(hipModuleGetFunction(&functions[kind],modules[kind],names[kind]));
  for(unsigned i=0;i<2;++i){hipFunction_t absent=nullptr;require(hipModuleGetFunction(&absent,modules[kind],names[1-kind])!=hipSuccess&&!absent);}
  int value=0;ok(hipFuncGetAttribute(&value,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,functions[kind]));require(value==(kind?128:256));
  ok(hipFuncGetAttribute(&value,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,functions[kind]));require(value==(kind?384:0));
  ok(hipFuncGetAttribute(&value,HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,functions[kind]));require(value==0);
 }
 int ycap=0;ok(hipDeviceGetAttribute(&ycap,hipDeviceAttributeMaxBlockDimY,0));require(ycap==4);
 void* buffers[4]{};const size_t extents[]={22528,6336,15840,20};
 for(unsigned i=0;i<4;++i)ok(hipMalloc(&buffers[i],extents[i]));
 const std::vector<unsigned> offsets[2]={{0,8,16,24,32,40,48,56,60},{0,8,16,24,104,112,116,128,132,136,140,152,156,160,164,176,180,184,188}};
 const std::vector<unsigned> sizes[2]={{8,8,8,8,8,8,8,4,12},{8,8,8,80,8,4,12,4,4,4,12,4,4,4,12,4,4,4,4}};
 for(unsigned round=0;round<2;++round)for(unsigned kind=0;kind<2;++kind){
  current_kind=kind;const unsigned width=round?5632:2048,rows=round?5:3;
  expected.assign(kind?192:328,0);expected_threads=kind?std::array<unsigned,3>{32,4,1}:std::array<unsigned,3>{256,1,1};expected_groups=kind?rows:width/256;
  const auto& kernel=objects[kind].kernels[0];
  require(kind?(kernel.rsrc1==0x60af00c3u||kernel.rsrc1==0x60af00c4u):kernel.rsrc1==0x60af00c1u);expected_rsrc1=kernel.rsrc1;expected_image=ps5hip::materialize_code_object(images[kind].data(),images[kind].size());expected_entry=kernel.entry_address;
  auto put=[&](size_t at,uint64_t value,unsigned n){for(unsigned i=0;i<n;++i)expected[at+i]=uint8_t(value>>(8*i));};
  auto fastdiv=[&](unsigned at){put(at,1,4);put(at+4,0,4);put(at+8,1,4);};
  if(!kind){put(0,uint64_t(buffers[0]),8);put(8,uint64_t(buffers[1]),8);for(unsigned at:{16u,24u,32u,40u,48u})put(at,width,8);put(56,1,4);fastdiv(60);put(72,expected_groups,4);put(76,1,4);put(80,1,4);put(84,256,2);put(86,1,2);put(88,1,2);put(136,1,2);}
  else{put(0,uint64_t(buffers[2]),8);put(8,uint64_t(buffers[1]),8);put(104,uint64_t(buffers[3]),8);put(112,width,4);put(128,width/32,4);put(132,width/32,4);put(136,rows,4);fastdiv(140);put(152,rows*width/32,4);put(156,width/32,4);put(160,rows,4);fastdiv(164);put(176,rows*width/32,4);put(180,width/32,4);put(184,rows,4);}
  std::vector<std::vector<uint8_t>> host;std::vector<void*> args;require(kernel.arguments.size()==offsets[kind].size());
  for(unsigned i=0;i<offsets[kind].size();++i){const auto o=offsets[kind][i],s=sizes[kind][i];require(kernel.arguments[i].offset==o&&kernel.arguments[i].size==s);host.emplace_back(expected.begin()+o,expected.begin()+o+s);}
  for(auto& row:host)args.push_back(row.data());
  auto launch=[&](unsigned x,unsigned y,unsigned z,unsigned shared,unsigned groups_z=1){return hipModuleLaunchKernel(functions[kind],expected_groups,1,groups_z,x,y,z,shared,nullptr,args.data(),nullptr);};
  const auto before=submits;auto reject=[&](hipError_t e){require(e==hipErrorInvalidValue&&submits==before&&!observed.terminal&&!observed.uncertain);(void)hipGetLastError();};
  if(!kind){reject(launch(64,1,1,0));reject(launch(512,1,1,0));reject(launch(256,2,1,0));}
  else{reject(launch(128,1,1,0));reject(launch(32,2,1,0));reject(launch(32,8,1,0));}
  reject(launch(expected_threads[0],expected_threads[1],2,0));reject(launch(expected_threads[0],expected_threads[1],1,1));reject(launch(expected_threads[0],expected_threads[1],1,0,2));
  reject(hipExtModuleLaunchKernel(functions[kind],expected_groups*expected_threads[0],expected_threads[1],1,expected_threads[0],expected_threads[1],1,size_t(UINT32_MAX)+1,nullptr,args.data(),nullptr));
  if(kind)reject(hipExtModuleLaunchKernel(functions[kind],0xffffffe0u,4,1,32,4,1,0,nullptr,args.data(),nullptr));
  {std::lock_guard<std::mutex> lock(mutex);entered=false;resume=false;}
  ok(launch(expected_threads[0],expected_threads[1],1,0));
  {std::unique_lock<std::mutex> lock(mutex);require(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}));for(auto& row:host)std::fill(row.begin(),row.end(),0xee);resume=true;changed.notify_all();}
  ok(hipDeviceSynchronize());require(allocations.size()==4&&!observed.terminal&&!observed.uncertain);
 }
 require(submits==4&&backend_calls==1);for(auto buffer:buffers)ok(hipFree(buffer));for(auto module:modules)ok(hipModuleUnload(module));completed=true;
 std::puts("PUBLIC_Q4_PASS modules=2 launches=4 arguments=9,19 kernargs=328,192 threads=256x1,32x4 static_lds=0,384 caller_mutation=true rejects=30 numerical_execution=false gpu_execution=false");
}
