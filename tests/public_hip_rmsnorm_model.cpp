// Unchanged compiled module -> public HIP -> CLR -> kernarg packer. GPU transport only.
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
static std::array<uint8_t,488> expected;
static unsigned expected_rows=0;
static bool completed=false;
static bool weighted=false;
static uint32_t expected_rsrc1=0x60af00c3;
static std::vector<uint8_t> expected_image;
static uint64_t expected_entry=0;
#define require(value) checked(bool(value),__LINE__)
class RmsTransport final:public ps5hip::SubmissionBackend {
 uint64_t clock=0;
public:
 void flush(void*,size_t)override{}
 int32_t submit(uint32_t* w,uint32_t count)override{
  ++submits;
  {std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();require(changed.wait_for(lock,std::chrono::seconds(5),[]{return resume;}));}
  std::map<unsigned,std::vector<unsigned>> regs;bool dispatched=false;
  for(size_t at=0;at<count;){const auto h=w[at];size_t n=((h>>16)&0x3fff)+1;require(h>>30==3&&n<count-at);
   if((h&0xffff)==0x7600)require(regs.emplace(w[at+1],std::vector<unsigned>(w+at+2,w+at+n+1)).second);
   if((h&0xffff)==0x1502){require(!dispatched&&n==4&&w[at+1]==expected_rows&&w[at+2]==1&&w[at+3]==1&&w[at+4]==0x8005);dispatched=true;}
   at+=n+1;
  }
  require(dispatched&&regs.at(0x212)==std::vector<unsigned>({expected_rsrc1,0x1038c}));
  require(regs.at(0x207)==std::vector<unsigned>({1024,1,1})&&regs.at(0x215)==std::vector<unsigned>({0x400000}));
  const auto& program=regs.at(0x20c);require(program.size()==2);
  const auto entry=(uint64_t(program[0])<<8)|(uint64_t(program[1])<<40);require(entry>=expected_entry);
  const auto code=entry-expected_entry;live(code,expected_image.size());
  require(std::any_of(allocations.begin(),allocations.end(),[&](const auto& allocation){return uint64_t(allocation.second.p)==code;}));
  require(std::equal(expected_image.begin(),expected_image.end(),reinterpret_cast<const uint8_t*>(code)));
  const auto& user=regs.at(0x240);auto arg=uint64_t(user.at(4))|(uint64_t(user.at(5))<<32);live(arg,488);
  auto* packed=reinterpret_cast<const uint8_t*>(arg);
  auto oracle=expected;if(!std::strcmp(mode,"mismatch"))oracle[352]^=1;
  if(!std::equal(oracle.begin(),oracle.end(),packed)){std::fputs("PUBLIC_RMSNORM_PACKET_MISMATCH\n",stderr);std::abort();}
  uint64_t a,b,mul,add;std::memcpy(&a,packed,8);std::memcpy(&b,packed+8,8);std::memcpy(&mul,packed+64,8);std::memcpy(&add,packed+144,8);
  live(a,17*2048*4);live(b,17*2048*4);require(!add);
  if(weighted){require(mul);live(mul,2048*4);}else require(!mul);
  // No numerical emulation: only validate emitted arguments/resources and completion.
  const auto*end=w+count-8;require(end[0]==0xc0064900);auto marker=uint64_t(end[3])|(uint64_t(end[4])<<32);live(marker,4);*reinterpret_cast<volatile unsigned*>(marker)=1;return 0;
 }
 int32_t kick()override{return 0;}uint64_t now_ns()override{return clock;}void sleep_us(unsigned n)override{clock+=uint64_t(n)*1000;}
};
static std::unique_ptr<ps5hip::SubmissionBackend> rms_backend(){++backend_calls;return std::make_unique<RmsTransport>();}
extern "C" bool ps5hip_clr_admit_device(){return ps5hip::clr::register_runtime_device(rms_backend,observed);}
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_create(pthread_t*p,const pthread_attr_t*a,void*(*f)(void*),void*d){return __real_pthread_create(p,a,f,d);}
extern "C" int __wrap_pthread_join(pthread_t p,void**v){return __real_pthread_join(p,v);}
static void ok(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"RMSNORM_HIP_ERROR status=%d\n",int(e));std::abort();}}
struct Final{~Final(){require(completed&&observed.gate_destroyed&&!observed.terminal&&!observed.uncertain);require(observed.allocations==(weighted?3u:2u)&&observed.frees==(weighted?3u:2u)&&observed.launches==3&&observed.queues==observed.retired_queues&&allocations.empty());std::puts("PUBLIC_RMSNORM_FINAL ownership=clean gpu_execution=false");}};
__attribute__((init_priority(101))) static Final final;
int main(int argc,char**argv){
 require(argc==4||(argc==5&&!std::strcmp(argv[4],"--weighted")));weighted=argc==5;expected_rsrc1=weighted?0x60af0184:0x60af00c3;mode=argv[2];ok(hipInit(0));
 std::ifstream in(argv[1],std::ios::binary);require(bool(in));std::vector<uint8_t> image{std::istreambuf_iterator<char>(in),{}};
 auto object=ps5hip::inspect_code_object(image.data(),image.size());require(object.kernels.size()==(weighted?1u:16u));
 const std::string symbol=std::string("_ZL12rms_norm_f32ILi1024ELb")+(weighted?"1":"0")+"ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f";const char*name=symbol.c_str();
 auto selected=std::find_if(object.kernels.begin(),object.kernels.end(),[&](const ps5hip::Kernel& k){return k.name==name;});require(selected!=object.kernels.end()&&selected->arguments.size()==26&&selected->kernarg_size==488&&selected->rsrc1==expected_rsrc1&&selected->rsrc2==0x38c);
 expected_image=ps5hip::materialize_code_object(image.data(),image.size());expected_entry=selected->entry_address;
 hipModule_t module=nullptr;ok(hipModuleLoad(&module,argv[1]));hipFunction_t function=nullptr;ok(hipModuleGetFunction(&function,module,name));int maximum=0;ok(hipFuncGetAttribute(&maximum,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function));require(maximum==1024);
 int shared_limit=0;ok(hipFuncGetAttribute(&shared_limit,HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,function));require(shared_limit==65536);
 int device_threads=0,device_shared=0;ok(hipDeviceGetAttribute(&device_threads,hipDeviceAttributeMaxThreadsPerBlock,0));ok(hipDeviceGetAttribute(&device_shared,hipDeviceAttributeMaxSharedMemoryPerBlock,0));require(device_threads==1024&&device_shared==65536);
 unsigned unsupported=0;
 for(const auto& k:object.kernels)if(!ps5hip::initial_launch_rejection(k).empty()&&!ps5hip::reduction_lds_launch_rejection(k).empty()){
  ++unsupported;for(unsigned repeat=0;repeat<2;++repeat){hipFunction_t absent=nullptr;require(hipModuleGetFunction(&absent,module,k.name.c_str())!=hipSuccess&&absent==nullptr);require(hipFuncGetAttribute(&maximum,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,absent)==hipErrorInvalidHandle);}
 }
 require(weighted?unsupported==0:unsupported>0);
 for(unsigned repeat=0;repeat<2;++repeat){hipFunction_t absent=nullptr;require(hipModuleGetFunction(&absent,module,"ps5hip_absent_rmsnorm")!=hipSuccess&&absent==nullptr);require(hipFuncGetAttribute(&maximum,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,absent)==hipErrorInvalidHandle);}
 auto denied=image;
 for(const auto& k:object.kernels){const auto address=k.descriptor_address+48;bool patched=false;for(const auto& segment:object.segments)if(address>=segment.address&&address-segment.address+4<=segment.file_size){const size_t offset=segment.file_offset+address-segment.address;denied.at(offset+2)|=0x10;patched=true;break;}require(patched);}
 const auto denied_object=ps5hip::inspect_code_object(denied.data(),denied.size());for(const auto& k:denied_object.kernels)require(!ps5hip::initial_launch_rejection(k).empty()&&!ps5hip::reduction_lds_launch_rejection(k).empty());
 auto write=[&](const char* suffix,const std::vector<uint8_t>& bytes){const auto path=std::string(argv[3])+"/"+suffix;std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());out.close();require(bool(out));return path;};
 const auto denied_path=write("all-unsupported.hsaco",denied);auto malformed=image;malformed[0]^=1;const auto malformed_path=write("malformed.hsaco",malformed);
 for(const auto& path:{denied_path,malformed_path})for(unsigned repeat=0;repeat<2;++repeat){hipModule_t absent=nullptr;require(hipModuleLoad(&absent,path.c_str())!=hipSuccess&&absent==nullptr);}
 std::vector<void*> buffers(weighted?3:2);for(size_t i=0;i<buffers.size();++i)ok(hipMalloc(&buffers[i],i==2?2048*4:17*2048*4));
 const unsigned offsets[]={0,8,16,20,24,32,40,48,56,64,72,80,88,96,108,120,132,144,152,160,168,176,188,200,212,224};
 const unsigned sizes[]={8,8,4,4,4,8,8,8,4,8,8,8,8,12,12,12,12,8,8,8,8,12,12,12,12,4};
 for(unsigned rows:{1u,2u,17u}){
  expected.fill(0);expected_rows=rows;auto put=[&](size_t at,uint64_t v,unsigned n){for(unsigned i=0;i<n;++i)expected[at+i]=uint8_t(v>>(8*i));};
  std::vector<std::vector<uint8_t>> host(26);std::vector<void*> args;
  for(unsigned i=0;i<26;++i){require(selected->arguments[i].offset==offsets[i]&&selected->arguments[i].size==sizes[i]);if(i<2)put(offsets[i],uint64_t(buffers[i]),8);else if(weighted&&i==9)put(offsets[i],uint64_t(buffers[2]),8);else if(i!=9&&i!=17){if(sizes[i]<=8)put(offsets[i],100+i+rows,sizes[i]);else for(unsigned b=0;b<12;++b)expected[offsets[i]+b]=uint8_t(i+b+rows);}host[i].assign(expected.begin()+offsets[i],expected.begin()+offsets[i]+sizes[i]);args.push_back(host[i].data());}
  put(232,rows,4);put(236,1,4);put(240,1,4);put(244,1024,2);put(246,1,2);put(248,1,2);put(296,1,2);put(352,128,4);
  const auto before_submits=submits;
  auto rejected=[&](hipError_t status){require(status==hipErrorInvalidValue);require(submits==before_submits&&!observed.terminal&&!observed.uncertain);(void)hipGetLastError();};
  for(unsigned block:{64u,1025u})rejected(hipModuleLaunchKernel(function,rows,1,1,block,1,1,128,nullptr,args.data(),nullptr));
  rejected(hipModuleLaunchKernel(function,rows,1,1,1024,2,1,128,nullptr,args.data(),nullptr));
  rejected(hipModuleLaunchKernel(function,rows,2,1,1024,1,1,128,nullptr,args.data(),nullptr));
  rejected(hipModuleLaunchKernel(function,rows,1,2,1024,1,1,128,nullptr,args.data(),nullptr));
  for(unsigned shared:{0u,65537u})rejected(hipModuleLaunchKernel(function,rows,1,1,1024,1,1,shared,nullptr,args.data(),nullptr));
  rejected(hipExtModuleLaunchKernel(function,rows*1024,1,1,1024,1,1,size_t(UINT32_MAX)+1,nullptr,args.data(),nullptr));
  if(weighted){const auto saved=host[9];host[9].assign(8,0);host[9][0]=1;args[9]=host[9].data();rejected(hipModuleLaunchKernel(function,rows,1,1,1024,1,1,128,nullptr,args.data(),nullptr));host[9]=saved;args[9]=host[9].data();}
  {std::lock_guard<std::mutex> lock(mutex);entered=false;resume=false;}
  ok(hipModuleLaunchKernel(function,rows,1,1,1024,1,1,128,nullptr,args.data(),nullptr));
  {std::unique_lock<std::mutex> lock(mutex);require(changed.wait_for(lock,std::chrono::seconds(5),[]{return entered;}));for(auto& h:host)std::fill(h.begin(),h.end(),0xee);resume=true;changed.notify_all();}
  ok(hipDeviceSynchronize());require(!observed.terminal&&!observed.uncertain);
 }
 require(submits==3&&backend_calls==1);for(auto b:buffers)ok(hipFree(b));ok(hipModuleUnload(module));completed=true;
 std::printf("PUBLIC_RMSNORM_PASS rows=1,2,17 arguments=26 kernargs=488 dynamic_lds=128 packed_nulls=%u caller_mutation=true unsupported_absent=true modules_rejected=true numerical_execution=false gpu_execution=false\n",weighted?1u:2u);
}
