#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include <hip/hip_runtime_api.h>
#include <ps5platform/libc.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

extern "C" int __cxa_thread_atexit_impl(void (*destructor)(void*), void* object,
                                          void* dso) {
  return ps5___cxa_thread_atexit_impl(destructor, object, dso);
}

namespace {
ps5hip::clr::Observation observed;
bool finalizer_installed = false;
bool completed = false;
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  return std::make_unique<ps5hip::AgcBackend>();
}
void finalized() {
  if (!completed || observed.terminal || observed.uncertain ||
      observed.launches != 28 || observed.allocations != 4 || observed.queues != 1 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_Q4 FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_Q4 FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_Q4 QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

namespace {
constexpr unsigned prefix=16;
constexpr std::array<unsigned,4> capacities{5632*4,256*64*18,176*36,256*4};
constexpr std::array<const char*,4> buffer_names{"input","weights","q8","output"};
constexpr const char* title_root="/system_ex/app/PPSA99014/q4/";
constexpr const char* symbols[]={
 "_ZL13quantize_q8_1PKfPvlllllj15HIP_vector_typeIjLj3EE",
 "_ZL13mul_mat_vec_qIL9ggml_type2ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj"};
struct Case {const char* name;unsigned width,rows,padded;};
constexpr Case cases[]={
 {"q8-zero",256,0,256},{"q8-constant",256,0,256},{"q8-signed-ties",256,0,256},{"q8-padding",255,0,256},
 {"attn-k-1",2048,1,2048},{"attn-k-2",2048,2,2048},{"attn-k-17",2048,17,2048},{"attn-k-256",2048,256,2048},
 {"ffn-down-1",5632,1,5632},{"ffn-down-17",5632,17,5632},{"synthetic-zero",2048,1,2048},{"synthetic-pattern",2048,1,2048}};
struct Uint3 {uint32_t x,y,z;};
struct alignas(8) Fusion {uint8_t bytes[80];};
static_assert(sizeof(Uint3)==12 && sizeof(Fusion)==80 && sizeof(float)==4 && sizeof(double)==8,"kernel ABI");
// Fixed full-capacity host storage remains alive on quarantine, including pending transfers.
alignas(16) std::array<uint8_t,22560> input_seed{},input_readback{};
alignas(16) std::array<uint8_t,294944> weights_seed{},weights_readback{};
alignas(16) std::array<uint8_t,6368> q8_seed{},q8_readback{},q8_expected{};
alignas(16) std::array<uint8_t,1056> output_seed{},output_readback{};
std::array<double,256> references{},budgets{};
std::array<float,256> lanes{};
void read_exact(const std::string& path,void* data,size_t bytes){
 std::ifstream f(path,std::ios::binary);
 if(!f||!f.read(static_cast<char*>(data),bytes)||f.peek()!=std::char_traits<char>::eof()||f.bad())ps5hip_clr_gpu_fail("asset-size-or-read");
}
void dump_exclusive(const std::string& path,const uint8_t* data,size_t bytes){
 const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
 if(fd<0)ps5hip_clr_gpu_fail("raw-dump-open");
 size_t written=0;unsigned interruptions=0;bool ok=true;
 while(written<bytes){const auto count=::write(fd,data+written,bytes-written);
  if(count<0&&errno==EINTR&&++interruptions<=8)continue;
  if(count<=0||size_t(count)>bytes-written){ok=false;break;}written+=size_t(count);
 }
 const bool synced=::fsync(fd)==0;const bool closed=::close(fd)==0;
 if(!ok||!synced||!closed)ps5hip_clr_gpu_fail("raw-dump-write-sync-close");
}
void hip_ok(hipError_t status,const char* operation){
 if(status!=hipSuccess){std::fprintf(stderr,"HIP_Q4 ERROR operation=%s status=%d\n",operation,int(status));ps5hip_clr_gpu_fail(operation);}
}
double absolute(double value){return value<0?-value:value;}
}
int main(){try{
 std::ifstream config("/app0/results-dir.txt",std::ios::binary);char bytes[242]{};config.read(bytes,sizeof(bytes));const auto length=config.gcount();
 if(!config.eof()||length<2||length>240)return 2;
 const std::string directory(bytes,size_t(length));if(!ps5hip_canary::valid_results_directory(directory))return 2;
 if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok)return 2;
 if(!ps5hip_canary::redirect_log((directory+"/ps5-hip-public-q4.log").c_str()))return 2;
 std::puts("PS5HIP START target=gfx1013 kernel=public_hip_q4");
 if(!finalizer_installed)ps5hip_clr_gpu_fail("finalizer-install");
 ps5hip_canary::check_runtime();ps5hip_clr_gpu::watchdog_start(30000);
 hip_ok(hipInit(0),"initialize");int count=-1;hip_ok(hipGetDeviceCount(&count),"discovery");if(count!=1)ps5hip_clr_gpu_fail("device-count");
 hipModule_t modules[2]{};hipFunction_t functions[2]{};
 for(unsigned i=0;i<2;++i){
  const auto path=std::string(title_root)+(i?"q4.hsaco":"q8.hsaco");
  hip_ok(hipModuleLoad(&modules[i],path.c_str()),"module-load");
  hip_ok(hipModuleGetFunction(&functions[i],modules[i],symbols[i]),"module-function");
  if(!modules[i]||!functions[i])ps5hip_clr_gpu_fail("module-null");
 }
 std::array<void*,4> buffers{};
 for(unsigned i=0;i<4;++i){hip_ok(hipMalloc(&buffers[i],capacities[i]+32),"allocate");if(!buffers[i])ps5hip_clr_gpu_fail("allocation-null");}
 if(observed.allocations!=4||observed.frees)ps5hip_clr_gpu_fail("allocation-count");
 const std::array<uint8_t*,4> seeds{input_seed.data(),weights_seed.data(),q8_seed.data(),output_seed.data()};
 const std::array<uint8_t*,4> readbacks{input_readback.data(),weights_readback.data(),q8_readback.data(),output_readback.data()};
 unsigned launches=0,values=0,q8_bytes=0,dumps=0;size_t raw_bytes=0;double max_error=0,max_fraction=0;
 for(const auto& c:cases){
  const auto asset=std::string(title_root)+c.name+"/";
  const unsigned quant_bytes=c.padded/32*36,weight_bytes=c.rows*(c.width/32)*18;
  q8_expected.fill(0x5a);read_exact(asset+"q8.bin",q8_expected.data()+prefix,quant_bytes);
  if(c.rows){read_exact(asset+"fp64le",references.data(),c.rows*8);read_exact(asset+"budget.f64le",budgets.data(),c.rows*8);read_exact(asset+"lane.f32le",lanes.data(),c.rows*4);
   for(unsigned r=0;r<c.rows;++r)if(!std::isfinite(references[r])||!std::isfinite(budgets[r])||budgets[r]<0||!std::isfinite(lanes[r])||absolute(double(lanes[r])-references[r])>budgets[r])ps5hip_clr_gpu_fail("reference-domain");
  }
  for(unsigned index=0;index<(c.rows?3u:1u);++index){
   const char* stage=!c.rows||index==1?"q8":index==0?"frozen":"chained";
   const bool quantize=!c.rows||index==1,chained=c.rows&&index==2;
   input_seed.fill(0xa5);weights_seed.fill(0xa5);output_seed.fill(0x5a);q8_seed=q8_expected;
   read_exact(asset+"input.f32le",input_seed.data()+prefix,c.width*4);
   for(unsigned i=0;i<c.width;++i){float value;std::memcpy(&value,input_seed.data()+prefix+i*4,4);if(!std::isfinite(value))ps5hip_clr_gpu_fail("input-domain");}
   if(c.rows){read_exact(asset+"weights.q4_0",weights_seed.data()+prefix,weight_bytes);const uint32_t nan=0x7fc00000;for(unsigned r=0;r<c.rows;++r)std::memcpy(output_seed.data()+prefix+r*4,&nan,4);}
   if(quantize)std::memset(q8_seed.data()+prefix,0xcc,quant_bytes);
   for(unsigned i=0;i<4;++i){
    // The chained Q4 consumes the prior kernel's Q8 allocation without any re-upload.
    if(i==2&&chained)continue;
    hip_ok(hipMemcpy(buffers[i],seeds[i],capacities[i]+32,hipMemcpyHostToDevice),"upload");
   }
   void* input=static_cast<uint8_t*>(buffers[0])+prefix;
   void* weights=static_cast<uint8_t*>(buffers[1])+prefix;
   void* q8=static_cast<uint8_t*>(buffers[2])+prefix;
   void* output=static_cast<uint8_t*>(buffers[3])+prefix;
   Uint3 one{1,0,1};
   if(quantize){
    int64_t width=c.width,padded=c.padded,stride=c.width;uint32_t ne1=1;
    void* args[]={&input,&q8,&width,&stride,&stride,&stride,&padded,&ne1,&one};
    static_assert(sizeof(args)/sizeof(args[0])==9,"Q8 arguments");
    hip_ok(hipModuleLaunchKernel(functions[0],(c.padded+255)/256,1,1,256,1,1,0,nullptr,args,nullptr),"launch-q8");
   }else{
    void* ids=nullptr;Fusion fusion{};uint32_t width=c.width,row_blocks=c.width/32,rows=c.rows,matrix_blocks=rows*row_blocks,ids_stride=0;
    void* args[]={&weights,&q8,&ids,&fusion,&output,&width,&one,&row_blocks,&row_blocks,&rows,&one,&matrix_blocks,&row_blocks,&rows,&one,&matrix_blocks,&row_blocks,&rows,&ids_stride};
    static_assert(sizeof(args)/sizeof(args[0])==19,"Q4 arguments");
    hip_ok(hipModuleLaunchKernel(functions[1],rows,1,1,32,4,1,0,nullptr,args,nullptr),"launch-q4");
   }
   hip_ok(hipDeviceSynchronize(),"synchronize");
   if(observed.launches!=launches+1||observed.terminal||observed.uncertain)ps5hip_clr_gpu_fail("launch-observation");
   for(unsigned i=0;i<4;++i){hip_ok(hipMemcpy(readbacks[i],buffers[i],capacities[i]+32,hipMemcpyDeviceToHost),"readback");dump_exclusive(directory+"/q4-"+c.name+"."+stage+"."+buffer_names[i]+".raw",readbacks[i],capacities[i]+32);++dumps;raw_bytes+=capacities[i]+32;}
   std::printf("HIP_Q4 DUMP case=%s stage=%s files=4 bytes=324928 exclusive=true synced=true\n",c.name,stage);
   if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("dump-log");
   if(std::memcmp(input_seed.data(),input_readback.data(),input_seed.size())||std::memcmp(weights_seed.data(),weights_readback.data(),weights_seed.size())||std::memcmp(q8_expected.data(),q8_readback.data(),q8_expected.size()))ps5hip_clr_gpu_fail("input-quantization-guard-inactive");
   double error_max=0,fraction_max=0;
   if(quantize){if(std::memcmp(output_seed.data(),output_readback.data(),output_seed.size()))ps5hip_clr_gpu_fail("q8-mutated-output");}
   else{
    for(unsigned i=0;i<output_seed.size();++i)if((i<prefix||i>=prefix+c.rows*4)&&output_seed[i]!=output_readback[i])ps5hip_clr_gpu_fail("output-guard-inactive");
    for(unsigned r=0;r<c.rows;++r){float actual;std::memcpy(&actual,output_readback.data()+prefix+r*4,4);const double error=absolute(double(actual)-references[r]);
     if(!std::isfinite(actual)||error>budgets[r]){std::fprintf(stderr,"HIP_Q4 MISMATCH case=%s stage=%s row=%u actual=%.9g reference=%.17g error=%.17g budget=%.17g\n",c.name,stage,r,double(actual),references[r],error,budgets[r]);ps5hip_clr_gpu_fail("numerical-oracle");}
     const double fraction=budgets[r]?error/budgets[r]:0;if(error>error_max)error_max=error;if(fraction>fraction_max)fraction_max=fraction;
    }values+=c.rows;
   }
   if(error_max>max_error)max_error=error_max;if(fraction_max>max_fraction)max_fraction=fraction_max;
   q8_bytes+=quant_bytes;++launches;
   std::printf("HIP_Q4 CASE case=%s stage=%s rows=%u values=%u q8_bytes=%u max_abs=%.9g max_budget_fraction=%.9g inputs_unchanged=true guards=true inactive=true\n",c.name,stage,c.rows,quantize?0:c.rows,quant_bytes,error_max,fraction_max);
   if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("case-log");
  }
 }
 if(launches!=28||values!=592||q8_bytes!=80640||dumps!=112||raw_bytes!=9097984||observed.terminal||observed.uncertain)ps5hip_clr_gpu_fail("totals");
 for(auto buffer:buffers)hip_ok(hipFree(buffer),"free");for(auto module:modules)hip_ok(hipModuleUnload(module),"module-unload");
 if(observed.allocations!=4||observed.frees!=4||observed.launches!=28||observed.queues!=1)ps5hip_clr_gpu_fail("retire");
 ps5hip_clr_gpu::watchdog_finish();completed=true;
 std::printf("HIP_Q4 PASS cases=12 stages=28 launches=28 values=592 q8_bytes=80640 raw_files=112 raw_bytes=9097984 max_abs=%.9g max_budget_fraction=%.9g allocations=4 frees=4 modules_unloaded=2 watchdog_joined=true finalization=pending\n",max_error,max_fraction);
 if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("result-log");return 0;
}catch(...){ps5hip_clr_gpu_fail("exception");}}
