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
      observed.launches != 5 || observed.allocations != 6 || observed.queues != 1 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_WEIGHTED_QKV FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_WEIGHTED_QKV FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_WEIGHTED_QKV QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}


namespace {
constexpr unsigned prefix=16,width=2048;
constexpr std::array<unsigned,6> capacities{8192,8192,8192,2304,2359296,8192};
constexpr std::array<const char*,6> names{"input","norm-weight","weighted","q8","projection-weight","output"};
constexpr std::array<const char*,5> stages{"weighted","q8","q","k","v"};
constexpr const char* title_root="/system_ex/app/PPSA99014/weighted-qkv/";
constexpr const char* symbols[]={
 "_ZL12rms_norm_f32ILi1024ELb1ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f",
 "_ZL13quantize_q8_1PKfPvlllllj15HIP_vector_typeIjLj3EE",
 "_ZL13mul_mat_vec_qIL9ggml_type2ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj"};
struct Uint3{uint32_t x,y,z;};
struct alignas(8) Fusion{uint8_t bytes[80];};
static_assert(sizeof(Uint3)==12&&sizeof(Fusion)==80&&sizeof(float)==4&&sizeof(double)==8,"kernel ABI");
// Fixed backing survives quarantine, including any uncertain pending transfers.
alignas(16) std::array<uint8_t,8224> input_seed{},input_read{},norm_seed{},norm_read{},weighted_seed{},weighted_read{},output_seed{},output_read{};
alignas(16) std::array<uint8_t,2336> q8_seed{},q8_read{};
alignas(16) std::array<uint8_t,2359328> weight_seed{},weight_read{};
std::array<double,2048> reference{},budgets{};
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
 if(status!=hipSuccess){std::fprintf(stderr,"HIP_WEIGHTED_QKV ERROR operation=%s status=%d\n",operation,int(status));ps5hip_clr_gpu_fail(operation);}
}

void finite_floats(const uint8_t* data,unsigned count){
 for(unsigned i=0;i<count;++i){float value;std::memcpy(&value,data+i*4,4);if(!std::isfinite(value))ps5hip_clr_gpu_fail("nonfinite-or-unwritten-float");}
}
void nan_seed(uint8_t* data,unsigned count){const uint32_t nan=0x7fc00000;for(unsigned i=0;i<count;++i)std::memcpy(data+i*4,&nan,4);}
void q8_domain(const uint8_t* data){
 for(unsigned block=0;block<64;++block){uint16_t d,s;std::memcpy(&d,data+block*36,2);std::memcpy(&s,data+block*36+2,2);
  if((d&0x8000)||(d&0x7c00)==0x7c00||(s&0x7c00)==0x7c00)ps5hip_clr_gpu_fail("q8-nonfinite-or-unwritten-scale");
  for(unsigned i=0;i<32;++i)if(data[block*36+4+i]==0x80)ps5hip_clr_gpu_fail("q8-integer-domain");
 }
}
}
int main(){try{
 std::ifstream config("/app0/results-dir.txt",std::ios::binary);char config_bytes[242]{};config.read(config_bytes,sizeof(config_bytes));const auto length=config.gcount();
 if(!config.eof()||length<2||length>240)return 2;
 const std::string directory(config_bytes,size_t(length));if(!ps5hip_canary::valid_results_directory(directory))return 2;
 if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok)return 2;
 if(!ps5hip_canary::redirect_log((directory+"/ps5-hip-public-weighted-qkv.log").c_str()))return 2;
 std::puts("PS5HIP START target=gfx1013 kernel=public_hip_weighted_qkv");
 if(!finalizer_installed)ps5hip_clr_gpu_fail("finalizer-install");
 ps5hip_canary::check_runtime();ps5hip_clr_gpu::watchdog_start(30000);
 hip_ok(hipInit(0),"initialize");int count=-1;hip_ok(hipGetDeviceCount(&count),"discovery");if(count!=1)ps5hip_clr_gpu_fail("device-count");
 hipModule_t modules[3]{};hipFunction_t functions[3]{};const char* module_names[]={"weighted.hsaco","q8.hsaco","q4.hsaco"};
 for(unsigned i=0;i<3;++i){hip_ok(hipModuleLoad(&modules[i],(std::string(title_root)+module_names[i]).c_str()),"module-load");hip_ok(hipModuleGetFunction(&functions[i],modules[i],symbols[i]),"module-function");if(!modules[i]||!functions[i])ps5hip_clr_gpu_fail("module-null");}
 std::array<void*,6> buffers{};
 const std::array<uint8_t*,6> seeds{input_seed.data(),norm_seed.data(),weighted_seed.data(),q8_seed.data(),weight_seed.data(),output_seed.data()};
 const std::array<uint8_t*,6> readbacks{input_read.data(),norm_read.data(),weighted_read.data(),q8_read.data(),weight_read.data(),output_read.data()};
 for(unsigned i=0;i<6;++i){std::memset(seeds[i],i<2||i==4?0xa5:0x5a,capacities[i]+32);hip_ok(hipMalloc(&buffers[i],capacities[i]+32),"allocate");if(!buffers[i])ps5hip_clr_gpu_fail("allocation-null");}
 if(observed.allocations!=6||observed.frees)ps5hip_clr_gpu_fail("allocation-count");
 read_exact(std::string(title_root)+"input.f32le",input_seed.data()+prefix,8192);
 read_exact(std::string(title_root)+"norm.f32le",norm_seed.data()+prefix,8192);
 read_exact(std::string(title_root)+"q.weights.q4_0",weight_seed.data()+prefix,2359296);
 read_exact(std::string(title_root)+"weighted.fp64le",reference.data(),16384);
 read_exact(std::string(title_root)+"weighted.budget.f64le",budgets.data(),16384);
 finite_floats(input_seed.data()+prefix,width);finite_floats(norm_seed.data()+prefix,width);
 for(unsigned i=0;i<width;++i)if(!std::isfinite(reference[i])||!std::isfinite(budgets[i])||budgets[i]<=0)ps5hip_clr_gpu_fail("reference-domain");
 nan_seed(weighted_seed.data()+prefix,width);nan_seed(output_seed.data()+prefix,width);std::memset(q8_seed.data()+prefix,0xcc,2304);
 for(unsigned i=0;i<6;++i)hip_ok(hipMemcpy(buffers[i],seeds[i],capacities[i]+32,hipMemcpyHostToDevice),"initial-upload");
 void* input=static_cast<uint8_t*>(buffers[0])+prefix;void* norm=static_cast<uint8_t*>(buffers[1])+prefix;
 void* weighted=static_cast<uint8_t*>(buffers[2])+prefix;void* q8=static_cast<uint8_t*>(buffers[3])+prefix;
 void* weights=static_cast<uint8_t*>(buffers[4])+prefix;void* output=static_cast<uint8_t*>(buffers[5])+prefix;
 unsigned launches=0,dumps=0,projection_values=0;size_t raw_bytes=0;double max_error=0,max_fraction=0;
 for(unsigned stage=0;stage<5;++stage){
  const unsigned rows=stage<=2?2048:256;
  if(stage>=2){
   output_seed.fill(0x5a);nan_seed(output_seed.data()+prefix,rows);
   hip_ok(hipMemcpy(buffers[5],output_seed.data(),output_seed.size(),hipMemcpyHostToDevice),"projection-output-reset");
   if(stage>2){weight_seed.fill(0xa5);read_exact(std::string(title_root)+stages[stage]+".weights.q4_0",weight_seed.data()+prefix,rows*1152);hip_ok(hipMemcpy(buffers[4],weight_seed.data(),weight_seed.size(),hipMemcpyHostToDevice),"projection-weight-upload");}
  }
  Uint3 one{1,0,1};
  if(stage==0){
   int ncols=2048,nchannels=1,nsamples=1;int64_t stride=2048,zero_stride=0;float epsilon=1e-5f,scale=1.0f;void* absent=nullptr;Uint3 columns{1,11,2048},zero{};
   void* args[]={&input,&weighted,&ncols,&nchannels,&nsamples,&stride,&stride,&stride,&epsilon,
    &norm,&stride,&stride,&stride,&columns,&one,&one,&one,
    &absent,&zero_stride,&zero_stride,&zero_stride,&zero,&zero,&zero,&zero,&scale};
   static_assert(sizeof(args)/sizeof(args[0])==26,"weighted arguments");
   hip_ok(hipModuleLaunchKernel(functions[0],1,1,1,1024,1,1,128,nullptr,args,nullptr),"launch-weighted");
  }else if(stage==1){
   int64_t n=2048;uint32_t ne1=1;void* args[]={&weighted,&q8,&n,&n,&n,&n,&n,&ne1,&one};
   static_assert(sizeof(args)/sizeof(args[0])==9,"Q8 arguments");
   hip_ok(hipModuleLaunchKernel(functions[1],8,1,1,256,1,1,0,nullptr,args,nullptr),"launch-q8");
  }else{
   void* ids=nullptr;Fusion fusion{};uint32_t n=2048,row_blocks=64,matrix_blocks=rows*64,ids_stride=0,output_rows=rows;
   void* args[]={&weights,&q8,&ids,&fusion,&output,&n,&one,&row_blocks,&row_blocks,&output_rows,&one,&matrix_blocks,&row_blocks,&output_rows,&one,&matrix_blocks,&row_blocks,&output_rows,&ids_stride};
   static_assert(sizeof(args)/sizeof(args[0])==19,"Q4 arguments");
   hip_ok(hipModuleLaunchKernel(functions[2],rows,1,1,32,4,1,0,nullptr,args,nullptr),"launch-projection");
  }
  hip_ok(hipDeviceSynchronize(),"synchronize");
  if(observed.launches!=launches+1||observed.terminal||observed.uncertain)ps5hip_clr_gpu_fail("launch-observation");
  for(unsigned i=0;i<6;++i){hip_ok(hipMemcpy(readbacks[i],buffers[i],capacities[i]+32,hipMemcpyDeviceToHost),"readback");dump_exclusive(directory+"/weighted-qkv-"+stages[stage]+"."+names[i]+".raw",readbacks[i],capacities[i]+32);++dumps;raw_bytes+=capacities[i]+32;}
  std::printf("HIP_WEIGHTED_QKV DUMP stage=%s files=6 bytes=2394560 exclusive=true synced=true\n",stages[stage]);if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("dump-log");
  const unsigned active=stage==0?2:stage==1?3:5,active_bytes=stage==1?2304:stage==0?8192:rows*4;
  for(unsigned i=0;i<6;++i)for(unsigned b=0;b<capacities[i]+32;++b)
   if((i!=active||b<prefix||b>=prefix+active_bytes)&&readbacks[i][b]!=seeds[i][b])ps5hip_clr_gpu_fail("unchanged-guard-inactive");
  if(stage==0){
   finite_floats(weighted_read.data()+prefix,width);
   for(unsigned i=0;i<width;++i){float value;std::memcpy(&value,weighted_read.data()+prefix+i*4,4);const double error=std::fabs(double(value)-reference[i]);if(error>budgets[i])ps5hip_clr_gpu_fail("weighted-numerical-budget");if(error>max_error)max_error=error;const double fraction=error/budgets[i];if(fraction>max_fraction)max_fraction=fraction;}
   // Host snapshots are comparison baselines only, never re-uploaded to the GPU.
   weighted_seed=weighted_read;
  }else if(stage==1){q8_domain(q8_read.data()+prefix);q8_seed=q8_read;}
  else{finite_floats(output_read.data()+prefix,rows);projection_values+=rows;}
  ++launches;
  std::printf("HIP_WEIGHTED_QKV STAGE stage=%s weighted_values=%u q8_bytes=%u projection_values=%u unchanged=true guards=true inactive=true shared_intermediates=true numerical_verification=pending\n",stages[stage],stage==0?2048:0,stage==1?2304:0,stage>=2?rows:0);
  if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("stage-log");
 }
 if(launches!=5||projection_values!=2560||dumps!=30||raw_bytes!=11972800||observed.terminal||observed.uncertain)ps5hip_clr_gpu_fail("totals");
 for(auto buffer:buffers)hip_ok(hipFree(buffer),"free");for(auto module:modules)hip_ok(hipModuleUnload(module),"module-unload");
 if(observed.allocations!=6||observed.frees!=6||observed.launches!=5||observed.queues!=1)ps5hip_clr_gpu_fail("retirement");
 ps5hip_clr_gpu::watchdog_finish();completed=true;
 std::printf("HIP_WEIGHTED_QKV TRANSPORT_COMPLETE stages=5 launches=5 weighted_values=2048 projection_values=2560 q8_bytes=2304 raw_files=30 raw_bytes=11972800 weighted_max_abs=%.9g weighted_max_budget_fraction=%.9g allocations=6 frees=6 modules_unloaded=3 watchdog_joined=true numerical_verification=pending finalization=pending\n",max_error,max_fraction);
 if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("result-log");return 0;
}catch(...){ps5hip_clr_gpu_fail("exception");}}
