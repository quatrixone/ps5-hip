#include "clr_gemm/client.h"
#include "clr_gpu/client.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include "os/os.hpp"
#include <fstream>
#include <cstdlib>
#include <unistd.h>
namespace {
bool installed=false,completed=false;
void finalized(){
 if(!completed||!ps5hip_clr_gemm::passed())ps5hip_clr_gpu_fail("gemm-final-observation");
 std::puts("CLR_GEMM_TITLE FINAL launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 watchdog_joined=true application_fini=pending");
 if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("gemm-final-log");
}
}
extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason){
 std::fprintf(stderr,"CLR_GEMM_TITLE QUARANTINED reason=%s no_shell_request=true\n",reason);
 (void)ps5hip_canary::sync_log();for(;;)::sleep(1);
}
extern "C" void ps5hip_install_vector_title_finalizer(){installed=std::atexit(finalized)==0;}
int main(){
 try{
  std::ifstream config("/app0/results-dir.txt",std::ios::binary);char bytes[242]{};config.read(bytes,sizeof(bytes));const auto n=config.gcount();
  if(!config.eof()||n<2||n>240)return 2;
  const std::string directory(bytes,static_cast<size_t>(n));if(!ps5hip_canary::valid_results_directory(directory))return 2;
  if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok)return 2;
  if(!ps5hip_canary::redirect_log((directory+"/ps5-hip-clr-gemm.log").c_str()))return 2;
  std::puts("PS5HIP START target=gfx1013 kernel=clr_gemm");
  if(!installed)ps5hip_clr_gpu_fail("gemm-finalizer-install");
  ps5hip_canary::check_runtime();
  const size_t page=amd::Os::pageSize();
  if(!page||amd::Os::reserveMemory(nullptr,page,0,amd::Os::MEM_PROT_RWX)!=nullptr||amd::Os::commitMemory(nullptr,page,amd::Os::MEM_PROT_RWX)||amd::Os::protectMemory(nullptr,page,amd::Os::MEM_PROT_RWX))ps5hip_clr_gpu_fail("gemm-rwx-rejection");
  ps5hip_clr_gpu::watchdog_start(15000);ps5hip_clr_gemm::run();ps5hip_clr_gpu::watchdog_finish();
  completed=true;
  std::puts("CLR_GEMM_TITLE PASS launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 watchdog_joined=true finalization=pending");
  if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("gemm-result-log");
  return 0;
 }catch(...){ps5hip_clr_gpu_fail("gemm-exception");}
}
