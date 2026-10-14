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
void finalized() {

 if(!completed||!ps5hip_clr_gpu::passed())
  ps5hip_clr_gpu_fail("final-observation");
 std::puts("CLR_GPU_TITLE FINAL launches=1 exact_values=73 guards=8 allocations=2 frees=2 queues_joined=2 watchdog_joined=true application_fini=pending");
 if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("final-log");
}
}
extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
 std::fprintf(stderr,"CLR_GPU_TITLE QUARANTINED reason=%s no_shell_request=true\n",reason);
 (void)ps5hip_canary::sync_log();for(;;)::sleep(1);
}
extern "C" void ps5hip_install_vector_title_finalizer() {installed=std::atexit(finalized)==0;}
int main() {
 try {
  std::ifstream config("/app0/results-dir.txt",std::ios::binary);char bytes[242]{};config.read(bytes,sizeof(bytes));const auto n=config.gcount();
  if(!config.eof()||n<2||n>240)return 2;
  const std::string directory(bytes,static_cast<size_t>(n));if(!ps5hip_canary::valid_results_directory(directory))return 2;
  if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok)return 2;
  if(!ps5hip_canary::redirect_log((directory+"/ps5-hip-clr-gpu.log").c_str()))return 2;
  std::puts("PS5HIP START target=gfx1013 kernel=clr_gpu");
  if(!installed)ps5hip_clr_gpu_fail("finalizer-install");
  ps5hip_canary::check_runtime();
  // Exercise the public unsupported path, never an absent executable-memory provider.
  const size_t page=amd::Os::pageSize();if(!page||amd::Os::reserveMemory(nullptr,page,0,amd::Os::MEM_PROT_RWX)!=nullptr||amd::Os::commitMemory(nullptr,page,amd::Os::MEM_PROT_RWX)||amd::Os::protectMemory(nullptr,page,amd::Os::MEM_PROT_RWX))ps5hip_clr_gpu_fail("rwx-rejection");
  std::puts("CLR_GPU_TITLE RWX rejected=3");
  ps5hip_clr_gpu::watchdog_start(15000);ps5hip_clr_gpu::run();ps5hip_clr_gpu::watchdog_finish();
  completed=true;
  std::puts("CLR_GPU_TITLE PASS launches=1 exact_values=73 guards=8 allocations=2 frees=2 queues_joined=2 watchdog_joined=true finalization=pending");
  if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("result-log");
  return 0;
 } catch(...) {ps5hip_clr_gpu_fail("exception");}
}
