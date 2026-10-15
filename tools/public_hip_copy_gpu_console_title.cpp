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
#include <unistd.h>

extern "C" int sceKernelDebugOutText(int channel, const char *text);

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
  if (!completed || observed.terminal || observed.uncertain || observed.launches != 84 ||
      observed.allocations != 4 || observed.frees != 4 || observed.queues != 7 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_COPY_GPU FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_COPY_GPU FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_COPY_GPU QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

static void checked(bool value,int line) {
  if(!value){std::fprintf(stderr,"HIP_COPY_GPU CHECK line=%d\n",line);ps5hip_clr_gpu_fail("oracle");}
}
static void ok(hipError_t status,int line) {
  if(status!=hipSuccess){std::fprintf(stderr,"HIP_COPY_GPU ERROR line=%d status=%d\n",line,int(status));ps5hip_clr_gpu_fail("public-api");}
}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_copy_gpu_fixture.h"
int main() {
  try {
    std::ifstream config("/app0/results-dir.txt", std::ios::binary);
    char config_bytes[242]{};
    config.read(config_bytes, sizeof(config_bytes));
    const auto length = config.gcount();
    if (!config.eof() || length < 2 || length > 240) return 2;
    const std::string directory(config_bytes, static_cast<size_t>(length));
    if (!ps5hip_canary::valid_results_directory(directory)) return 2;
    (void)sceKernelDebugOutText(0, "PS5HIP BOOT elevation-begin\n");
    const auto elevated = elevation::request(elevation::Capability::filesystem, "/app0/lapy.elf");
    char elevation_result[128]{};
    std::snprintf(elevation_result, sizeof(elevation_result),
                  "PS5HIP BOOT elevation-returned status=%d path=%s\n",
                  int(elevated), elevation::path());
    (void)sceKernelDebugOutText(0, elevation_result);
    if (elevated != elevation::Status::ok) return 2;
    (void)sceKernelDebugOutText(0, "PS5HIP BOOT log-open-begin\n");
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-copy_gpu.log").c_str())) return 2;
    (void)sceKernelDebugOutText(0, "PS5HIP BOOT log-ready\n");
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_copy_gpu");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    copy_gpu_fixture::run("/system_ex/app/PPSA99014/thread_words.hsaco",true,nullptr,true);
    if(observed.allocations!=2 || observed.frees!=2 || observed.queues!=4 ||
       observed.retired_queues!=3 || observed.terminal || observed.uncertain ||
       observed.launches!=4) ps5hip_clr_gpu_fail("public-retire");
    copy_gpu_fixture::run_replay("/system_ex/app/PPSA99014/thread_words.hsaco");
    if(observed.allocations!=4 || observed.frees!=4 || observed.queues!=7 ||
       observed.retired_queues!=6 || observed.terminal || observed.uncertain ||
       observed.launches!=84)ps5hip_clr_gpu_fail("replay-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed=true;
    std::puts("HIP_COPY_GPU PASS init=0 count=1 rounds=2 launches=84 values=292 guards=32 recovery_cases=27 setter_rejections=4 update_rejections=2 joined_streams=2 rejoined=true events_reused=true eager_equal=true captured=true owned_args=true updated=true source_destroyed=true disabled_replays=6 callbacks=28 allocations=4 frees=4 replay_rounds=20 warmup_rounds=4 verification_rounds=16 kernels_per_replay=2 replay_values=5840 replay_guards=640 alternating_order=true watchdog_joined=true finalization=pending");
    if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("result-log");
    return 0;
  }catch(...){ps5hip_clr_gpu_fail("exception");}
}
