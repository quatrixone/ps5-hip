#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include <hip/hip_runtime_api.h>
#include <ps5platform/libc.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
  if (!completed || observed.terminal ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_MEMORY FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_MEMORY FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_MEMORY QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

int main() {
  try {
    std::ifstream config("/app0/results-dir.txt", std::ios::binary);
    char bytes[242]{};
    config.read(bytes, sizeof(bytes));
    const auto length = config.gcount();
    if (!config.eof() || length < 2 || length > 240) return 2;
    const std::string directory(bytes, static_cast<size_t>(length));
    if (!ps5hip_canary::valid_results_directory(directory)) return 2;
    if (elevation::request(elevation::Capability::filesystem, "/app0/lapy.elf") != elevation::Status::ok) return 2;
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-memory.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_memory");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    void* first = nullptr;
    void* second = nullptr;
    if (hipMalloc(&first, 257) != hipSuccess || !first ||
        hipMalloc(&second, 4097) != hipSuccess || !second || first == second ||
        observed.allocations != 2 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    if (hipFree(static_cast<char*>(first) + 1) == hipSuccess ||
        observed.allocations != 2 || observed.frees != 0)
      ps5hip_clr_gpu_fail("interior-free");
    if (hipFree(second) != hipSuccess || hipFree(first) != hipSuccess ||
        observed.allocations != 2 || observed.frees != 2 ||
        hipFree(first) == hipSuccess)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_MEMORY PASS init=0 count=1 allocations=2 frees=2 interior_rejected=true double_free_rejected=true watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
