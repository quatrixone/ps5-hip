#include "ps5-device.hpp"
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
std::atomic<unsigned> backend_requests{0};
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  ++backend_requests;
  return {}; // This loader-only fixture must never construct a GPU backend.
}
void finalized() {
  if (!completed || observed.terminal || observed.uncertain || observed.launches || backend_requests ||
      observed.allocations || observed.frees || observed.queues || observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_MODULE FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_MODULE FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_MODULE QUARANTINED reason=%s no_shell_request=true\n", reason);
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-module.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_module");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    unsigned loads = 0, lookups = 0, unloads = 0;
    for (unsigned round = 0; round != 2; ++round) {
      hipModule_t first = nullptr, second = nullptr;
      if (hipModuleLoad(&first, "/app0/canary.hsaco") != hipSuccess || !first ||
          hipModuleLoad(&second, "/app0/canary.hsaco") != hipSuccess || !second || first == second)
        ps5hip_clr_gpu_fail("module-load");
      loads += 2;
      hipFunction_t first_kernel = nullptr, second_kernel = nullptr, vector_kernel = nullptr;
      if (hipModuleGetFunction(&first_kernel, first, "indexed_store") != hipSuccess || !first_kernel ||
          hipModuleGetFunction(&second_kernel, second, "indexed_store") != hipSuccess || !second_kernel ||
          hipModuleGetFunction(&vector_kernel, second, "vector_add") != hipSuccess || !vector_kernel ||
          first_kernel == second_kernel || second_kernel == vector_kernel || first_kernel == vector_kernel)
        ps5hip_clr_gpu_fail("module-functions");
      lookups += 3;
      hipFunction_t missing = nullptr;
      if (hipModuleGetFunction(&missing, second, "absent_kernel") != hipErrorNotFound || missing)
        ps5hip_clr_gpu_fail("missing-function");
      if (hipModuleUnload(first) != hipSuccess || hipModuleUnload(first) != hipErrorNotFound)
        ps5hip_clr_gpu_fail("first-unload");
      first = nullptr;
      ++unloads;
      hipFunction_t retained = nullptr;
      if (hipModuleGetFunction(&retained, second, "indexed_store") != hipSuccess || retained != second_kernel)
        ps5hip_clr_gpu_fail("module-isolation");
      ++lookups;
      if (hipModuleUnload(second) != hipSuccess)
        ps5hip_clr_gpu_fail("second-unload");
      second = nullptr;
      ++unloads;
    }
    if (loads != 4 || lookups != 8 || unloads != 4 || observed.allocations || observed.frees ||
        observed.queues || observed.retired_queues || observed.launches || observed.uncertain ||
        observed.terminal || backend_requests)
      ps5hip_clr_gpu_fail("module-counts");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_MODULE PASS init=0 count=1 loads=4 lookups=8 unloads=4 isolated=true gpu_execution=false watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
