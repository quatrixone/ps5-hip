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
      observed.launches != 2 || observed.allocations != 1 || observed.queues != 1 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_LAUNCH FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_LAUNCH FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_LAUNCH QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

int main() {
  try {
    std::ifstream config("/app0/results-dir.txt", std::ios::binary);
    char config_bytes[242]{};
    config.read(config_bytes, sizeof(config_bytes));
    const auto length = config.gcount();
    if (!config.eof() || length < 2 || length > 240) return 2;
    const std::string directory(config_bytes, static_cast<size_t>(length));
    if (!ps5hip_canary::valid_results_directory(directory)) return 2;
    if (elevation::request(elevation::Capability::filesystem, "/app0/lapy.elf") != elevation::Status::ok) return 2;
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-launch.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_launch");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    const hipError_t loaded = hipModuleLoad(&module, "/app0/thread_words.hsaco");
    if (loaded != hipSuccess || !module) {
      std::fprintf(stderr, "HIP_LAUNCH MODULE_ERROR operation=load status=%d module_present=%s\n",
                   static_cast<int>(loaded), module ? "true" : "false");
      ps5hip_clr_gpu_fail("public-module-load");
    }
    const hipError_t resolved = hipModuleGetFunction(&function, module, "thread_words");
    if (resolved != hipSuccess || !function) {
      std::fprintf(stderr, "HIP_LAUNCH MODULE_ERROR operation=lookup status=%d function_present=%s\n",
                   static_cast<int>(resolved), function ? "true" : "false");
      ps5hip_clr_gpu_fail("public-module-lookup");
    }
    constexpr unsigned words = 81, active = 73;
    void* buffer = nullptr;
    if (hipMalloc(&buffer, words * sizeof(uint32_t)) != hipSuccess || !buffer ||
        observed.allocations != 1 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    // Static host storage remains live if completion becomes uncertain.
    static std::array<uint32_t, words> input{}, output{};
    unsigned compared = 0, rejected = 0;
    for (unsigned round = 0; round != 2; ++round) {
      for (unsigned i = 0; i != words; ++i) {
        input[i] = 0xa5000000u + i * 7u + round * 113u;
        output[i] = 0xdeadbeefu;
      }
      if (hipMemcpy(buffer, input.data(), sizeof(input), hipMemcpyHostToDevice) != hipSuccess)
        ps5hip_clr_gpu_fail("public-upload");
      unsigned count = active, salt = 19 + round * 24;
      void* args[] = {&buffer, &count, &salt};
      if (hipModuleLaunchKernel(function, 2, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) != hipErrorInvalidValue ||
          observed.launches != round || observed.terminal || observed.uncertain)
        ps5hip_clr_gpu_fail("launch-admission");
      ++rejected;
      if (hipModuleLaunchKernel(function, 2, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr) != hipSuccess ||
          hipDeviceSynchronize() != hipSuccess || observed.launches != round + 1 ||
          hipMemcpy(output.data(), buffer, sizeof(output), hipMemcpyDeviceToHost) != hipSuccess)
        ps5hip_clr_gpu_fail("public-dispatch");
      for (unsigned i = 0; i != words; ++i) {
        const uint32_t expected = i < active ? i * 17u + salt : input[i];
        if (output[i] != expected || input[i] != 0xa5000000u + i * 7u + round * 113u)
          ps5hip_clr_gpu_fail("output-or-guard-mismatch");
        ++compared;
      }
    }
    if (compared != 162 || rejected != 2 || hipFree(buffer) != hipSuccess ||
        hipModuleUnload(module) != hipSuccess || observed.allocations != 1 || observed.frees != 1 ||
        observed.launches != 2 || observed.terminal || observed.uncertain)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_LAUNCH PASS init=0 count=1 rounds=2 launches=2 values=146 guards=16 rejected=2 allocations=1 frees=1 watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
