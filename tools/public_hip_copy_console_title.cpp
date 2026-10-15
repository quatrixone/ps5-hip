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
  if (!completed || observed.terminal ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_COPY FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_COPY FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_COPY QUARANTINED reason=%s no_shell_request=true\n", reason);
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-copy.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_copy");
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
    constexpr size_t bytes = 289, guard = 16, payload = 257;
    if (hipMalloc(&first, bytes) != hipSuccess || !first ||
        hipMalloc(&second, bytes) != hipSuccess || !second || first == second ||
        observed.allocations != 2 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    auto* a = static_cast<unsigned char*>(first);
    auto* b = static_cast<unsigned char*>(second);
    unsigned compared = 0, rejected = 0;
    // Retain host storage even if a command throws before its completion is known.
    static std::array<unsigned char, bytes> input{}, seed{}, expected{}, output{}, source_after{};
    static std::array<unsigned char, 2> sentinel{};
    for (unsigned round = 0; round != 2; ++round) {
      for (size_t i = 0; i != bytes; ++i) {
        input[i] = static_cast<unsigned char>((i * 17 + round * 43) ^ 0x59);
        seed[i] = static_cast<unsigned char>(i * 7 + round * 29 + 0xa5);
      }
      expected = seed;
      for (size_t i = 0; i != payload; ++i) expected[guard + i] = input[guard + i];
      if (hipMemcpy(a, input.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy(b, seed.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy(b + guard, a + guard, payload, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemcpy(output.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
          hipMemcpy(source_after.data(), a, bytes, hipMemcpyDeviceToHost) != hipSuccess)
        ps5hip_clr_gpu_fail("public-copy");
      if (output != expected || source_after != input) ps5hip_clr_gpu_fail("byte-or-guard-mismatch");
      compared += 2 * bytes;
      // In-range starts with out-of-range lengths must fail before submission.
      sentinel = {0x39, 0xa7};
      if (hipMemcpy(b + guard, a + guard, bytes, hipMemcpyDeviceToDevice) != hipErrorInvalidValue ||
          hipMemcpy(b + bytes - 1, input.data(), 2, hipMemcpyHostToDevice) != hipErrorInvalidValue ||
          hipMemcpy(sentinel.data(), a + bytes - 1, 2, hipMemcpyDeviceToHost) != hipErrorInvalidValue ||
          sentinel != std::array<unsigned char, 2>{0x39, 0xa7})
        ps5hip_clr_gpu_fail("copy-bounds");
      rejected += 3;
      if (hipMemcpy(output.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected)
        ps5hip_clr_gpu_fail("rejected-copy-mutation");
      compared += bytes;
      for (size_t i = 0; i != bytes; ++i)
        if (input[i] != static_cast<unsigned char>((i * 17 + round * 43) ^ 0x59) ||
            seed[i] != static_cast<unsigned char>(i * 7 + round * 29 + 0xa5))
          ps5hip_clr_gpu_fail("host-source-mutation");
    }
    if (compared != 1734 || rejected != 6 || hipFree(second) != hipSuccess ||
        hipFree(first) != hipSuccess || observed.allocations != 2 || observed.frees != 2)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_COPY PASS init=0 count=1 rounds=2 compared_bytes=1734 bounds_rejected=6 allocations=2 frees=2 watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
