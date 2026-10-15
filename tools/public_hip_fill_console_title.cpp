#include "ps5-device.hpp"
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
std::atomic<unsigned> backend_requests{0};
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  ++backend_requests;
  return {}; // Buffer fill must not submit GPU work.
}
void finalized() {
  if (!completed || observed.terminal || observed.uncertain || observed.launches || backend_requests ||
      observed.allocations != 1 || observed.frees != 1 || observed.queues != 1 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_FILL FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_FILL FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_FILL QUARANTINED reason=%s no_shell_request=true\n", reason);
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-fill.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_fill");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    void* buffer = nullptr;
    constexpr size_t bytes = 128;
    if (hipMalloc(&buffer, bytes) != hipSuccess || !buffer ||
        observed.allocations != 1 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    auto* data = static_cast<unsigned char*>(buffer);
    // Static host storage remains valid if completion becomes uncertain.
    static std::array<unsigned char, bytes> seed{}, expected{}, output{};
    unsigned compared = 0, rejected = 0;
    for (unsigned round = 0; round != 2; ++round) {
      for (size_t i = 0; i != bytes; ++i)
        seed[i] = static_cast<unsigned char>(i * 7 + round * 29 + 0x39);
      expected = seed;
      const unsigned char byte = static_cast<unsigned char>(0xa5 + round * 17);
      const unsigned short word = static_cast<unsigned short>(0xb37d + round * 0x203);
      const unsigned dword = 0x87654321u + round * 0x01020304u;
      for (size_t i = 3; i != 12; ++i) expected[i] = byte;
      for (size_t i = 16; i != 30; ++i) expected[i] = static_cast<unsigned char>(word >> (8 * ((i - 16) % 2)));
      for (size_t i = 40; i != 84; ++i) expected[i] = static_cast<unsigned char>(dword >> (8 * ((i - 40) % 4)));
      for (size_t i = 121; i != 126; ++i) expected[i] = static_cast<unsigned char>(0xe4 + round);
      if (hipMemcpy(buffer, seed.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemset(data + 3, 0x100 | byte, 9) != hipSuccess ||
          hipMemsetD16(data + 16, word, 7) != hipSuccess ||
          hipMemsetD32Async(data + 40, static_cast<int>(dword), 11, nullptr) != hipSuccess ||
          hipMemsetAsync(data + 121, 0xe4 + round, 5, nullptr) != hipSuccess ||
          hipDeviceSynchronize() != hipSuccess ||
          hipMemcpy(output.data(), buffer, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected)
        ps5hip_clr_gpu_fail("fill-byte-or-guard-mismatch");
      compared += bytes;
      if (hipMemset(data + 127, 0, 2) != hipErrorInvalidValue ||
          hipMemsetD16(data + 1, 0, 1) != hipErrorInvalidValue ||
          hipMemsetD32(buffer, 0, SIZE_MAX / 4 + 1) != hipErrorInvalidValue ||
          hipMemset2D(buffer, 16, 0, 8, 2) != hipErrorNotSupported)
        ps5hip_clr_gpu_fail("fill-admission");
      rejected += 4;
      if (hipMemcpy(output.data(), buffer, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected)
        ps5hip_clr_gpu_fail("rejected-fill-mutation");
      compared += bytes;
      for (size_t i = 0; i != bytes; ++i)
        if (seed[i] != static_cast<unsigned char>(i * 7 + round * 29 + 0x39))
          ps5hip_clr_gpu_fail("host-source-mutation");
    }
    if (compared != 512 || rejected != 8 || hipFree(buffer) != hipSuccess ||
        observed.allocations != 1 || observed.frees != 1 || observed.launches ||
        backend_requests || observed.terminal || observed.uncertain)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_FILL PASS init=0 count=1 rounds=2 compared_bytes=512 rejected=8 allocations=1 frees=1 gpu_execution=false watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
