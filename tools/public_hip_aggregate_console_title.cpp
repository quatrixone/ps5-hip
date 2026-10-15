#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include "../tests/hip_aggregate_fixture.h"
#include <cstring>
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
    std::puts("HIP_AGGREGATE FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_AGGREGATE FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_AGGREGATE QUARANTINED reason=%s no_shell_request=true\n", reason);
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-aggregate.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_aggregate");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    constexpr unsigned prefix = 16, payload_bytes = 164, suffix = 16;
    constexpr unsigned total = prefix + payload_bytes + suffix;
    void* buffer = nullptr;
    if (hipMalloc(&buffer, total) != hipSuccess || !buffer ||
        observed.allocations != 1 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    // Static host storage remains live if completion becomes uncertain.
    static std::array<uint8_t, total> input{}, output{}, expected{};
    unsigned compared_members = 0, compared_guards = 0;
    for (unsigned round = 0; round != 2; ++round) {
      for (unsigned i = 0; i != total; ++i) {
        input[i] = uint8_t(0xa5u + i * 7u + round * 113u);
        output[i] = 0xee;
      }
      expected = input;
      const uint32_t before = 0x12345678u + round * 0x11111111u;
      const uint3 v{0x10203040u + round, 0x50607080u + round * 3u,
                    0x90a0b0c0u + round * 5u};
      const Aggregate16 a{0x1020304050607080ull + round * 17u,
                          0xa1b2c3d4u + round * 7u,
                          uint16_t(0x4567u + round * 11u),
                          uint8_t(0x89u + round), uint8_t(0xabu + round * 3u)};
      Softmax128 p{};
      p.nheads = 13 + round; p.n_head_log2 = 8 + round;
      p.ncols = 257 + round * 3; p.nrows_x = 73 + round * 5;
      p.nrows_y = 19 + round * 7; p.ne00 = 101 + round * 11;
      p.ne01 = 103 + round * 13; p.ne02 = 107 + round * 17;
      p.ne03 = 109 + round * 19; p.nb11 = 113 + round * 23;
      p.nb12 = 127 + round * 29; p.nb13 = 131 + round * 31;
      p.ne12 = 137 + round * 37; p.ne13 = 139 + round * 41;
      p.scale = 0.5f + round; p.max_bias = 2.0f + round;
      p.m0 = 4.0f + round; p.m1 = 8.0f + round;
      const uint32_t after = 0xfedcba98u - round * 0x1010101u;
      std::memcpy(expected.data() + prefix, &before, sizeof(before));
      std::memcpy(expected.data() + prefix + 4, &v, sizeof(v));
      std::memcpy(expected.data() + prefix + 16, &a, sizeof(a));
      std::memcpy(expected.data() + prefix + 32, &p, sizeof(p));
      std::memcpy(expected.data() + prefix + 160, &after, sizeof(after));
      if (hipMemcpy(buffer, input.data(), total, hipMemcpyHostToDevice) != hipSuccess)
        ps5hip_clr_gpu_fail("public-upload");
      launch_aggregate_fixture(static_cast<unsigned char*>(buffer) + prefix,
                               before, v, a, p, after);
      if (hipGetLastError() != hipSuccess ||
          hipDeviceSynchronize() != hipSuccess || observed.launches != round + 1 ||
          hipMemcpy(output.data(), buffer, total, hipMemcpyDeviceToHost) != hipSuccess)
        ps5hip_clr_gpu_fail("public-dispatch");
      for (unsigned i = 0; i != total; ++i) {
        if (input[i] != uint8_t(0xa5u + i * 7u + round * 113u))
          ps5hip_clr_gpu_fail("host-input-changed");
        // C++ by-value copies need not preserve Softmax128's four padding bytes.
        // Only those bytes are excluded; every member and both guards are checked.
        if (i >= prefix + 44 && i < prefix + 48) continue;
        if (output[i] != expected[i]) ps5hip_clr_gpu_fail("member-or-guard-mismatch");
        if (i < prefix || i >= prefix + payload_bytes) ++compared_guards;
        else ++compared_members;
      }
    }
    if (compared_members != 320 || compared_guards != 64 ||
        hipFree(buffer) != hipSuccess || observed.allocations != 1 || observed.frees != 1 ||
        observed.launches != 2 || observed.terminal || observed.uncertain)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_AGGREGATE PASS init=0 count=1 rounds=2 launches=2 member_bytes=320 guard_bytes=64 padding=unspecified compiler_stubs=true allocations=1 frees=1 watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
