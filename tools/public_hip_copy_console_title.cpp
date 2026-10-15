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
  return {};
}
void finalized() {
  if (!completed || observed.terminal || observed.uncertain || observed.launches || backend_requests ||
      observed.allocations != 8 || observed.frees != 8 || observed.queues != 4 ||
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

static void checked(bool value,int line){
  if(!value){std::fprintf(stderr,"HIP_COPY LIFETIME_ERROR line=%d\n",line);ps5hip_clr_gpu_fail("allocation-lifetime");}
}
static void lifetime_ok(hipError_t status,int line){
  if(status!=hipSuccess){std::fprintf(stderr,"HIP_COPY LIFETIME_ERROR line=%d status=%d\n",line,int(status));ps5hip_clr_gpu_fail("allocation-lifetime-api");}
}
#define HIP_OK(x) lifetime_ok(x,__LINE__)
#include "public_hip_allocation_lifetime_fixture.h"
#undef HIP_OK

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
    constexpr size_t bytes = 512, guard = 16, payload = 257;
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
      expected = seed;
      for (size_t y = 0; y != 4; ++y)
        for (size_t x = 0; x != 8; ++x) expected[5 + y * 24 + x] = input[3 + y * 16 + x];
      if (hipMemcpy(b, seed.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy2D(b + 5, 24, a + 3, 16, 8, 4, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemcpy(output.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected)
        ps5hip_clr_gpu_fail("rectangular-2d-padding");
      compared += bytes;
      expected = seed;
      for (size_t z = 0; z != 3; ++z)
        for (size_t y = 0; y != 3; ++y)
          for (size_t x = 0; x != 7; ++x)
            expected[4 + 24 + z * 120 + y * 24 + x] = input[2 + 16 + z * 80 + y * 16 + x];
      hipMemcpy3DParms region{};
      region.srcPtr = make_hipPitchedPtr(a, 16, 12, 5);
      region.dstPtr = make_hipPitchedPtr(b, 24, 20, 5);
      region.srcPos = make_hipPos(2, 1, 0); region.dstPos = make_hipPos(4, 1, 0);
      region.extent = make_hipExtent(7, 3, 3); region.kind = hipMemcpyDeviceToDevice;
      if (hipMemcpy(b, seed.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy3DAsync(&region, nullptr) != hipSuccess || hipDeviceSynchronize() != hipSuccess ||
          hipMemcpy(output.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected)
        ps5hip_clr_gpu_fail("rectangular-3d-padding");
      compared += bytes;
      region.srcPos.x = 10;
      if (hipMemcpy3D(&region) != hipErrorInvalidValue ||
          hipMemcpy2D(b + 500, 24, a, 16, 8, 4, hipMemcpyDeviceToDevice) != hipErrorInvalidValue ||
          hipMemcpy2D(b, 24, a + 500, 16, 8, 4, hipMemcpyDeviceToDevice) != hipErrorInvalidValue ||
          hipMemcpy2D(b + 1, 16, b, 16, 8, 2, hipMemcpyDeviceToDevice) != hipErrorUnknown)
        ps5hip_clr_gpu_fail("rectangular-admission");
      region.srcPos.x = 2; region.srcPtr.ysize = SIZE_MAX;
      if (hipMemcpy3D(&region) != hipErrorInvalidValue) ps5hip_clr_gpu_fail("rectangular-overflow");
      rejected += 5;
      if (hipMemcpy(output.data(), b, bytes, hipMemcpyDeviceToHost) != hipSuccess || output != expected ||
          hipMemcpy(source_after.data(), a, bytes, hipMemcpyDeviceToHost) != hipSuccess || source_after != input)
        ps5hip_clr_gpu_fail("rejected-rectangular-mutation");
      compared += 2 * bytes;
      for (size_t i = 0; i != bytes; ++i)
        if (input[i] != static_cast<unsigned char>((i * 17 + round * 43) ^ 0x59) ||
            seed[i] != static_cast<unsigned char>(i * 7 + round * 29 + 0xa5))
          ps5hip_clr_gpu_fail("host-source-mutation");
    }
    if (compared != 7168 || rejected != 16 || hipFree(second) != hipSuccess ||
        hipFree(first) != hipSuccess || observed.allocations != 2 || observed.frees != 2 ||
        observed.terminal || observed.uncertain || observed.launches || backend_requests)
      ps5hip_clr_gpu_fail("public-retire");
    allocation_lifetime_fixture::run();
    if(observed.allocations!=8 || observed.frees!=8 || observed.queues!=4 ||
       observed.retired_queues!=3 || observed.terminal || observed.uncertain ||
       observed.launches || backend_requests)ps5hip_clr_gpu_fail("lifetime-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("HIP_COPY PASS init=0 count=1 rounds=2 compared_bytes=7168 bounds_rejected=16 rectangular=true padding=true gpu_execution=false lifetime_cases=2 lifetime_bytes=384 gated_free_observed=true migration=true allocations=8 frees=8 watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
