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
#include <cmath>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
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
      observed.launches != 6 || observed.allocations != 2 || observed.queues != 1 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_RMSNORM FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_RMSNORM FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_RMSNORM QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

namespace {
constexpr unsigned width = 2048, max_rows = 17, prefix = 16, suffix = 16;
constexpr unsigned payload_capacity = width * max_rows * sizeof(float);
constexpr unsigned total = prefix + payload_capacity + suffix;
constexpr const char* title_root = "/system_ex/app/PPSA99014/";
constexpr const char* symbol = "_ZL12rms_norm_f32ILi1024ELb0ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f";
struct Uint3 { uint32_t x, y, z; };
static_assert(sizeof(Uint3) == 12 && sizeof(float) == 4 && sizeof(int64_t) == 8, "kernel ABI");
// All host transfer storage remains alive on the quarantine path.
alignas(16) std::array<uint8_t,total> input{}, seed{}, readback{}, input_readback{};
std::array<float,width*max_rows> reference{};
void read_exact(const std::string& path, void* destination, size_t bytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file || !file.read(static_cast<char*>(destination), bytes) ||
      file.peek() != std::char_traits<char>::eof() || file.bad())
    ps5hip_clr_gpu_fail("asset-size-or-read");
}
void dump_exclusive(const std::string& path, const uint8_t* data, size_t bytes) {
  const int fd = ::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if (fd < 0) ps5hip_clr_gpu_fail("raw-dump-open");
  size_t written = 0;
  unsigned interruptions = 0;
  bool ok = true;
  while (written < bytes) {
    const auto count = ::write(fd,data+written,bytes-written);
    if (count < 0 && errno == EINTR && ++interruptions <= 8) continue;
    if (count <= 0 || size_t(count) > bytes-written) { ok=false; break; }
    written += size_t(count);
  }
  // Keep partial files as diagnostic evidence; never replace a prior run's dumps.
  const bool synced = ::fsync(fd) == 0;
  const bool closed = ::close(fd) == 0;
  if (!ok || !synced || !closed) ps5hip_clr_gpu_fail("raw-dump-write-sync-close");
}
double absolute(double value) { return value < 0 ? -value : value; }
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-rmsnorm.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_rmsnorm");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    if (initialized != hipSuccess || hipGetDeviceCount(&count) != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    const std::string module_path = std::string(title_root) + "rmsnorm/norm.hsaco";
    const auto loaded = hipModuleLoad(&module, module_path.c_str());
    if (loaded != hipSuccess || !module) {
      std::fprintf(stderr,"HIP_RMSNORM MODULE_ERROR operation=load status=%d\n",int(loaded));
      ps5hip_clr_gpu_fail("public-module-load");
    }
    const auto resolved = hipModuleGetFunction(&function,module,symbol);
    if (resolved != hipSuccess || !function) {
      std::fprintf(stderr,"HIP_RMSNORM MODULE_ERROR operation=lookup status=%d\n",int(resolved));
      ps5hip_clr_gpu_fail("public-module-lookup");
    }
    void* source_buffer = nullptr;
    void* destination_buffer = nullptr;
    if (hipMalloc(&source_buffer,total) != hipSuccess || !source_buffer ||
        hipMalloc(&destination_buffer,total) != hipSuccess || !destination_buffer ||
        observed.allocations != 2 || observed.frees != 0)
      ps5hip_clr_gpu_fail("public-allocate");
    unsigned cases = 0, compared = 0, guards = 0, inactive = 0, unchanged = 0;
    double maximum_error = 0, maximum_fraction = 0;
    for (const char* family : {"stress","embedding"}) for (unsigned rows : {1u,2u,17u}) {
      const unsigned elements = rows * width, bytes = elements * sizeof(float);
      input.fill(0xa5); seed.fill(0x5a); readback.fill(0xee); input_readback.fill(0xee);
      const std::string asset = std::string(title_root) + "rmsnorm/" + family + "-" + std::to_string(rows);
      read_exact(asset + ".input.f32le",input.data()+prefix,bytes);
      read_exact(asset + ".cpu.f32le",reference.data(),bytes);
      const uint32_t unwritten = 0x7fc00000u;
      for (unsigned i = 0; i < elements; ++i) {
        float value;
        std::memcpy(&value,input.data()+prefix+i*sizeof(float),sizeof(value));
        if (!std::isfinite(value) || !std::isfinite(reference[i])) ps5hip_clr_gpu_fail("nonfinite-asset");
        std::memcpy(seed.data()+prefix+i*sizeof(float),&unwritten,sizeof(unwritten));
      }
      if (hipMemcpy(source_buffer,input.data(),total,hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy(destination_buffer,seed.data(),total,hipMemcpyHostToDevice) != hipSuccess)
        ps5hip_clr_gpu_fail("public-upload");
      void* x = static_cast<uint8_t*>(source_buffer)+prefix;
      void* dst = static_cast<uint8_t*>(destination_buffer)+prefix;
      int ncols=width, nchannels=1, nsamples=1;
      int64_t stride_row=width, stride_channel=int64_t(width)*rows, stride_sample=stride_channel;
      float epsilon=1e-5f, scale=1.0f;
      void* optional=nullptr;
      int64_t zero_stride=0;
      Uint3 zero_packed{};
      void* arguments[] = {&x,&dst,&ncols,&nchannels,&nsamples,&stride_row,&stride_channel,&stride_sample,&epsilon,
        &optional,&zero_stride,&zero_stride,&zero_stride,&zero_packed,&zero_packed,&zero_packed,&zero_packed,
        &optional,&zero_stride,&zero_stride,&zero_stride,&zero_packed,&zero_packed,&zero_packed,&zero_packed,&scale};
      static_assert(sizeof(arguments)/sizeof(arguments[0]) == 26,"kernel argument count");
      const auto launched = hipModuleLaunchKernel(function,rows,1,1,1024,1,1,128,nullptr,arguments,nullptr);
      if (launched != hipSuccess) {
        std::fprintf(stderr,"HIP_RMSNORM LAUNCH_ERROR case=%u status=%d\n",cases,int(launched));
        ps5hip_clr_gpu_fail("public-launch");
      }
      if (hipDeviceSynchronize() != hipSuccess || observed.launches != cases+1 ||
          hipMemcpy(readback.data(),destination_buffer,total,hipMemcpyDeviceToHost) != hipSuccess ||
          hipMemcpy(input_readback.data(),source_buffer,total,hipMemcpyDeviceToHost) != hipSuccess)
        ps5hip_clr_gpu_fail("public-readback");
      const std::string dump = directory + "/rmsnorm-" + family + "-" + std::to_string(rows);
      dump_exclusive(dump + ".output.raw",readback.data(),total);
      dump_exclusive(dump + ".input-readback.raw",input_readback.data(),total);
      std::printf("HIP_RMSNORM DUMP family=%s rows=%u output_bytes=%u input_readback_bytes=%u exclusive=true synced=true\n",
                  family,rows,total,total);
      if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("dump-log");
      if (std::memcmp(input.data(),input_readback.data(),total) != 0) ps5hip_clr_gpu_fail("input-mutated");
      unchanged += total;
      for (unsigned i=0;i<total;++i) if (i<prefix || i>=prefix+bytes) {
        if (readback[i]!=seed[i]) ps5hip_clr_gpu_fail("output-guard-or-inactive");
        if (i<prefix || i<prefix+bytes+suffix) ++guards; else ++inactive;
      }
      double case_error=0, case_fraction=0;
      for (unsigned i=0;i<elements;++i) {
        float actual;std::memcpy(&actual,readback.data()+prefix+i*sizeof(float),sizeof(actual));
        const double error=absolute(double(actual)-double(reference[i]));
        const double budget=2e-5+2e-4*absolute(double(reference[i]));
        if (!std::isfinite(actual) || error>budget) {
          std::fprintf(stderr,"HIP_RMSNORM MISMATCH family=%s rows=%u index=%u actual=%.9g reference=%.9g error=%.9g budget=%.9g\n",
                       family,rows,i,double(actual),double(reference[i]),error,budget);
          ps5hip_clr_gpu_fail("numerical-oracle");
        }
        if (error>case_error) case_error=error;
        if (error/budget>case_fraction) case_fraction=error/budget;
      }
      if (case_error>maximum_error) maximum_error=case_error;
      if (case_fraction>maximum_fraction) maximum_fraction=case_fraction;
      compared += elements; ++cases;
      std::printf("HIP_RMSNORM CASE family=%s rows=%u values=%u max_abs=%.9g max_budget_fraction=%.9g input_unchanged=true guards=true inactive=true\n",
                  family,rows,elements,case_error,case_fraction);
      if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("case-log");
    }
    if (cases!=6 || compared!=81920 || guards!=192 || inactive!=507904 || unchanged!=835776 ||
        observed.terminal || observed.uncertain || hipFree(source_buffer)!=hipSuccess ||
        hipFree(destination_buffer)!=hipSuccess || hipModuleUnload(module)!=hipSuccess ||
        observed.allocations!=2 || observed.frees!=2 || observed.launches!=6 || observed.queues!=1)
      ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed=true;
    std::printf("HIP_RMSNORM PASS cases=6 launches=6 values=81920 guard_bytes=192 inactive_bytes=507904 input_bytes=835776 max_abs=%.9g max_budget_fraction=%.9g allocations=2 frees=2 module_unloaded=true watchdog_joined=true finalization=pending\n",
                maximum_error,maximum_fraction);
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
