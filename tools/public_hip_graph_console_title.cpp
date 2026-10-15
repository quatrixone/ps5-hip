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
      observed.allocations != 0 || observed.frees != 0 || observed.queues != 5 ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("HIP_GRAPH FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("HIP_GRAPH FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "HIP_GRAPH QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

static void checked(bool value,int line) {
  if(!value){std::fprintf(stderr,"HIP_GRAPH CHECK line=%d\n",line);ps5hip_clr_gpu_fail("oracle");}
}
static void ok(hipError_t status,int line) {
  if(status!=hipSuccess){std::fprintf(stderr,"HIP_GRAPH ERROR line=%d status=%d\n",line,int(status));ps5hip_clr_gpu_fail("public-api");}
}
#define HIP_OK(x) ok(x,__LINE__)
#include "public_hip_graph_fixture.h"
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
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-public-graph.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=public_hip_graph");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    run_graph_fixture("explicit",3,false);
    run_graph_fixture("recover",5,false);
    if(observed.allocations || observed.frees || observed.queues!=5 ||
       observed.retired_queues!=4 || observed.terminal || observed.uncertain ||
       observed.launches || backend_requests) ps5hip_clr_gpu_fail("public-retire");
    ps5hip_clr_gpu::watchdog_finish();
    completed=true;
    std::puts("HIP_GRAPH PASS init=0 count=1 graphs=2 replays=32 callbacks=64 invalidations=6 external_graphs=3 updates=true source_destroyed=true gpu_execution=false watchdog_joined=true finalization=pending");
    if(!ps5hip_canary::sync_log())ps5hip_clr_gpu_fail("result-log");
    return 0;
  }catch(...){ps5hip_clr_gpu_fail("exception");}
}
