#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include <hip/hip_runtime_api.h>
#include <ps5platform/libc.h>
#include <cstdio>

extern "C" int __cxa_thread_atexit_impl(void (*destructor)(void*), void* object,
                                          void* dso) {
  return ps5___cxa_thread_atexit_impl(destructor, object, dso);
}

namespace {
ps5hip::clr::Observation observed;
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  return std::make_unique<ps5hip::AgcBackend>();
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

int main() {
  const hipError_t initialized = hipInit(0);
  int count = -1;
  const hipError_t discovered = hipGetDeviceCount(&count);
  if (initialized != hipSuccess || discovered != hipSuccess || count != 1) return 1;
  void* first = nullptr;
  if (hipMalloc(&first, 257) != hipSuccess || !first || observed.allocations != 1)
    return 2;
  if (hipFree(static_cast<char*>(first) + 1) == hipSuccess || observed.frees != 0)
    return 3;
  if (hipFree(first) != hipSuccess || observed.frees != 1 ||
      hipFree(first) == hipSuccess) return 4;
  std::puts("PS5HIP_PUBLIC_MEMORY init=0 count=1 allocations=1 frees=1");
  return 0;
}
