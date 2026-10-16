#pragma once
#include <cstddef>
#include <cstdint>

namespace ps5hip::llm {
// External GGML aligned allocations only. Same-TU ggml.c allocations retain
// their original allocator/free pair. This does not change the HIP allocation cap.
struct NativeHostAllocatorStats {
  uint64_t allocations, releases, live_allocations;
  uint64_t live_requested_bytes, peak_requested_bytes, failures;
  bool poisoned;
};
NativeHostAllocatorStats native_host_allocator_stats() noexcept;
}
extern "C" void* __wrap_ggml_aligned_malloc(size_t size);
extern "C" void __wrap_ggml_aligned_free(void* pointer, size_t size);
