#pragma once
#include <cstddef>
#include <cstdint>

namespace ps5hip::llm {
// GGML aligned allocation pairs, including contexts, CPU buffers and threadpools.
// Ordinary libc allocation and the HIP allocation cap are unchanged.
struct NativeHostAllocatorStats {
  uint64_t allocations, releases, live_allocations;
  uint64_t live_requested_bytes, peak_requested_bytes, failures;
  bool poisoned;
};
NativeHostAllocatorStats native_host_allocator_stats() noexcept;
}
extern "C" void* __wrap_ggml_aligned_malloc(size_t size);
extern "C" void __wrap_ggml_aligned_free(void* pointer, size_t size);
