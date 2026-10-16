#include "native_host_allocator.h"
#include "ps5hip/native_memory.h"
#include <atomic>
#include <new>

extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason);

namespace {
// No heap allocation is needed to remember backing ownership. Exhaustion is a
// recoverable allocation failure; cleanup failures permanently poison admission.
constexpr size_t capacity = 256;
struct Owner {
  alignas(ps5hip::NativeMemory) unsigned char storage[sizeof(ps5hip::NativeMemory)];
  ps5hip::NativeMemory* memory = nullptr;
  size_t requested = 0;
};
Owner owners[capacity];
std::atomic_flag gate = ATOMIC_FLAG_INIT;
ps5hip::llm::NativeHostAllocatorStats stats{};
struct Guard {
  Guard() noexcept { while (gate.test_and_set(std::memory_order_acquire)) {} }
  ~Guard() { gate.clear(std::memory_order_release); }
};
}

namespace ps5hip::llm {
NativeHostAllocatorStats native_host_allocator_stats() noexcept {
  Guard guard;
  return stats;
}
}

extern "C" void* __wrap_ggml_aligned_malloc(size_t size) {
  const char* fatal = nullptr;
  {
    Guard guard;
    if (stats.poisoned) {
      fatal = "ggml-host-allocator-poisoned";
    } else {
      // Match NativeMemory's checked page rounding before constructing an owner.
      if (!size || size > (uint64_t(1) << 47) - 0x4000 ||
          size > UINT64_MAX - stats.live_requested_bytes) {
        ++stats.failures;
        return nullptr;
      }
      Owner* owner = nullptr;
      for (auto& candidate : owners) if (!candidate.memory) { owner = &candidate; break; }
      if (!owner) { ++stats.failures; return nullptr; }
      try {
        // Reuse the established direct-memory rollback/retention implementation.
        // Its CPU/GPU-readable mapping grants no GPU computation or HIP ownership.
        owner->memory = new (owner->storage) ps5hip::NativeMemory(size, 64, false);
        owner->requested = size;
        ++stats.allocations;
        ++stats.live_allocations;
        stats.live_requested_bytes += size;
        if (stats.live_requested_bytes > stats.peak_requested_bytes)
          stats.peak_requested_bytes = stats.live_requested_bytes;
        return owner->memory->cpu();
      } catch (const ps5hip::NativeMemorySetupError& error) {
        ++stats.failures;
        if (!error.retained_backing) return nullptr;
        // A failed constructor has no destructible object. NativeMemory reports
        // retained physical backing in its global stats; never reuse/retry it.
        stats.poisoned = true;
        fatal = "ggml-host-setup-retained";
      } catch (...) {
        ++stats.failures;
        return nullptr;
      }
    }
  }
  ps5hip_llm_fail(fatal);
}

extern "C" void __wrap_ggml_aligned_free(void* pointer, size_t size) {
  if (!pointer) return;
  const char* fatal = nullptr;
  {
    Guard guard;
    if (stats.poisoned) {
      fatal = "ggml-host-allocator-poisoned";
    } else {
      Owner* owner = nullptr;
      for (auto& candidate : owners)
        if (candidate.memory && candidate.memory->cpu() == pointer) { owner = &candidate; break; }
      if (!owner || owner->requested != size) {
        ++stats.failures;
        stats.poisoned = true;
        fatal = "ggml-host-free-ownership";
      } else if (!owner->memory->close()) {
        // Do not run a destructor (which retries close), or discard the owner.
        ++stats.failures;
        stats.poisoned = true;
        fatal = "ggml-host-cleanup-retained";
      } else {
        owner->memory->~NativeMemory();
        owner->memory = nullptr;
        stats.live_requested_bytes -= owner->requested;
        owner->requested = 0;
        --stats.live_allocations;
        ++stats.releases;
        return;
      }
    }
  }
  ps5hip_llm_fail(fatal);
}
