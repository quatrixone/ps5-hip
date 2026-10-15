#include "ps5hip/native_memory.h"
#include <ps5platform/kernel.h>
#include <algorithm>
#include <cstdio>
#include <atomic>

namespace ps5hip {
namespace {
std::atomic_flag stats_lock = ATOMIC_FLAG_INIT;
NativeMemoryStats stats;
struct StatsGuard {
    StatsGuard() noexcept { while (stats_lock.test_and_set(std::memory_order_acquire)) {} }
    ~StatsGuard() { stats_lock.clear(std::memory_order_release); }
};
void acquired(uint64_t bytes) noexcept {
    StatsGuard guard;
    ++stats.allocations; stats.live_bytes += bytes;
    stats.peak_bytes = std::max(stats.peak_bytes, stats.live_bytes);
}
void released(uint64_t bytes) noexcept {
    StatsGuard guard;
    ++stats.releases; stats.live_bytes -= bytes;
}
}
NativeMemoryStats native_memory_stats() noexcept {
    StatsGuard guard;
    return stats;
}

NativeMemory::NativeMemory(uint64_t bytes, uint64_t alignment, bool code_window) {
    constexpr uint64_t page = 0x4000, large = 0x200000, limit = uint64_t(1) << 47;
    if (!bytes || bytes > limit - page || !alignment || (alignment & (alignment - 1)) || alignment > (uint64_t(1) << 32))
        throw ObjectError("invalid native allocation size/alignment");
    bytes_ = (bytes + page - 1) & ~(page - 1);
    if (code_window && bytes_ > (uint64_t(1) << 32)) throw ObjectError("code allocation exceeds address window");
    alignment = std::max<uint64_t>(alignment, 0x10000);
    if (bytes_ >= large) alignment = std::max(alignment, large);
    const int64_t pool = sceKernelGetDirectMemorySize();
    if (pool <= 0 || bytes_ > uint64_t(pool)) throw ObjectError("insufficient direct-memory pool");
    if (sceKernelAllocateDirectMemory(0, pool, bytes_, alignment, PS5_KERNEL_DIRECT_TYPE_CPU, &physical_) != 0) {
        physical_ = -1;
        throw ObjectError("direct-memory allocation failed");
    }
    acquired(bytes_);
    void *mapping = nullptr;
    constexpr int protection = PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE |
                               PS5_KERNEL_PROT_GPU_READ | PS5_KERNEL_PROT_GPU_WRITE;
    if (sceKernelMapDirectMemory(&mapping, bytes_, protection, 0, physical_, alignment) != 0)
        fail("direct-memory mapping failed");
    cpu_ = mapping;
    mapped_ = true;
    const uint64_t address = reinterpret_cast<uintptr_t>(cpu_);
    if (!address || address >= limit || bytes_ > limit - address || address % alignment)
        fail("direct-memory mapping outside address/alignment contract");
    if (code_window && (address < 0x200000000ull || address >= 0x300000000ull || bytes_ > 0x300000000ull - address))
        fail("direct-memory code mapping outside shader window");
}

[[noreturn]] void NativeMemory::fail(const char *message) {
    const bool retained = !close();
    if (retained) std::fprintf(stderr, "ps5hip: setup cleanup failed; native allocation retained\n");
    throw NativeMemorySetupError(message, retained);
}

NativeMemory::~NativeMemory() {
    if (!close()) std::fprintf(stderr, "ps5hip: native allocation retained (GPU use or kernel cleanup failure)\n");
}

GpuRange NativeMemory::range() const {
    if (!cpu_ || physical_ < 0) throw ObjectError("native allocation is not mapped");
    return {reinterpret_cast<uintptr_t>(cpu_), bytes_};
}

void NativeMemory::acquire_gpu_use() {
    range();
    if (gpu_uses_ == UINT32_MAX) throw ObjectError("GPU allocation reference overflow");
    ++gpu_uses_;
}

void NativeMemory::complete_gpu_use() {
    if (!gpu_uses_) throw ObjectError("GPU allocation reference underflow");
    --gpu_uses_;
}

bool NativeMemory::close() {
    if (gpu_uses_) return false;
    if (mapped_) {
        if (sceKernelMunmap(cpu_, bytes_) != 0) return false;
        cpu_ = nullptr;
        mapped_ = false;
    }
    if (physical_ >= 0) {
        if (sceKernelReleaseDirectMemory(physical_, bytes_) != 0) return false;
        released(bytes_);
        physical_ = -1;
    }
    bytes_ = 0;
    return true;
}

} // namespace ps5hip
