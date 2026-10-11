#pragma once
#include "ps5hip/launch.h"

namespace ps5hip {

// One CPU/GPU shared direct-memory allocation. Native submission owns a GPU
// reference until completion; timeout must retain that reference and poison the queue.
// Calls must be serialized by the owning device/queue; this is not a shared allocator.
class NativeMemory {
public:
    NativeMemory(uint64_t bytes, uint64_t alignment, bool code_window);
    ~NativeMemory();
    NativeMemory(const NativeMemory &) = delete;
    NativeMemory &operator=(const NativeMemory &) = delete;
    GpuRange range() const;
    void *cpu() const { return cpu_; }
    void acquire_gpu_use();
    void complete_gpu_use();
    bool close();
private:
    void *cpu_ = nullptr;
    bool mapped_ = false;
    int64_t physical_ = -1;
    uint64_t bytes_ = 0;
    uint32_t gpu_uses_ = 0;
    [[noreturn]] void fail(const char *message);
};

} // namespace ps5hip
