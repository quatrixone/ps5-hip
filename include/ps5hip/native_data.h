#pragma once
#include "ps5hip/native_memory.h"
#include <memory>
#include <vector>

namespace ps5hip {
enum class NativeDataStatus { InvalidValue, InvalidPointer, InUse, Poisoned, CleanupFailed, NativeFailure };
class NativeDataError : public ObjectError {
public:
    NativeDataError(NativeDataStatus status, const char *detail) : ObjectError(detail), status_(status) {}
    NativeDataStatus status() const { return status_; }
private:
    NativeDataStatus status_;
};

// Serialized owner of data buffers only; never insert code or launch storage.
// A hold is not a GPU lease. The caller must serialize resolve/acquire/submit
// with free; NativeMemory::acquire_gpu_use rejects a previously closed owner.
class NativeDataRegistry {
public:
    struct Hold {
        std::shared_ptr<NativeMemory> memory;
        uint64_t offset, remaining;
    };
    NativeDataRegistry() = default;
    NativeDataRegistry(const NativeDataRegistry &) = delete;
    NativeDataRegistry &operator=(const NativeDataRegistry &) = delete;
    void *allocate(uint64_t bytes);
    Hold resolve(const void *pointer, uint64_t bytes) const;
    std::vector<GpuRange> ranges() const;
    void free(void *base);
    // Host storage must be readable/writable for bytes and remain valid for the
    // whole call. Arithmetic checks do not prove arbitrary CPU memory is mapped.
    void copy_to_device(void *device, const void *host, uint64_t bytes);
    void copy_from_device(void *host, const void *device, uint64_t bytes);
    // Both nonempty ranges must be live, CPU-owned and nonoverlapping.
    void copy_device(void *destination, const void *source, uint64_t bytes);
    void fill(void *device, int value, uint64_t bytes);
    void poison() noexcept { poisoned_ = true; }
    bool poisoned() const { return poisoned_; }
private:
    struct Entry {
        std::shared_ptr<NativeMemory> memory;
        GpuRange backing{};
        uint64_t logical_bytes = 0;
        bool live = false;
    };
    std::vector<Entry> entries_;
    bool poisoned_ = false;
    void require_live_registry() const;
    void validate_host(const void *, uint64_t) const;
};
} // namespace ps5hip
