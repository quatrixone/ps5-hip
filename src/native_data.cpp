#include "ps5hip/native_data.h"
#include <cstring>
#include <limits>

namespace ps5hip {
namespace {
[[noreturn]] void fail(NativeDataStatus status, const char *detail) { throw NativeDataError(status, detail); }
bool overlap(GpuRange a, GpuRange b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
void NativeDataRegistry::require_live_registry() const {
    if (poisoned_) fail(NativeDataStatus::Poisoned, "native data registry is poisoned");
}
void *NativeDataRegistry::allocate(uint64_t bytes) {
    require_live_registry();
    if (!bytes || bytes > (uint64_t(1) << 47) - 0x4000)
        fail(NativeDataStatus::InvalidValue, "invalid logical allocation size");
    // Bookkeeping allocation must precede native mapping, including its owner.
    entries_.reserve(entries_.size() + 1);
    std::shared_ptr<NativeMemory> memory;
    try { memory = std::make_shared<NativeMemory>(bytes, 0x10000, false); }
    catch (const NativeMemorySetupError &error) {
        poisoned_ = error.retained_backing;
        throw NativeDataError(NativeDataStatus::NativeFailure, error.what());
    }
    catch (const ObjectError &error) {
        throw NativeDataError(NativeDataStatus::NativeFailure, error.what());
    }
    const auto backing = memory->range();
    bool conflict = false;
    for (const auto &entry : entries_)
        if (entry.memory && overlap(backing, entry.backing)) conflict = true;
    entries_.push_back({memory, backing, bytes, !conflict});
    if (conflict) {
        poisoned_ = true;
        fail(NativeDataStatus::NativeFailure, "native provider returned overlapping data mappings");
    }
    return memory->cpu();
}
NativeDataRegistry::Hold NativeDataRegistry::resolve(const void *pointer, uint64_t bytes) const {
    require_live_registry();
    const uint64_t address = reinterpret_cast<uintptr_t>(pointer);
    if (!address || !bytes) fail(NativeDataStatus::InvalidValue, "null pointer or zero data extent");
    for (const auto &entry : entries_) {
        if (!entry.live || address < entry.backing.address) continue;
        const auto offset = address - entry.backing.address;
        if (offset >= entry.logical_bytes) continue;
        if (bytes > entry.logical_bytes - offset)
            fail(NativeDataStatus::InvalidValue, "data extent exceeds logical allocation");
        if (entry.memory->gpu_in_use()) fail(NativeDataStatus::InUse, "data allocation is GPU-owned");
        // A shared hold may outlive a successful explicit free; it stays closed.
        try { entry.memory->range(); }
        catch (const ObjectError &) { fail(NativeDataStatus::InvalidPointer, "data backing was closed outside its owner"); }
        return {entry.memory, offset, entry.logical_bytes - offset};
    }
    fail(NativeDataStatus::InvalidPointer, "pointer is not in a live logical data allocation");
}
std::vector<GpuRange> NativeDataRegistry::ranges() const {
    require_live_registry();
    std::vector<GpuRange> result;
    for (const auto &entry : entries_) {
        if (!entry.live) continue;
        // Resolve validates ownership as well as logical extent.
        resolve(reinterpret_cast<void *>(entry.backing.address), entry.logical_bytes);
        result.push_back({entry.backing.address, entry.logical_bytes});
    }
    return result;
}
void NativeDataRegistry::free(void *base) {
    require_live_registry();
    const auto address = reinterpret_cast<uintptr_t>(base);
    for (auto &entry : entries_) {
        if (!entry.live || address != entry.backing.address) continue;
        if (entry.memory->gpu_in_use()) fail(NativeDataStatus::InUse, "cannot free GPU-owned data");
        if (!entry.memory->close()) {
            entry.live = false;
            poisoned_ = true;
            fail(NativeDataStatus::CleanupFailed, "native data cleanup failed; backing retained");
        }
        entry.live = false;
        entry.memory.reset();
        return;
    }
    fail(NativeDataStatus::InvalidPointer, "free requires a current allocation base");
}
void NativeDataRegistry::validate_host(const void *host, uint64_t bytes) const {
    const auto address = reinterpret_cast<uintptr_t>(host);
    if (!address || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - address ||
        bytes > std::numeric_limits<size_t>::max())
        fail(NativeDataStatus::InvalidValue, "invalid host copy extent");
    for (const auto &entry : entries_)
        if (entry.memory && overlap({address, bytes}, entry.backing))
            fail(NativeDataStatus::InvalidValue, "host copy extent aliases a registered device backing");
}
void NativeDataRegistry::copy_to_device(void *device, const void *host, uint64_t bytes) {
    auto hold = resolve(device, bytes);
    validate_host(host, bytes);
    std::memcpy(static_cast<unsigned char *>(hold.memory->cpu()) + hold.offset, host, size_t(bytes));
}
void NativeDataRegistry::copy_from_device(void *host, const void *device, uint64_t bytes) {
    auto hold = resolve(device, bytes);
    validate_host(host, bytes);
    std::memcpy(host, static_cast<unsigned char *>(hold.memory->cpu()) + hold.offset, size_t(bytes));
}
void NativeDataRegistry::fill(void *device, int value, uint64_t bytes) {
    auto hold = resolve(device, bytes);
    std::memset(static_cast<unsigned char *>(hold.memory->cpu()) + hold.offset, value, size_t(bytes));
}
} // namespace ps5hip
