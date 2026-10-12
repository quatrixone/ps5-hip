#include <ps5platform/kernel.h>
#include <cstdlib>
#include <map>
#include <stdexcept>
namespace {
struct Allocation { void *cpu = nullptr; size_t bytes, alignment; };
// Created before main, destroyed after the function-local control registry.
struct MockState {
    std::map<int64_t, Allocation> values;
    ~MockState() { if (!values.empty()) std::abort(); }
} mock_state;
std::map<int64_t, Allocation> &allocations() { return mock_state.values; }
int failure = 0;
int64_t next_physical = 1;
}
extern "C" void ps5hip_memory_mock_failure(int value) { failure = value; }
extern "C" unsigned ps5hip_memory_mock_live() { return unsigned(allocations().size()); }
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int, int64_t *physical) {
    if (failure == 1) return -1;
    *physical = next_physical++;
    allocations().emplace(*physical, Allocation{nullptr, bytes, alignment}); return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t bytes, int protection, int, int64_t physical, size_t alignment) {
    if (failure == 2) return -1;
    if (protection != 0x33) throw std::runtime_error("unexpected mock map protection");
    auto &allocation = allocations().at(physical);
    allocation.cpu = std::aligned_alloc(alignment, (bytes + alignment - 1) & ~(alignment - 1));
    if (!allocation.cpu) return -1;
    *address = allocation.cpu; return 0;
}
extern "C" int32_t sceKernelMunmap(void *address, size_t) {
    if (failure == 3) return -1;
    for (auto &allocation : allocations()) if (allocation.second.cpu == address) {
        std::free(address); allocation.second.cpu = nullptr; return 0;
    }
    return -1;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t) {
    allocations().erase(physical); return 0;
}
