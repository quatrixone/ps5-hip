#include "ps5hip/native_data.h"
#include <ps5platform/kernel.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sys/wait.h>
#include <unistd.h>

namespace {
struct Allocation { void *cpu = nullptr; size_t bytes, alignment; };
std::map<int64_t, Allocation> allocations;
int64_t next_physical = 1;
unsigned unmaps = 0, releases = 0;
bool fail_allocate = false, fail_map = false, fail_unmap = false, fail_release = false, duplicate_map = false;
void check(bool yes) { if (!yes) throw std::runtime_error("native data assertion failed"); }
template<class F> void reject(ps5hip::NativeDataStatus status, F action) {
    try { action(); } catch (const ps5hip::NativeDataError &error) { check(error.status() == status); return; }
    throw std::runtime_error("invalid native data operation accepted");
}
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int type, int64_t *physical) {
    check(type == 12);
    if (fail_allocate) return -1;
    *physical = next_physical++;
    allocations.emplace(*physical, Allocation{nullptr, bytes, alignment});
    return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t bytes, int protection, int flags, int64_t physical, size_t alignment) {
    check(protection == 0x33 && flags == 0);
    if (fail_map) return -1;
    auto &allocation = allocations.at(physical);
    check(allocation.bytes == bytes && allocation.alignment == alignment);
    if (duplicate_map) { *address = allocations.begin()->second.cpu; return 0; }
    allocation.cpu = std::aligned_alloc(alignment, (bytes + alignment - 1) & ~(alignment - 1));
    check(allocation.cpu != nullptr);
    std::memset(allocation.cpu, 0xa5, bytes);
    *address = allocation.cpu;
    return 0;
}
extern "C" int32_t sceKernelMunmap(void *address, size_t bytes) {
    ++unmaps;
    if (fail_unmap) return -1;
    for (auto &item : allocations) {
        if (item.second.cpu != address) continue;
        check(item.second.bytes == bytes);
        std::free(address); item.second.cpu = nullptr; break;
    }
    return 0;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t bytes) {
    ++releases;
    if (fail_release) return -1;
    check(allocations.at(physical).bytes == bytes);
    allocations.erase(physical);
    return 0;
}
int main() {
    try {
        using ps5hip::NativeDataRegistry;
        using S = ps5hip::NativeDataStatus;
        {
            NativeDataRegistry data;
            reject(S::InvalidValue, [&] { data.allocate(0); });
            reject(S::InvalidValue, [&] { data.allocate(UINT64_MAX); });
            auto *p = static_cast<unsigned char *>(data.allocate(17));
            auto *other = data.allocate(5);
            auto hold = data.resolve(p + 1, 16);
            check(hold.offset == 1 && hold.remaining == 16 && hold.memory->range().bytes == 0x4000);
            auto ranges = data.ranges();
            check(ranges.size() == 2 && ranges[0].bytes == 17 && ranges[1].bytes == 5);
            unsigned char source[16], destination[16]{};
            for (unsigned i = 0; i < 16; ++i) source[i] = static_cast<unsigned char>(i + 3);
            data.copy_to_device(p + 1, source, 16);
            std::memset(source, 0, sizeof(source));
            data.copy_from_device(destination, p + 1, 16);
            for (unsigned i = 0; i < 16; ++i) check(destination[i] == i + 3);
            check(p[0] == 0xa5 && p[17] == 0xa5); // logical boundary, not mapped padding
            reject(S::InvalidValue, [&] { data.resolve(p + 16, 2); });
            reject(S::InvalidValue, [&] { data.resolve(p, UINT64_MAX); });
            reject(S::InvalidPointer, [&] { data.resolve(p + 17, 1); });
            reject(S::InvalidPointer, [&] { data.resolve(p + 100, 1); });
            reject(S::InvalidValue, [&] { data.resolve(nullptr, 1); });
            reject(S::InvalidValue, [&] { data.resolve(p, 0); });
            reject(S::InvalidPointer, [&] { data.resolve(reinterpret_cast<void *>(UINTPTR_MAX), 1); });
            reject(S::InvalidPointer, [&] { data.free(p + 1); });
            reject(S::InvalidPointer, [&] { data.free(nullptr); });
            reject(S::InvalidValue, [&] { data.copy_to_device(p, nullptr, 1); });
            reject(S::InvalidValue, [&] { data.copy_to_device(p, reinterpret_cast<void *>(UINTPTR_MAX), 2); });
            reject(S::InvalidValue, [&] { data.copy_to_device(p, p + 1, 1); });
            reject(S::InvalidValue, [&] { data.copy_from_device(other, p, 1); });
            reject(S::InvalidValue, [&] { data.copy_to_device(p, p + 100, 1); });
            auto old_unmaps = unmaps;
            hold.memory->acquire_gpu_use();
            reject(S::InUse, [&] { data.free(p); });
            reject(S::InUse, [&] { data.copy_from_device(destination, p, 1); });
            reject(S::InUse, [&] { data.ranges(); });
            check(unmaps == old_unmaps);
            hold.memory->complete_gpu_use();
            data.free(p); // an ordinary shared hold does not extend mapping lifetime
            check(hold.memory->cpu() == nullptr);
            try { hold.memory->acquire_gpu_use(); throw std::runtime_error("closed hold submitted"); }
            catch (const ps5hip::ObjectError &) {}
            reject(S::InvalidPointer, [&] { data.free(p); });
            reject(S::InvalidPointer, [&] { data.resolve(p, 1); });
            data.free(other);
            check(data.ranges().empty());
        }
        check(allocations.empty());
        for (bool release_failure : {false, true}) {
            NativeDataRegistry data;
            auto *p = data.allocate(16);
            auto hold = data.resolve(p, 16);
            fail_unmap = !release_failure; fail_release = release_failure;
            reject(S::CleanupFailed, [&] { data.free(p); });
            check(data.poisoned() && !allocations.empty());
            auto old_unmaps = unmaps, old_releases = releases;
            reject(S::Poisoned, [&] { data.free(p); });
            reject(S::Poisoned, [&] { data.resolve(p, 1); });
            reject(S::Poisoned, [&] { data.ranges(); });
            reject(S::Poisoned, [&] { data.allocate(1); });
            check(unmaps == old_unmaps && releases == old_releases);
            // Shutdown may retry non-GPU cleanup; no registry operation retries it.
            fail_unmap = fail_release = false;
        }
        check(allocations.empty());
        for (bool map_failure : {false, true}) {
            NativeDataRegistry data;
            auto *p = data.allocate(16);
            fail_allocate = !map_failure; fail_map = map_failure;
            reject(S::NativeFailure, [&] { data.allocate(1); });
            check(!data.poisoned());
            fail_allocate = fail_map = false;
            check(data.resolve(p, 16).remaining == 16);
            data.free(p);
        }
        check(allocations.empty());
        {
            NativeDataRegistry data;
            data.allocate(16); duplicate_map = true;
            reject(S::NativeFailure, [&] { data.allocate(32); });
            duplicate_map = false; check(data.poisoned());
        }
        check(allocations.empty());
        // Unknown GPU ownership deliberately ends in a child process, never
        // recovered/reused in the parent and never disguised as leak-free cleanup.
        const auto child = fork(); check(child >= 0);
        if (child == 0) {
            const auto old_unmaps = unmaps, old_releases = releases;
            {
                NativeDataRegistry data;
                auto *p = data.allocate(16);
                auto hold = data.resolve(p, 16);
                hold.memory->acquire_gpu_use(); data.poison();
                reject(S::Poisoned, [&] { data.copy_to_device(p, "x", 1); });
            }
            _exit(unmaps == old_unmaps && releases == old_releases && allocations.size() == 1 ? 0 : 1);
        }
        int status = 0; check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        const auto setup_child = fork(); check(setup_child >= 0);
        if (setup_child == 0) {
            NativeDataRegistry data;
            fail_map = fail_release = true;
            reject(S::NativeFailure, [&] { data.allocate(1); });
            _exit(data.poisoned() && allocations.size() == 1 ? 0 : 1);
        }
        check(waitpid(setup_child, &status, 0) == setup_child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        std::cout << "native data model: logical bounds, copy ownership, free and quarantine passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
