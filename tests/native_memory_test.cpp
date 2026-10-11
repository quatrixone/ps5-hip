#include "ps5hip/native_memory.h"
#include <ps5platform/kernel.h>
#include <iostream>

namespace {
uint64_t mapped_address = 0x200040000, allocated_bytes = 0, allocated_alignment = 0;
unsigned releases = 0, unmaps = 0;
int fail_allocate = 0, fail_map = 0, fail_unmap = 0, fail_release = 0;
void check(bool ok) { if (!ok) throw std::runtime_error("native memory assertion failed"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid allocation accepted");
}
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int type, int64_t *physical) {
    check(type == 12); allocated_bytes = bytes; allocated_alignment = alignment;
    *physical = 0x10000; return fail_allocate;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t, int protection, int flags, int64_t, size_t) {
    check(protection == 0x33 && flags == 0);
    *address = reinterpret_cast<void *>(mapped_address); return fail_map;
}
extern "C" int32_t sceKernelMunmap(void *, size_t) { ++unmaps; return fail_unmap; }
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t, size_t) { ++releases; return fail_release; }

int main() {
    try {
        using ps5hip::NativeMemory;
        {
            NativeMemory memory(123, 256, true);
            check(allocated_bytes == 0x4000 && allocated_alignment == 0x10000);
            check(memory.range().address == mapped_address && memory.range().bytes == 0x4000);
            memory.acquire_gpu_use(); memory.acquire_gpu_use();
            check(!memory.close() && !unmaps && !releases);
            memory.complete_gpu_use(); check(!memory.close());
            memory.complete_gpu_use();
            fail_unmap = 1; check(!memory.close() && !releases);
            fail_unmap = 0; fail_release = 1; check(!memory.close());
            fail_release = 0; check(memory.close());
            reject([&] { memory.range(); }); reject([&] { memory.complete_gpu_use(); });
        }
        auto prior = releases;
        fail_map = 1; reject([] { NativeMemory m(1, 256, true); }); fail_map = 0;
        check(releases == prior + 1);
        prior = releases;
        fail_allocate = 1; reject([] { NativeMemory m(1, 256, true); }); fail_allocate = 0;
        check(releases == prior);
        mapped_address = 0x4000000000;
        reject([] { NativeMemory m(1, 256, true); });
        mapped_address = 0;
        prior = unmaps;
        reject([] { NativeMemory m(1, 256, true); });
        check(unmaps == prior + 1);
        mapped_address = 0x4000000000;
        { NativeMemory m(0x200001, 256, false); check(allocated_alignment == 0x200000 && allocated_bytes == 0x204000); }
        reject([] { NativeMemory m(0, 256, false); });
        reject([] { NativeMemory m(UINT64_MAX, 256, false); });
        reject([] { NativeMemory m(1, 3, false); });
        std::cout << "native allocation mock: mapping, cleanup failures and GPU ownership checks passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
