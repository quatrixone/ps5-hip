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
        using ps5hip::native_memory_stats;
        check(native_memory_stats().live_bytes==0);
        {
            NativeDataRegistry data;
            reject(S::InvalidValue, [&] { data.allocate(0); });
            reject(S::InvalidValue, [&] { data.allocate(UINT64_MAX); });
            auto *p = static_cast<unsigned char *>(data.allocate(17));
            auto *other = data.allocate(5);
            auto pair_stats=native_memory_stats();
            check(pair_stats.allocations==2&&pair_stats.releases==0&&pair_stats.live_bytes==0x8000&&pair_stats.peak_bytes==0x8000);
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
            data.fill(p + 1, 0x123, 16);
            for (unsigned i = 1; i < 17; ++i) check(p[i] == 0x23);
            data.fill(p + 2, -1, 3);
            for (unsigned i = 2; i < 5; ++i) check(p[i] == 0xff);
            check(p[0] == 0xa5 && p[1] == 0x23 && p[5] == 0x23 && p[17] == 0xa5);
            reject(S::InvalidValue, [&] { data.fill(p + 16, 0, 2); });
            reject(S::InvalidValue, [&] { data.fill(p, 0, UINT64_MAX); });
            reject(S::InvalidValue, [&] { data.fill(nullptr, 0, 1); });
            reject(S::InvalidValue, [&] { data.fill(p, 0, 0); });
            reject(S::InvalidPointer, [&] { data.fill(p + 17, 0, 1); });
            check(p[16] == 0x23 && p[17] == 0xa5);
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
            data.copy_device(other, p+1, 5);
            const unsigned char copied[]{0x23,0xff,0xff,0xff,0x23};
            check(std::memcmp(other,copied,5)==0 && p[0]==0xa5 && p[17]==0xa5);
            data.copy_device(p+10,p+1,5); // disjoint ranges of the same allocation
            check(std::memcmp(p+10,copied,5)==0 && p[9]==0x23 && p[15]==0x23);
            unsigned char snapshot[17];std::memcpy(snapshot,p,17);
            for(bool reverse:{false,true})reject(S::InvalidValue,[&]{data.copy_device(reverse?p:p+1,reverse?p+1:p,5);});
            reject(S::InvalidValue,[&]{data.copy_device(p,p,1);});
            reject(S::InvalidValue,[&]{data.copy_device(p,other,6);});
            reject(S::InvalidValue,[&]{data.copy_device(other,p,6);});
            reject(S::InvalidValue,[&]{data.copy_device(p,other,UINT64_MAX);});
            reject(S::InvalidValue,[&]{data.copy_device(p,nullptr,1);});
            reject(S::InvalidValue,[&]{data.copy_device(nullptr,p,1);});
            reject(S::InvalidValue,[&]{data.copy_device(p,other,0);});
            reject(S::InvalidPointer,[&]{data.copy_device(p,source,1);});
            reject(S::InvalidPointer,[&]{data.copy_device(source,p,1);});
            reject(S::InvalidPointer,[&]{data.copy_device(p+17,other,1);});
            check(std::memcmp(p,snapshot,17)==0 && std::memcmp(other,copied,5)==0);
            auto old_unmaps = unmaps;
            hold.memory->acquire_gpu_use();
            reject(S::InUse, [&] { data.free(p); });
            reject(S::InUse, [&] { data.copy_from_device(destination, p, 1); });
            reject(S::InUse, [&] { data.fill(p, 0, 1); });
            reject(S::InUse, [&] { data.copy_device(p,other,1); });
            reject(S::InUse, [&] { data.copy_device(other,p,1); });
            check(p[0] == 0xa5);
            reject(S::InUse, [&] { data.ranges(); });
            check(unmaps == old_unmaps);
            hold.memory->complete_gpu_use();
            data.free(p); // an ordinary shared hold does not extend mapping lifetime
            check(hold.memory->cpu() == nullptr);
            try { hold.memory->acquire_gpu_use(); throw std::runtime_error("closed hold submitted"); }
            catch (const ps5hip::ObjectError &) {}
            reject(S::InvalidPointer, [&] { data.free(p); });
            reject(S::InvalidPointer, [&] { data.resolve(p, 1); });
            reject(S::InvalidPointer, [&] { data.fill(p, 0, 1); });
            reject(S::InvalidPointer, [&] { data.copy_device(p,other,1); });
            reject(S::InvalidPointer, [&] { data.copy_device(other,p,1); });
            data.free(other);
            check(data.ranges().empty());
        }
        check(allocations.empty());
        {auto clean=native_memory_stats();check(clean.live_bytes==0&&clean.allocations==clean.releases);}
        {
            NativeDataRegistry data;
            auto* src = static_cast<unsigned char*>(data.allocate(512));
            auto* dst = static_cast<unsigned char*>(data.allocate(512));
            unsigned char expected[512], source[512], actual[512];
            for (unsigned round=0; round<2; ++round) {
                for(unsigned i=0;i<512;++i) { source[i]=static_cast<unsigned char>(i*13+round*37); expected[i]=0xa5; }
                data.copy_to_device(src,source,512);data.copy_to_device(dst,expected,512);
                for(unsigned z=0;z<3;++z)for(unsigned y=0;y<3;++y)for(unsigned x=0;x<7;++x)
                    expected[5+z*96+y*24+x]=source[3+z*80+y*16+x];
                data.copy_device_rect(dst+5,24,96,src+3,16,80,7,3,3);
                data.copy_from_device(actual,dst,512);check(std::memcmp(actual,expected,512)==0);
                data.copy_from_device(actual,src,512);check(std::memcmp(actual,source,512)==0);
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst+500,24,96,src,16,80,7,3,3);});
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst,24,96,src+500,16,80,7,3,3);});
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst,6,96,src,16,80,7,3,3);});
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst,24,48,src,16,80,7,3,3);});
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst,24,96,src,16,80,7,3,UINT64_MAX);});
                reject(S::InvalidValue,[&]{data.copy_device_rect(dst,24,96,src,16,80,0,3,3);});
                data.copy_from_device(actual,dst,512);check(std::memcmp(actual,expected,512)==0);
            }
            // Interleaved rows share a span but do not overlap; preserve exact source bytes.
            data.copy_to_device(src,source,512);
            std::memcpy(expected,source,512);
            for(unsigned y=0;y<4;++y)std::memcpy(expected+8+y*16,source+y*16,4);
            data.copy_device_rect(src+8,16,64,src,16,64,4,4,1);
            data.copy_from_device(actual,src,512);check(std::memcmp(actual,expected,512)==0);
            // Last destination row overlaps the first source row: reject before any write.
            reject(S::InvalidValue,[&]{data.copy_device_rect(src,16,64,src+48,16,64,4,4,1);});
            reject(S::InvalidValue,[&]{data.copy_device_rect(src+48,16,64,src,16,64,4,4,1);});
            data.copy_from_device(actual,src,512);check(std::memcmp(actual,expected,512)==0);
            {auto hold=data.resolve(src,1);hold.memory->acquire_gpu_use();
             reject(S::InUse,[&]{data.copy_device_rect(dst,16,64,src,16,64,4,4,1);});
             reject(S::InUse,[&]{data.copy_device_rect(src,16,64,dst,16,64,4,4,1);});
             hold.memory->complete_gpu_use();}
            data.free(src);data.free(dst);check(data.ranges().empty());
        }
        check(allocations.empty());
        {auto clean=native_memory_stats();check(clean.live_bytes==0&&clean.allocations==clean.releases);}
        for (bool release_failure : {false, true}) {
            NativeDataRegistry data;
            auto *p = data.allocate(16);
            auto hold = data.resolve(p, 16);
            fail_unmap = !release_failure; fail_release = release_failure;
            reject(S::CleanupFailed, [&] { data.free(p); });
            check(data.poisoned() && !allocations.empty());
            auto retained=native_memory_stats();check(retained.live_bytes==0x4000&&retained.allocations==retained.releases+1);
            auto old_unmaps = unmaps, old_releases = releases;
            reject(S::Poisoned, [&] { data.free(p); });
            reject(S::Poisoned, [&] { data.resolve(p, 1); });
            reject(S::Poisoned, [&] { data.ranges(); });
            reject(S::Poisoned, [&] { data.allocate(1); });
            reject(S::Poisoned, [&] { data.fill(p, 0, 1); });
            reject(S::Poisoned, [&] { data.copy_device(p,p,1); });
            check(unmaps == old_unmaps && releases == old_releases);
            // Shutdown may retry non-GPU cleanup; no registry operation retries it.
            fail_unmap = fail_release = false;
        }
        check(allocations.empty());
        {auto clean=native_memory_stats();check(clean.live_bytes==0&&clean.allocations==clean.releases);}
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
        {auto clean=native_memory_stats();check(clean.live_bytes==0&&clean.allocations==clean.releases);}
        {
            NativeDataRegistry data;
            const auto before=native_memory_stats();
            data.allocate(16); duplicate_map = true;
            reject(S::NativeFailure, [&] { data.allocate(32); });
            duplicate_map = false; check(data.poisoned());
            const auto overlap=native_memory_stats();
            check(overlap.allocations==before.allocations+2&&overlap.releases==before.releases&&overlap.live_bytes==0x8000);
        }
        check(allocations.empty());
        {auto clean=native_memory_stats();check(clean.live_bytes==0&&clean.allocations==clean.releases);}
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
            const auto retained=native_memory_stats();
            _exit(unmaps == old_unmaps && releases == old_releases && allocations.size() == 1 &&
                  retained.live_bytes==0x4000&&retained.allocations==retained.releases+1 ? 0 : 1);
        }
        int status = 0; check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        const auto setup_child = fork(); check(setup_child >= 0);
        if (setup_child == 0) {
            NativeDataRegistry data;
            fail_map = fail_release = true;
            reject(S::NativeFailure, [&] { data.allocate(1); });
            const auto retained=native_memory_stats();
            _exit(data.poisoned() && allocations.size() == 1 && retained.live_bytes==0x4000&&
                  retained.allocations==retained.releases+1 ? 0 : 1);
        }
        check(waitpid(setup_child, &status, 0) == setup_child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        std::cout << "native data model: logical bounds, copy ownership, free and quarantine passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
