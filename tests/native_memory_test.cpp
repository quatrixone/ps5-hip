#include "ps5hip/native_memory.h"
#include <ps5platform/kernel.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <chrono>

namespace {
uint64_t mapped_address = 0x200040000, allocated_bytes = 0, allocated_alignment = 0;
unsigned releases = 0, unmaps = 0;
std::mutex provider_lock;
int64_t next_physical = 0x10000;
bool parallel_mapping = false;
int fail_allocate = 0, fail_map = 0, fail_unmap = 0, fail_release = 0;
void check(bool ok) { if (!ok) throw std::runtime_error("native memory assertion failed"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid allocation accepted");
}
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int type, int64_t *physical) {
    std::lock_guard<std::mutex> lock(provider_lock);
    check(type == 12); allocated_bytes = bytes; allocated_alignment = alignment;
    *physical = next_physical++; return fail_allocate;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t, int protection, int flags, int64_t physical, size_t) {
    std::lock_guard<std::mutex> lock(provider_lock);
    check(protection == 0x33 && flags == 0);
    *address = reinterpret_cast<void *>(mapped_address + (parallel_mapping ? uint64_t(physical)*0x400000 : 0)); return fail_map;
}
extern "C" int32_t sceKernelMunmap(void *, size_t) { std::lock_guard<std::mutex> lock(provider_lock); ++unmaps; return fail_unmap; }
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t, size_t) { std::lock_guard<std::mutex> lock(provider_lock); ++releases; return fail_release; }

int main() {
    try {
        using ps5hip::NativeMemory;
        using ps5hip::native_memory_stats;
        auto initial=native_memory_stats();
        check(initial.allocations==0&&initial.releases==0&&initial.live_bytes==0&&initial.peak_bytes==0);
        {
            NativeMemory memory(123, 256, true);
            check(allocated_bytes == 0x4000 && allocated_alignment == 0x10000);
            check(memory.range().address == mapped_address && memory.range().bytes == 0x4000);
            auto allocated=native_memory_stats();
            check(allocated.allocations==1&&allocated.releases==0&&allocated.live_bytes==0x4000&&allocated.peak_bytes==0x4000);
            memory.acquire_gpu_use(); memory.acquire_gpu_use();
            check(!memory.close() && !unmaps && !releases);
            memory.complete_gpu_use(); check(!memory.close());
            memory.complete_gpu_use();
            fail_unmap = 1; check(!memory.close() && !releases);
            fail_unmap = 0; fail_release = 1; check(!memory.close());
            auto retained=native_memory_stats();check(retained.allocations==1&&retained.releases==0&&retained.live_bytes==0x4000);
            fail_release = 0; check(memory.close());
            auto closed=native_memory_stats();check(closed.allocations==1&&closed.releases==1&&closed.live_bytes==0&&closed.peak_bytes==0x4000);
            check(memory.close());check(native_memory_stats().releases==closed.releases);
            reject([&] { memory.range(); }); reject([&] { memory.complete_gpu_use(); });
        }
        auto prior = releases;
        auto before_map_failure=native_memory_stats();
        fail_map = 1; reject([] { NativeMemory m(1, 256, true); }); fail_map = 0;
        check(releases == prior + 1);
        auto after_map_failure=native_memory_stats();
        check(after_map_failure.allocations==before_map_failure.allocations+1&&
              after_map_failure.releases==before_map_failure.releases+1&&
              after_map_failure.live_bytes==before_map_failure.live_bytes);
        prior = releases;
        auto allocation_failure=native_memory_stats();
        fail_allocate = 1; reject([] { NativeMemory m(1, 256, true); }); fail_allocate = 0;
        check(releases == prior);check(native_memory_stats().allocations==allocation_failure.allocations);
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
        // Concurrent independent owners: every coherent snapshot obeys exact rounded-byte accounting.
        mapped_address=0x200200000;parallel_mapping=true;
        auto before=native_memory_stats();check(before.live_bytes==0);
        std::mutex gate_lock;std::condition_variable gate;unsigned ready=0,finished=0;bool proceed=false;
        std::vector<std::thread> workers;
        constexpr unsigned worker_count=4,rounds=32;constexpr uint64_t rounded=0x400000;
        for(unsigned i=0;i<worker_count;++i)workers.emplace_back([&]{
            for(unsigned round=0;round<rounds;++round){
                NativeMemory memory(rounded-1,256,false);
                if(round==0){std::unique_lock<std::mutex> lock(gate_lock);++ready;gate.notify_all();
                    check(gate.wait_for(lock,std::chrono::seconds(5),[&]{return proceed;}));}
                auto now=native_memory_stats();
                check(now.allocations>=before.allocations&&now.releases>=before.releases);
                check(now.live_bytes==((now.allocations-before.allocations)-(now.releases-before.releases))*rounded);
                check(now.peak_bytes>=now.live_bytes);
            }
            std::lock_guard<std::mutex> lock(gate_lock);++finished;gate.notify_all();
        });
        {std::unique_lock<std::mutex> lock(gate_lock);
         check(gate.wait_for(lock,std::chrono::seconds(5),[&]{return ready==worker_count;}));
         auto overlap=native_memory_stats();check(overlap.allocations==before.allocations+worker_count&&overlap.releases==before.releases);
         check(overlap.live_bytes==worker_count*rounded&&overlap.peak_bytes==worker_count*rounded);
         proceed=true;gate.notify_all();
         check(gate.wait_for(lock,std::chrono::seconds(5),[&]{return finished==worker_count;}));}
        for(auto& worker:workers)worker.join();
        parallel_mapping=false;
        auto after=native_memory_stats();
        check(after.allocations==before.allocations+worker_count*rounds&&after.releases==before.releases+worker_count*rounds);
        check(after.live_bytes==0&&after.peak_bytes==worker_count*rounded);
        // Failed constructors have no destructor: retained backing remains in the lifetime totals.
        fail_map=fail_release=1;
        try {NativeMemory memory(1,256,false);check(false);}
        catch(const ps5hip::NativeMemorySetupError& error){check(error.retained_backing);}
        fail_map=fail_release=0;
        auto map_retained=native_memory_stats();
        check(map_retained.allocations==after.allocations+1&&map_retained.releases==after.releases&&map_retained.live_bytes==0x4000);
        mapped_address=1;fail_unmap=1;
        try {NativeMemory memory(0x4001,256,false);check(false);}
        catch(const ps5hip::NativeMemorySetupError& error){check(error.retained_backing);}
        fail_unmap=0;
        auto unmap_retained=native_memory_stats();
        check(unmap_retained.allocations==after.allocations+2&&unmap_retained.releases==after.releases&&unmap_retained.live_bytes==0xc000);
        check(unmap_retained.peak_bytes==after.peak_bytes);
        std::cout << "native allocation mock: mapping, cleanup failures and GPU ownership checks passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
