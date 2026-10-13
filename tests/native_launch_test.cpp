#include "ps5hip/native_launch.h"
#include "ps5hip/host_launch.h"
#include "ps5hip/submission.h"
#include <ps5platform/kernel.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
struct Allocation { uint64_t bytes, alignment; void *cpu = nullptr; bool alias = false, unmapped = false; };
std::map<int64_t, Allocation> allocations;
int64_t next_id = 1;
unsigned allocate_calls = 0, releases = 0;
unsigned fail_allocate_at = 0, fail_map_at = 0, alias_map_at = 0;
bool fail_unmap = false, fail_release = false;
void checked(bool yes, unsigned line) { if (!yes) throw std::runtime_error("native launch model assertion at line " + std::to_string(line)); }
#define check(value) checked((value), __LINE__)
template<class F> void reject(F action) {
    try { action(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid native launch accepted");
}
uint64_t get(const uint8_t *p, size_t width) {
    uint64_t result = 0;
    for (size_t i = 0; i < width; ++i) result |= uint64_t(p[i]) << (8*i);
    return result;
}
struct Buffers {
    ps5hip::NativeDataRegistry data;
    std::array<uint64_t, 3> pointers{};
    std::array<void *, 3> bases{};
    uint32_t n = 73;
    explicit Buffers() {
        for (size_t i = 0; i < 3; ++i) {
            bases[i] = data.allocate(1024);
            pointers[i] = reinterpret_cast<uintptr_t>(bases[i]) + 64;
        }
    }
    std::vector<ps5hip::LaunchArgument> capture(const ps5hip::Kernel &kernel) {
        std::array<void *, 4> array{{&pointers[0], &pointers[1], &pointers[2], &n}};
        std::vector<ps5hip::ReadableSpan> spans;
        for (size_t i = 0; i < 3; ++i) spans.push_back({reinterpret_cast<const uint8_t *>(&pointers[i]), 8});
        spans.push_back({reinterpret_cast<const uint8_t *>(&n), 4});
        return ps5hip::capture_initial_arguments(kernel, {reinterpret_cast<const uint8_t *>(array.data()), sizeof(array)},
            spans, data.ranges());
    }
};
struct TransportModel : ps5hip::SubmissionBackend {
    unsigned submits = 0, flushes = 0;
    uint64_t clock = 0;
    int failure = 0;
    volatile uint32_t *marker = nullptr;
    ps5hip::NativeDataRegistry *data = nullptr;
    void *base = nullptr;
    void flush(void *, size_t) override { if (failure == 1 && ++flushes == 1) throw ps5hip::ObjectError("clean flush failure"); }
    int32_t submit(uint32_t *words, uint32_t count) override {
        ++submits;
        check(count > 10);
        try { data->free(base); throw std::runtime_error("in-flight free accepted"); }
        catch (const ps5hip::NativeDataError &error) { check(error.status() == ps5hip::NativeDataStatus::InUse); }
        const auto *tail = words + count - 10;
        check(tail[0] == 0xc0004600 && tail[2] == 0xc0064900);
        marker = reinterpret_cast<volatile uint32_t *>(uint64_t(tail[5]) | uint64_t(tail[6]) << 32);
        if (failure == 2) return -1;
        if (failure == 3) throw ps5hip::ObjectError("uncertain submit exception");
        if (failure != 4) *marker = 1;
        return 0;
    }
    int32_t kick() override { return 0; }
    uint64_t now_ns() override { return clock; }
    void sleep_us(unsigned us) override { clock += uint64_t(us) * 1000; }
};
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int type, int64_t *physical) {
    check(type == 12);
    if (++allocate_calls == fail_allocate_at) return -1;
    *physical = next_id++;
    allocations.emplace(*physical, Allocation{bytes, alignment});
    return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t bytes, int protection, int flags, int64_t physical, size_t alignment) {
    check(protection == 0x33 && flags == 0);
    if (allocate_calls == fail_map_at) return -1;
    auto &a = allocations.at(physical);
    check(a.bytes == bytes && a.alignment == alignment && alignment <= 0x200000);
#ifdef PS5HIP_NATIVE_ROUTES_MODEL
    check(bytes <= 0x2000000);
#else
    check(bytes <= 0x200000);
#endif
    if (allocate_calls == alias_map_at) {
        a.cpu = allocations.begin()->second.cpu; a.alias = true;
    } else {
        // Real host pages, no ISA execution/emulation. ASan reserves this native
        // shader window in its gap; the dedicated model remaps only its bounded
        // disjoint pages with protect_shadow_gap=0. Normal builds use NOREPLACE.
        auto target = 0x200000000ull + uint64_t(physical) * 0x200000;
#ifdef PS5HIP_NATIVE_ROUTES_MODEL
        // Reuse bounded disjoint slots across sequential scratch preparations.
        target = 0x202000000ull;
        while (std::any_of(allocations.begin(),allocations.end(),[&](const auto &entry) {
            return !entry.second.unmapped && uint64_t(entry.second.cpu)==target;
        })) target += 0x2000000;
#endif
        check(target >= 0x200000000ull && target < 0x300000000ull && bytes <= 0x300000000ull-target);
        int map_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE;
#ifndef __has_feature
#define __has_feature(value) 0
#endif
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
        const char *options = std::getenv("ASAN_OPTIONS");
        check(options && std::strstr(options, "protect_shadow_gap=0"));
        map_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
#endif
        a.cpu = mmap(reinterpret_cast<void *>(target), bytes, PROT_READ | PROT_WRITE, map_flags, -1, 0);
        check(a.cpu != MAP_FAILED);
        std::memset(a.cpu, 0xa5, bytes);
    }
    *address = a.cpu; return 0;
}
extern "C" int32_t sceKernelMunmap(void *address, size_t bytes) {
    if (fail_unmap) return -1;
    // The faulty-alias model owns no second host mapping; prefer its record.
    for (auto i = allocations.rbegin(); i != allocations.rend(); ++i)
        if (i->second.cpu == address && !i->second.unmapped) {
            auto &a = i->second;
            check(a.bytes == bytes);
            if (!a.alias) check(munmap(address, bytes) == 0);
            a.unmapped = true; return 0;
        }
    return -1;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t bytes) {
    if (fail_release) return -1;
    check(allocations.at(physical).bytes == bytes);
    allocations.erase(physical); ++releases; return 0;
}
int main(int argc, char **argv) {
    try {
        check(argc == 3 && std::string(argv[2]) == "gfx1013");
        std::ifstream stream(argv[1], std::ios::binary); check(bool(stream));
        const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(stream), {}};
        auto bundle = ps5hip::read_fat_binary(bytes.data(), bytes.size(), "gfx1013");
        const auto &kernel = bundle.object.kernels.front();
        {
            Buffers buffers;
            auto values = buffers.capture(kernel);
            const auto original = values;
            auto prepared = ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data);
            check(prepared.resources.size() == 5 && !prepared.commands.empty() && allocations.size() == 5);
            check(prepared.temporaries.size() == 2 && prepared.temporaries[0] == prepared.resources[0] &&
                prepared.temporaries[1] == prepared.resources[1]);
            buffers.pointers.fill(0); buffers.n = 0; values.clear();
            const auto *args = static_cast<const uint8_t *>(prepared.resources[1]->cpu());
            for (size_t i = 0; i < 3; ++i) check(get(args+i*8, 8) == original[i].value);
            check(get(args+24, 4) == 73);
            std::array<uint8_t, 288> expected{};
            auto put = [&](size_t at, uint64_t value, size_t width) {
                for (size_t i = 0; i < width; ++i) expected.at(at+i) = uint8_t(value >> (i*8));
            };
            for (size_t i = 0; i < 3; ++i) put(i*8, original[i].value, 8);
            put(24, 73, 4); put(32, 2, 4); put(36, 1, 4); put(40, 1, 4);
            put(44, 64, 2); put(46, 1, 2); put(48, 1, 2); put(96, 1, 2);
            check(std::memcmp(args, expected.data(), expected.size()) == 0);
            const auto image = ps5hip::materialize_code_object(bundle.bytes.data(), bundle.bytes.size());
            check(std::memcmp(prepared.resources[0]->cpu(), image.data(), image.size()) == 0);
            check(prepared.resources[0]->range().address % 0x10000 == 0);
            TransportModel model; model.data = &buffers.data; model.base = buffers.bases[0];
            ps5hip::SubmissionQueue queue(model);
            check(queue.submit_and_wait(prepared.commands, prepared.resources, 1).status == ps5hip::SubmissionStatus::Complete);
            for (auto &owner : prepared.resources) check(!owner->gpu_in_use());
            ps5hip::close_native_launch_temporaries(prepared,buffers.data);
            for (auto *base : buffers.bases) buffers.data.free(base);
            reject([&] { queue.submit_and_wait(prepared.commands, prepared.resources, 1); });
            check(model.submits == 1);
        }
        check(allocations.empty());
        {
            Buffers buffers; auto values = buffers.capture(kernel);
            auto count = allocate_calls;
            auto copied_kernel = kernel;
            reject([&] { ps5hip::prepare_native_vector(bundle, copied_kernel, values, buffers.data); });
            auto changed = bundle; changed.object.kernels.front().kernarg_size = 32;
            reject([&] { ps5hip::prepare_native_vector(changed, changed.object.kernels.front(), values, buffers.data); });
            for (size_t i = 0; i < 4; ++i) {
                auto bad = values;
                if (i == 3) bad[i].value = 74; else ++bad[i].buffer_bytes;
                reject([&] { ps5hip::prepare_native_vector(bundle, kernel, bad, buffers.data); });
            }
            auto bad = values; bad[0].value = UINT64_MAX;
            reject([&] { ps5hip::prepare_native_vector(bundle, kernel, bad, buffers.data); });
            bad = values; bad[0].value = values[0].value + 1024 - 64 - 4; bad[0].buffer_bytes = 4;
            reject([&] { ps5hip::prepare_native_vector(bundle, kernel, bad, buffers.data); });
            check(allocate_calls == count);
            auto hold = buffers.data.resolve(buffers.bases[0], 1); hold.memory->acquire_gpu_use();
            reject([&] { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); });
            hold.memory->complete_gpu_use();
            // Equal input pointers are a single backing hold; output remains separate.
            values[1] = values[0];
            auto prepared = ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data);
            check(prepared.resources.size() == 4);
            ps5hip::close_native_launch_temporaries(prepared,buffers.data);
            buffers.data.free(buffers.bases[0]);
            reject([&] { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); });
        }
        check(allocations.empty());
        for (unsigned stage : {1u, 2u}) {
            Buffers buffers; auto values = buffers.capture(kernel);
            fail_allocate_at = allocate_calls + stage;
            reject([&] { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); });
            fail_allocate_at = 0;
            check(!buffers.data.poisoned() && allocations.size() == 3);
            check(buffers.data.resolve(buffers.bases[0], 1024).remaining == 1024);
        }
        check(allocations.empty());
        {
            Buffers buffers; auto values = buffers.capture(kernel);
            TransportModel model; model.failure = 1; model.data = &buffers.data; model.base = buffers.bases[0];
            ps5hip::SubmissionQueue queue(model);
            auto prepared = ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data);
            reject([&] { queue.submit_and_wait(prepared.commands, prepared.resources, 1); });
            check(!queue.poisoned() && !model.submits && !buffers.data.poisoned());
            for (auto &owner : prepared.resources) check(!owner->gpu_in_use());
            ps5hip::close_native_launch_temporaries(prepared,buffers.data);
        }
        check(allocations.empty());
        // Each unresolved ownership/faulty-provider case ends its own process.
        for (int mode = 0; mode < 6; ++mode) {
            const auto child = fork(); check(child >= 0);
            if (!child) {
                Buffers buffers; auto values = buffers.capture(kernel);
                if (mode < 3) {
                    TransportModel model; model.failure = mode + 2; model.data = &buffers.data; model.base = buffers.bases[0];
                    ps5hip::SubmissionQueue queue(model);
                    auto prepared = ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data);
                    const auto before = releases;
                    if (mode == 1) reject([&] { queue.submit_and_wait(prepared.commands, prepared.resources, 1); });
                    else check(queue.submit_and_wait(prepared.commands, prepared.resources, 1).status != ps5hip::SubmissionStatus::Complete);
                    check(queue.poisoned()); buffers.data.poison();
                    for (auto &owner : prepared.resources) check(owner->gpu_in_use() && !owner->close());
                    check(releases == before && allocations.size() == 7);
                    reject([&] { buffers.data.free(buffers.bases[0]); });
                    reject([&] { queue.submit_and_wait(prepared.commands, prepared.resources, 1); });
                    check(model.submits == 1);
                } else if (mode == 3) {
                    fail_map_at = allocate_calls + 2; fail_release = true;
                    reject([&] { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); });
                    check(buffers.data.poisoned());
                } else if (mode == 4) {
                    fail_allocate_at = allocate_calls + 2; fail_unmap = true;
                    try { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); check(false); }
                    catch (const ps5hip::NativeMemoryCleanupError &) { check(buffers.data.poisoned()); }
                } else {
                    alias_map_at = allocate_calls + 2;
                    reject([&] { ps5hip::prepare_native_vector(bundle, kernel, values, buffers.data); });
                    check(buffers.data.poisoned());
                }
                _exit(0);
            }
            int status = 0;
            check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
        check(allocations.empty());
        std::cout << "native launch ownership: measured fixture, 288-byte oracle, holds, clean unwind and quarantine passed; host transport model only\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
