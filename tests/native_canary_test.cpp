// Exercise the actual harness oracle without running its title entry point.
#include "../tools/native_runtime.h"
extern "C" unsigned mock_quarantine_sleep(unsigned);
#define sleep mock_quarantine_sleep
#define main native_title_main_not_called
#include "../tools/native_canary.cpp"
#undef main
#undef sleep
#include <ps5platform/kernel.h>
#include <cstdlib>
#include <map>
#include <sys/wait.h>

namespace {
std::map<int64_t, void *> allocations;
int64_t next_id = 1;
void require(bool value) { if (!value) throw std::runtime_error("canary test assertion"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("bad canary result accepted");
}
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int, int64_t *physical) {
    void *memory = nullptr;
    if (posix_memalign(&memory, alignment, bytes)) return -1;
    *physical = next_id++; allocations[*physical] = memory; return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t, int, int, int64_t physical, size_t) {
    *address = allocations.at(physical); return 0;
}
extern "C" int32_t sceKernelMunmap(void *, size_t) { return 0; }
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t) {
    std::free(allocations.at(physical)); allocations.erase(physical); return 0;
}
// An accidental entry into the GPU path must fail this host test immediately.
extern "C" int32_t sceAgcInit(uint32_t) { std::abort(); }
extern "C" int32_t sceAgcDriverSubmitDcb(void *) { std::abort(); }
extern "C" int32_t sceAgcSuspendPoint() { std::abort(); }
extern "C" unsigned mock_quarantine_sleep(unsigned seconds) { ::_exit(seconds == 1 ? 0 : 4); }

int main() {
    try {
        for (const char *reason : {static_cast<const char *>(nullptr), "monotonic clock failed", "unknown exception"}) {
            FILE *log = std::tmpfile();
            require(log != nullptr);
            const pid_t child = ::fork();
            if (child == 0) {
                if (::dup2(::fileno(log), STDOUT_FILENO) < 0 || ::dup2(::fileno(log), STDERR_FILENO) < 0)
                    ::_exit(2);
                quarantine(reason);
            }
            int status = 0;
            require(child > 0 && ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
            std::rewind(log);
            char text[512]{};
            const size_t bytes = std::fread(text, 1, sizeof(text), log);
            std::fclose(log);
            const std::string expected = (reason ? std::string("PS5HIP FAIL: submission exception: ") + reason + "\n" : "") +
                "PS5HIP QUARANTINED: GPU ownership unresolved; no retry or local teardown\n";
            require(std::string(text, bytes) == expected);
        }
        {
            auto output = buffer();
            require(argument(output).value == output->range().address + 64);
            require(argument(output).buffer_bytes == count * 4);
#ifdef PS5HIP_CANARY_LDS
            check_guards(output, count);
            require(count == 257 && groups == 257 && input_count == 16448);
            auto input = buffer(input_count);
            auto verify = [&] { validate(output, input); };
            reject(verify);
            for (uint32_t group = 0; group < 257; ++group) {
                uint32_t sum = 0;
                for (uint32_t lane = 0; lane < 64; ++lane) {
                    const uint32_t value = 4099 * group + 17 * lane + 65536 * (lane / 32);
                    data(input)[64 * group + lane] = value;
                    sum += value;
                }
#ifdef PS5HIP_CANARY_LDS_MIXED
                sum ^= (64 * group + 33) ^ 1403613629u;
#endif
                data(output)[group] = sum;
            }
            verify();
            for (uint32_t i = 0; i < 257; ++i) {
                data(output)[i] ^= 1; reject(verify); data(output)[i] ^= 1;
            }
            std::swap(data(output)[0], data(output)[256]); reject(verify);
            std::swap(data(output)[0], data(output)[256]);
            for (uint32_t i : {0u, 31u, 32u, 63u, 64u, 16447u}) {
                data(input)[i] ^= 1; reject(verify); data(input)[i] ^= 1;
            }
            auto *input_words = static_cast<uint32_t *>(input->cpu());
            for (size_t i : {size_t(0), size_t(15), size_t(16 + 16448), size_t(input->range().bytes / 4 - 1)}) {
                input_words[i] ^= 1; reject(verify); input_words[i] ^= 1;
            }
#elif defined(PS5HIP_CANARY_VECTOR_ADD)
            auto a = buffer(), b = buffer();
            for (uint32_t i = 0; i < count; ++i) {
                data(a)[i] = bits(float(i) / 2);
                data(b)[i] = bits(2 - float(i) / 4);
                data(output)[i] = bits(float(i) / 4 + 2);
            }
            auto verify = [&] { validate(output, a, b); };
            data(a)[37] ^= 1; reject(verify); data(a)[37] ^= 1;
            data(b)[72] ^= 1; reject(verify); data(b)[72] ^= 1;
            data(a)[-1] ^= 1; reject(verify); data(a)[-1] ^= 1;
            data(b)[count] ^= 1; reject(verify); data(b)[count] ^= 1;
#elif defined(PS5HIP_CANARY_REGISTER_SPILL)
            auto input = buffer();
            auto verify = [&] { validate(output, input); };
            reject(verify);
            for (uint32_t i = 0; i < count; ++i) {
                // Independently evaluate the 32-bit permutation of each element index.
                const auto expected = uint32_t(uint64_t(i) * 2654435769ULL) ^ 1821285621u;
                data(input)[i] = expected; data(output)[i] = expected;
            }
            verify();
            for (uint32_t i = 0; i < count; ++i) {
                data(output)[i] ^= 1; reject(verify); data(output)[i] ^= 1;
            }
            for (uint32_t i : {0u, 31u, 32u, 63u, 64u, count - 1}) {
                data(input)[i] ^= 1; reject(verify); data(input)[i] ^= 1;
            }
            std::swap(data(output)[0], data(output)[64]); reject(verify);
            std::swap(data(output)[0], data(output)[64]);
            data(input)[-1] ^= 1; reject(verify); data(input)[-1] ^= 1;
            data(input)[count] ^= 1; reject(verify); data(input)[count] ^= 1;
            const uint32_t bytes = ps5hip::buffer_scratch_ring_bytes(scratch_private_bytes);
            require(bytes == 9175040);
            auto scratch = std::make_shared<ps5hip::NativeMemory>(bytes + 2 * scratch_prefix_bytes, 65536, false);
            std::memset(scratch->cpu(), 0xa5, scratch->range().bytes);
            const ps5hip::GpuRange payload{scratch->range().address + scratch_prefix_bytes, bytes};
            auto scratch_verify = [&] { check_scratch(scratch, payload); };
            reject(scratch_verify);
            auto *scratch_words = static_cast<uint32_t *>(scratch->cpu());
            const size_t first = scratch_prefix_bytes / 4, end = first + bytes / 4;
            scratch_words[first] ^= 1; scratch_words[end - 1] ^= 1;
            scratch_verify();
            for (size_t i : {size_t(0), first - 1, end, scratch->range().bytes / 4 - 1}) {
                scratch_words[i] ^= 1; reject(scratch_verify); scratch_words[i] ^= 1;
            }
            reject([&] { check_scratch(scratch, {payload.address + 4, payload.bytes}); });
            reject([&] { check_scratch(scratch, {payload.address, payload.bytes - 4}); });
            scratch_verify();
#elif defined(PS5HIP_CANARY_WAVE)
            auto verify = [&] { validate(output); };
            reject(verify);
#ifdef PS5HIP_CANARY_WAVE_PARTIAL
            constexpr uint32_t first = 4, end = 20;
#else
            constexpr uint32_t first = 0, end = 32;
#endif
            for (uint32_t base : {0u, 32u}) {
                uint32_t sum = 0, ballot = 0;
                for (uint32_t lane = first; lane < end; ++lane) {
                    sum += base + lane + 1;
                    if ((base + lane + 1) % 2) ballot |= 1u << lane;
                }
                for (uint32_t lane = first; lane < end; ++lane) {
                    auto *record = data(output) + (base + lane) * 4;
                    record[0] = base + first + 1;
                    record[1] = base + lane + (lane % 2 ? 0 : 2);
                    record[2] = ballot;
                    record[3] = sum;
                }
            }
            verify();
            // Every component and every inactive sentinel must participate in validation.
            for (uint32_t i = 0; i < count; ++i) {
                data(output)[i] ^= 1; reject(verify); data(output)[i] ^= 1;
            }
#else
            auto verify = [&] { validate(output); };
            reject(verify); // A completion marker without shader writes must fail.
            for (uint32_t i = 0; i < count; ++i) data(output)[i] = i ^ 0x5a17c9e3u;
#endif
            verify();
            for (uint32_t i : {0u, 63u, 64u, 72u}) {
                data(output)[i] ^= 1; reject(verify); data(output)[i] ^= 1;
            }
            auto *words = static_cast<uint32_t *>(output->cpu());
            for (size_t i : {size_t(0), size_t(15), size_t(guard_words + count), size_t(output->range().bytes / 4 - 1)}) {
                words[i] ^= 1; reject(verify); words[i] ^= 1;
            }
            verify();
        }
        require(allocations.empty());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
