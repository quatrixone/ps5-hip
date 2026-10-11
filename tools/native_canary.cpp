#include "ps5hip/agc_backend.h"
#include "ps5hip/submission.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unistd.h>

#ifndef PS5HIP_CANARY_TARGET
#error The native experiment must select an explicit compiler target
#endif

namespace {
constexpr uint32_t count = 73, guard_words = 16, poison = 0xa5a5a5a5;
using Memory = std::shared_ptr<ps5hip::NativeMemory>;

Memory buffer() {
    auto memory = std::make_shared<ps5hip::NativeMemory>((count + 2 * guard_words) * 4, 256, false);
    std::memset(memory->cpu(), 0xa5, memory->range().bytes);
    return memory;
}
uint32_t *data(const Memory &memory) {
    return static_cast<uint32_t *>(memory->cpu()) + guard_words;
}
ps5hip::LaunchArgument argument(const Memory &memory) {
    return {memory->range().address + guard_words * 4, count * 4};
}
void check_guards(const Memory &memory) {
    const auto *words = static_cast<const uint32_t *>(memory->cpu());
    for (size_t i = 0; i < memory->range().bytes / 4; ++i)
        if ((i < guard_words || i >= guard_words + count) && words[i] != poison)
            throw ps5hip::ObjectError("canary guard modified");
}
#ifdef PS5HIP_CANARY_VECTOR_ADD
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
#endif
[[noreturn]] void quarantine() {
    std::fprintf(stderr, "PS5HIP QUARANTINED: GPU ownership unresolved; no retry or local teardown\n");
    std::fflush(nullptr);
    for (;;) ::sleep(1);
}

void validate(const Memory &output
#ifdef PS5HIP_CANARY_VECTOR_ADD
              , const Memory &a, const Memory &b
#endif
) {
    check_guards(output);
    for (uint32_t i = 0; i < count; ++i) {
#ifdef PS5HIP_CANARY_VECTOR_ADD
        const uint32_t expected = bits(float(i) * 0.25f + 2.0f);
        if (data(a)[i] != bits(float(i) * 0.5f) || data(b)[i] != bits(-float(i) * 0.25f + 2.0f))
            throw ps5hip::ObjectError("read-only input modified");
#else
        const uint32_t expected = i ^ 0x5a17c9e3u;
#endif
        if (data(output)[i] != expected) {
            std::fprintf(stderr, "PS5HIP mismatch index=%u expected=0x%08x actual=0x%08x\n",
                         i, expected, data(output)[i]);
            throw ps5hip::ObjectError("canary output mismatch");
        }
    }
#ifdef PS5HIP_CANARY_VECTOR_ADD
    check_guards(a); check_guards(b);
#endif
}

void run(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object,
         const std::vector<uint8_t> &image) {
    auto module = std::make_shared<ps5hip::NativeMemory>(image.size(), 65536, true);
    auto kernarg = std::make_shared<ps5hip::NativeMemory>(64, 256, false);
    auto output = buffer();
    std::memcpy(module->cpu(), image.data(), image.size());
    std::vector<Memory> resources{module, kernarg, output};
    std::vector<ps5hip::LaunchArgument> arguments;
    const char *kernel = "indexed_store";
#ifdef PS5HIP_CANARY_VECTOR_ADD
    kernel = "vector_add";
    auto a = buffer(), b = buffer();
    for (uint32_t i = 0; i < count; ++i) {
        data(a)[i] = bits(float(i) * 0.5f);
        data(b)[i] = bits(-float(i) * 0.25f + 2.0f);
    }
    resources.push_back(a); resources.push_back(b);
    arguments = {argument(a), argument(b), argument(output), {count}};
#else
    arguments = {argument(output), {count}};
#endif
    const auto launch = ps5hip::prepare_initial_launch(object, kernel, module->range(),
                                                      kernarg->range(), 2, arguments);
    std::memcpy(kernarg->cpu(), launch.kernargs.data(), launch.kernargs.size());
    std::printf("PS5HIP SUBMIT target=%s kernel=%s count=%u groups=2 timeout_ms=2000\n",
                PS5HIP_CANARY_TARGET, kernel, count);
    std::fflush(nullptr);
    const auto result = queue.submit_and_wait(ps5hip::encode_initial_dispatch(launch), resources, 2000);
    if (result.status != ps5hip::SubmissionStatus::Complete) {
        std::fprintf(stderr, "PS5HIP submission status=%u driver=0x%08x\n",
                     unsigned(result.status), unsigned(result.driver_error));
        quarantine();
    }
#ifdef PS5HIP_CANARY_VECTOR_ADD
    validate(output, a, b);
#else
    validate(output);
#endif
    for (auto &memory : resources)
        if (!memory->close()) throw ps5hip::ObjectError("canary allocation teardown failed");
    std::printf("PS5HIP PASS kernel=%s exact_values=%u guards=unchanged allocations=closed\n", kernel, count);
}
} // namespace

int main() {
    try {
        // Exercise exception unwinding before entering the GPU driver.
        bool caught = false;
        try { throw ps5hip::ObjectError("native runtime check"); }
        catch (const ps5hip::ObjectError &) { caught = true; }
        if (!caught) return 2;
        std::ifstream input("/app0/canary.hsaco", std::ios::binary | std::ios::ate);
        if (!input) throw ps5hip::ObjectError("cannot open staged canary.hsaco");
        const auto size = input.tellg();
        if (size < 64 || size > 64 * 1024 * 1024) throw ps5hip::ObjectError("invalid canary object size");
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char *>(bytes.data()), size)) throw ps5hip::ObjectError("cannot read canary");
        const auto object = ps5hip::inspect_code_object(bytes.data(), bytes.size());
        if (object.target != PS5HIP_CANARY_TARGET) throw ps5hip::ObjectError("staged target mismatch");
        const auto image = ps5hip::materialize_code_object(bytes.data(), bytes.size());
        ps5hip::AgcBackend backend;
        ps5hip::SubmissionQueue queue(backend);
        try { run(queue, object, image); }
        catch (...) { if (queue.poisoned()) quarantine(); throw; }
        std::fflush(nullptr);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "PS5HIP FAIL: %s\n", error.what());
        std::fflush(nullptr);
        return 1;
    }
}
