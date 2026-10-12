#pragma once

// Included inside the native harness namespace after its ownership helpers.
// Fixed diagnostic for the separately pinned six-entry compiler fixture.
namespace atomics {
constexpr uint32_t groups = 257, invocations = groups * 64;
constexpr const char *kernels[] = {"atomic_ticket32", "atomic_ticket64", "atomic_cas32",
                                  "atomic_cas64", "atomic_indirect32", "atomic_indirect64"};

inline Memory buffer(size_t payload) {
    auto memory = std::make_shared<ps5hip::NativeMemory>(payload + 2 * guard_words * 4, 256, false);
    std::memset(memory->cpu(), 0xa5, memory->range().bytes);
    return memory;
}
inline void guards(const Memory &memory, size_t payload) {
    if (memory->range().bytes < payload + 2 * guard_words * 4)
        throw ps5hip::ObjectError("atomic allocation too small");
    const auto *bytes = static_cast<const uint8_t *>(memory->cpu());
    for (size_t i = 0; i < memory->range().bytes; ++i)
        if ((i < guard_words * 4 || i >= guard_words * 4 + payload) && bytes[i] != 0xa5)
            throw ps5hip::ObjectError("atomic guard modified");
}
template<class T> T *values(const Memory &memory) { return reinterpret_cast<T *>(data(memory)); }
template<class T> constexpr T initial() { return T(T(~T(0)) - T(4095)); }
template<class T> constexpr T token_base() { return sizeof(T) == 8 ? T(0x13579bdf00000000ull) : T(0); }

template<class T> void validate(const Memory &counter, const Memory &output, bool cas,
                                const Memory &offsets) {
    guards(counter, sizeof(T)); guards(output, invocations * sizeof(T));
    const T final = *values<T>(counter);
    const auto *observed = values<T>(output);
    if (cas) {
        const uint64_t id = uint64_t(final) - uint64_t(token_base<T>());
        if (id < 1 || id > invocations || final != T(token_base<T>() | T(id)))
            throw ps5hip::ObjectError("atomic CAS winner token invalid");
        for (uint32_t i = 0; i < invocations; ++i)
            if (observed[i] != (i == id - 1 ? T(0) : final))
                throw ps5hip::ObjectError("atomic CAS observations inconsistent");
    } else {
        if (final != T(initial<T>() + T(invocations)))
            throw ps5hip::ObjectError("atomic final counter mismatch");
        std::vector<uint8_t> seen(invocations, 0);
        for (uint32_t i = 0; i < invocations; ++i) {
            const T ticket = T(observed[i] - initial<T>());
            if (ticket >= invocations || seen[static_cast<size_t>(ticket)]++)
                throw ps5hip::ObjectError("atomic ticket missing or duplicated");
        }
    }
    if (offsets) {
        guards(offsets, invocations * sizeof(uint32_t));
        for (uint32_t i = 0; i < invocations; ++i)
            if (data(offsets)[i] != 0) throw ps5hip::ObjectError("atomic offsets modified");
    }
}

template<class T> void run_one(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object,
                               const std::vector<uint8_t> &image, unsigned index) {
    const bool cas = index == 2 || index == 3, indirect = index >= 4;
    auto module = std::make_shared<ps5hip::NativeMemory>(image.size(), 65536, true);
    auto kernarg = std::make_shared<ps5hip::NativeMemory>(64, 256, false);
    auto counter = buffer(sizeof(T)), output = buffer(invocations * sizeof(T));
    *values<T>(counter) = cas ? T(0) : initial<T>();
    std::memcpy(module->cpu(), image.data(), image.size());
    std::vector<Memory> resources{module, kernarg, counter, output};
    std::vector<ps5hip::LaunchArgument> args{
        {counter->range().address + guard_words * 4, sizeof(T)},
        {output->range().address + guard_words * 4, invocations * sizeof(T)}};
    Memory offsets;
    if (indirect) {
        offsets = buffer(invocations * sizeof(uint32_t));
        std::memset(data(offsets), 0, invocations * sizeof(uint32_t));
        resources.push_back(offsets);
        args.push_back({offsets->range().address + guard_words * 4, invocations * sizeof(uint32_t)});
    }
    const auto launch = ps5hip::prepare_initial_launch(object, kernels[index], module->range(),
                                                       kernarg->range(), groups, args);
    std::memcpy(kernarg->cpu(), launch.kernargs.data(), launch.kernargs.size());
    std::printf("PS5HIP SUBMIT target=%s kernel=%s count=16448 groups=257 timeout_ms=2000\n",
                PS5HIP_CANARY_TARGET, kernels[index]);
    std::fflush(nullptr);
    const auto result = queue.submit_and_wait(ps5hip::encode_initial_dispatch(launch), resources, 2000);
    if (result.status != ps5hip::SubmissionStatus::Complete) {
        std::fprintf(stderr, "PS5HIP submission status=%u driver=0x%08x\n",
                     unsigned(result.status), unsigned(result.driver_error));
        quarantine();
    }
    validate<T>(counter, output, cas, offsets);
    for (auto &memory : resources)
        if (!memory->close()) throw ps5hip::ObjectError("atomic allocation teardown failed");
    std::printf("PS5HIP PASS kernel=%s exact_values=16449 guards=unchanged offsets=%s allocations=closed\n",
                kernels[index], indirect ? "unchanged" : "absent");
}
inline void run(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object,
                const std::vector<uint8_t> &image) {
    for (unsigned index = 0; index < 6; ++index) {
        std::printf("PS5HIP DISPATCH index=%u/6 kernel=%s\n", index + 1, kernels[index]);
        if (index % 2) run_one<uint64_t>(queue, object, image, index);
        else run_one<uint32_t>(queue, object, image, index);
    }
    std::printf("PS5HIP PASS kernel=atomic_suite dispatches=6 exact_values=98694 guards=unchanged allocations=closed\n");
}
} // namespace atomics
