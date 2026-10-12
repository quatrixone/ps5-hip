#pragma once

// Included inside the native harness namespace after its ownership helpers.
namespace aperture {
constexpr uint32_t lanes = 64, words_per_lane = 6, output_words = lanes * words_per_lane;
constexpr uint32_t sentinel = 0x6b18d3a5;
inline uint32_t raw_poison(uint32_t lane, uint32_t word) {
    return 0xc0000000u ^ (lane * 0x01020401u) ^ (word * 0x11223344u);
}

inline Memory buffer() {
    auto memory = std::make_shared<ps5hip::NativeMemory>((output_words + 2 * guard_words) * 4, 256, false);
    std::memset(memory->cpu(), 0xa5, memory->range().bytes);
    // Missing raw stores must not look like a uniform register observation.
    auto *output = data(memory);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        for (uint32_t word = 0; word < 4; ++word)
            output[lane * words_per_lane + word] = raw_poison(lane, word);
    return memory;
}
inline void validate(const Memory &memory) {
    const auto *words = static_cast<const uint32_t *>(memory->cpu());
    if (memory->range().bytes < (output_words + 2 * guard_words) * 4)
        throw ps5hip::ObjectError("aperture output allocation too small");
    for (size_t i = 0; i < memory->range().bytes / 4; ++i)
        if ((i < guard_words || i >= guard_words + output_words) && words[i] != poison)
            throw ps5hip::ObjectError("aperture output guard modified");
    const auto *output = data(memory);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const auto *record = output + lane * words_per_lane;
        if (record[4] != lane || record[5] != (lane ^ sentinel))
            throw ps5hip::ObjectError("aperture lane or sentinel mismatch");
        const auto *wave = output + (lane / 32) * 32 * words_per_lane;
        for (unsigned word = 0; word < 4; ++word)
            if (record[word] != wave[word])
                throw ps5hip::ObjectError("aperture registers differ within wave");
    }
}
inline void report(const Memory &memory) {
    if (memory->range().bytes < (output_words + 2 * guard_words) * 4)
        throw ps5hip::ObjectError("aperture output allocation too small");
    const auto *output = data(memory);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const auto *record = output + lane * words_per_lane;
        std::printf("PS5HIP APERTURE lane=%u base=0x%08x%08x limit=0x%08x%08x sentinel=0x%08x\n",
                    lane, record[1], record[0], record[3], record[2], record[5]);
    }
    // Preserve all raw observations before diagnosing inconsistent transport.
    validate(memory);
    const bool equal = std::memcmp(output, output + 32 * words_per_lane, 4 * sizeof(uint32_t)) == 0;
    std::printf("PS5HIP APERTURE waves=2 within_wave=consistent across_waves=%s interpretation=raw-only\n",
                equal ? "equal" : "different");
}
inline void run(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object,
                const std::vector<uint8_t> &image) {
    auto module = std::make_shared<ps5hip::NativeMemory>(image.size(), 65536, true);
    auto kernarg = std::make_shared<ps5hip::NativeMemory>(64, 256, false);
    auto output = buffer();
    std::memcpy(module->cpu(), image.data(), image.size());
    std::vector<Memory> resources{module, kernarg, output};
    const auto launch = ps5hip::prepare_initial_launch(object, "private_aperture", module->range(),
        kernarg->range(), 1, {{output->range().address + guard_words * 4, output_words * 4}});
    std::memcpy(kernarg->cpu(), launch.kernargs.data(), launch.kernargs.size());
    std::printf("PS5HIP SUBMIT target=%s kernel=private_aperture count=64 groups=1 timeout_ms=2000\n",
                PS5HIP_CANARY_TARGET);
    std::fflush(nullptr);
    const auto result = queue.submit_and_wait(ps5hip::encode_initial_dispatch(launch), resources, 2000);
    if (result.status != ps5hip::SubmissionStatus::Complete) {
        std::fprintf(stderr, "PS5HIP submission status=%u driver=0x%08x\n",
                     unsigned(result.status), unsigned(result.driver_error));
        quarantine();
    }
    report(output);
    for (auto &memory : resources)
        if (!memory->close()) throw ps5hip::ObjectError("aperture allocation teardown failed");
    std::printf("PS5HIP PASS kernel=private_aperture exact_values=128 raw_words=256 guards=unchanged allocations=closed\n");
}
} // namespace aperture
