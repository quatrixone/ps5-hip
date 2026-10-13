// Included inside the native canary namespace; uses its guarded allocation helpers.
namespace visibility {
constexpr uint32_t elements = 257, rounds = 3, groups = 5;
uint32_t input_value(uint32_t round, uint32_t i) {
    return (0xfedcba98u + round * 0x31415927u) ^ (i * 0x1020305u);
}
uint32_t produced(uint32_t round, uint32_t i) {
    return input_value(round, i) * 17u + i * 31u + 0x13579bdfu;
}
void validate_round(const Memory &input, const Memory &middle, const Memory &output, uint32_t round) {
    check_guards(input, elements); check_guards(middle, elements); check_guards(output, elements);
    for (uint32_t i = 0; i < elements; ++i) {
        const uint32_t expected = (produced(round, i) ^ 0xa39d71b5u) + produced(round, (i + 113u) % elements) * 3u;
        if (data(input)[i] != input_value(round, i) || data(middle)[i] != produced(round, i) || data(output)[i] != expected)
            throw ps5hip::ObjectError("dependent visibility result mismatch");
    }
}
void run(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object, const std::vector<uint8_t> &image) {
    auto module = std::make_shared<ps5hip::NativeMemory>(image.size(), 65536, true);
    auto kernarg = std::make_shared<ps5hip::NativeMemory>(16, 256, false);
    auto input = buffer(elements), middle = buffer(elements), output = buffer(elements);
    const std::vector<Memory> resources{module, kernarg, input, middle, output};
    std::memcpy(module->cpu(), image.data(), image.size());
    try {
        for (uint32_t round = 0; round < rounds; ++round) {
            for (const auto &memory : resources)
                if (memory->gpu_in_use()) throw ps5hip::ObjectError("CPU ownership not restored");
            for (uint32_t i = 0; i < elements; ++i) {
                data(input)[i] = input_value(round, i);
                data(output)[i] = poison;
            }
            // Never write middle on the CPU after allocation, including across rounds.
            for (unsigned phase = 0; phase < 2; ++phase) {
                const char *kernel = phase ? "visibility_consumer" : "visibility_producer";
                const std::vector<ps5hip::LaunchArgument> arguments = phase
                    ? std::vector<ps5hip::LaunchArgument>{argument(middle, elements), argument(output, elements)}
                    : std::vector<ps5hip::LaunchArgument>{argument(input, elements), argument(middle, elements)};
                const auto launch = ps5hip::prepare_initial_launch(object, kernel, module->range(), kernarg->range(), groups, arguments);
                if (launch.kernargs.size() != 16) throw ps5hip::ObjectError("visibility kernarg contract changed");
                std::memcpy(kernarg->cpu(), launch.kernargs.data(), launch.kernargs.size());
                std::printf("PS5HIP SUBMIT kernel=%s round=%u/3 count=257 groups=5 timeout_ms=2000\n", kernel, round + 1);
                std::fflush(nullptr);
                const auto result = queue.submit_and_wait(ps5hip::encode_initial_dispatch(launch), resources, 2000);
                if (result.status != ps5hip::SubmissionStatus::Complete) quarantine("dependent visibility submission failed");
            }
            validate_round(input, middle, output, round);
            std::printf("PS5HIP ROUND kernel=dependent_visibility index=%u/3 exact_values=257 inputs=unchanged guards=unchanged\n", round + 1);
        }
    } catch (const std::exception &error) {
        if (queue.poisoned()) quarantine(error.what());
        throw;
    } catch (...) {
        if (queue.poisoned()) quarantine("unknown exception");
        throw;
    }
    for (const auto &memory : resources)
        if (!memory->close()) throw ps5hip::ObjectError("visibility allocation teardown failed");
    std::printf("PS5HIP PASS kernel=dependent_visibility dispatches=6 rounds=3 exact_values=771 persistent_buffers=3 guards=unchanged allocations=closed\n");
}
} // namespace visibility
