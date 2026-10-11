// Included inside the native canary namespace only for the experimental LDS profiles.
constexpr uint32_t count = 257, groups = 257, input_count = groups * 64;
constexpr uint32_t guard_words = 16, poison = 0xa5a5a5a5;
#ifdef PS5HIP_CANARY_LDS_STATIC
constexpr const char *lds_kernel = "lds_static";
constexpr uint32_t lds_static_bytes = 256, lds_dynamic_bytes = 0, lds_stride = 0;
#elif defined(PS5HIP_CANARY_LDS_DYNAMIC)
constexpr const char *lds_kernel = "lds_dynamic";
constexpr uint32_t lds_static_bytes = 0, lds_dynamic_bytes = 1280, lds_stride = 5;
#else
constexpr const char *lds_kernel = "lds_mixed";
constexpr uint32_t lds_static_bytes = 256, lds_dynamic_bytes = 1280, lds_stride = 5;
#endif
using Memory = std::shared_ptr<ps5hip::NativeMemory>;

uint32_t lds_input(uint32_t index) {
    const uint32_t group = index / 64, lane = index % 64;
    return group * 4099 + lane * 17 + (lane >= 32 ? 65536 : 0);
}
uint32_t lds_expected(uint32_t group) {
    uint32_t value = group * 262336 + 2131424;
#ifdef PS5HIP_CANARY_LDS_MIXED
    value ^= (group * 64 + 33) ^ 0x53a971bdu;
#endif
    return value;
}
Memory buffer(uint32_t words = count) {
    auto memory = std::make_shared<ps5hip::NativeMemory>((uint64_t(words) + 2 * guard_words) * 4, 256, false);
    std::memset(memory->cpu(), 0xa5, memory->range().bytes);
    return memory;
}
uint32_t *data(const Memory &memory) {
    return static_cast<uint32_t *>(memory->cpu()) + guard_words;
}
ps5hip::LaunchArgument argument(const Memory &memory, uint32_t words = count) {
    return {memory->range().address + guard_words * 4, uint64_t(words) * 4};
}
void check_guards(const Memory &memory, uint32_t count_words) {
    const auto *words = static_cast<const uint32_t *>(memory->cpu());
    if (memory->range().bytes < (uint64_t(count_words) + 2 * guard_words) * 4)
        throw ps5hip::ObjectError("invalid LDS global guard layout");
    for (size_t i = 0; i < memory->range().bytes / 4; ++i)
        if ((i < guard_words || i >= guard_words + count_words) && words[i] != poison)
            throw ps5hip::ObjectError("LDS global guard modified");
}
void validate(const Memory &output, const Memory &input) {
    check_guards(output, count); check_guards(input, input_count);
    for (uint32_t i = 0; i < input_count; ++i)
        if (data(input)[i] != lds_input(i)) throw ps5hip::ObjectError("read-only LDS input modified");
    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t expected = lds_expected(group);
        if (data(output)[group] != expected) {
            std::fprintf(stderr, "PS5HIP mismatch group=%u expected=0x%08x actual=0x%08x\n",
                         group, expected, data(output)[group]);
            throw ps5hip::ObjectError("LDS reduction mismatch");
        }
    }
}
void run(ps5hip::SubmissionQueue &queue, const ps5hip::CodeObject &object,
         const std::vector<uint8_t> &image) {
    auto module = std::make_shared<ps5hip::NativeMemory>(image.size(), 65536, true);
    auto kernarg = std::make_shared<ps5hip::NativeMemory>(64, 256, false);
    auto input = buffer(input_count), output = buffer(count);
    std::memcpy(module->cpu(), image.data(), image.size());
    for (uint32_t i = 0; i < input_count; ++i) data(input)[i] = lds_input(i);
    const std::vector<Memory> resources{module, kernarg, input, output};
    std::vector<ps5hip::LaunchArgument> arguments{
        argument(input, input_count), argument(output)};
    if (lds_stride) arguments.push_back({lds_stride});
    const auto launch = ps5hip::prepare_lds_launch(object, lds_kernel, module->range(),
        kernarg->range(), groups, arguments, lds_dynamic_bytes);
    const auto expected = ps5hip::plan_lds(lds_static_bytes, lds_dynamic_bytes);
    if (launch.lds.static_bytes != lds_static_bytes || launch.lds.dynamic_bytes != lds_dynamic_bytes ||
        launch.lds.allocated_bytes != expected.allocated_bytes || launch.lds.rsrc2_lds != expected.rsrc2_lds ||
        launch.dispatch.kernargs.size() != (lds_stride ? 20u : 16u))
        throw ps5hip::ObjectError("LDS fixture resource contract changed");
    const auto commands = ps5hip::encode_lds_dispatch(launch);
    std::memcpy(kernarg->cpu(), launch.dispatch.kernargs.data(), launch.dispatch.kernargs.size());
    std::printf("PS5HIP SUBMIT target=%s kernel=%s invocations=16448 output_words=257 groups=257 static_lds_bytes=%u dynamic_lds_bytes=%u allocated_lds_bytes=%u stride=%u timeout_ms=2000\n",
                PS5HIP_CANARY_TARGET, lds_kernel, lds_static_bytes, lds_dynamic_bytes, expected.allocated_bytes, lds_stride);
    std::fflush(nullptr);
    const auto result = queue.submit_and_wait(commands, resources, 2000);
    if (result.status != ps5hip::SubmissionStatus::Complete) {
        std::fprintf(stderr, "PS5HIP submission status=%u driver=0x%08x\n",
                     unsigned(result.status), unsigned(result.driver_error));
        quarantine();
    }
    validate(output, input);
    for (auto &memory : resources)
        if (!memory->close()) throw ps5hip::ObjectError("LDS allocation teardown failed");
    std::printf("PS5HIP PASS kernel=%s exact_values=257 inputs=unchanged guards=unchanged allocations=closed\n", lds_kernel);
}
