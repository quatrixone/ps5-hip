#include "ps5hip/launch.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("hidden geometry assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid hidden geometry accepted");
}
void put(std::vector<uint8_t> &bytes, size_t offset, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) bytes.at(offset + i) = uint8_t(value >> (i * 8));
}

ps5hip::CodeObject synthetic(bool vector) {
    ps5hip::Kernel k{};
    k.name = vector ? "_Z10vector_addPKfS0_Pfj" : "_Z10dimensionsPj";
    const uint32_t base = vector ? 32 : 8;
    k.entry_address = 256; k.kernarg_size = base + 256; k.kernarg_alignment = 8;
    k.wave_size = 32; k.max_workgroup_size = 1024; k.properties = 0x409; k.rsrc2 = 0x8c;
    k.arguments = {{0, 8, "global_buffer"}};
    if (vector) k.arguments = {{0, 8, "global_buffer"}, {8, 8, "global_buffer"},
                              {16, 8, "global_buffer"}, {24, 4, "by_value"}};
    k.hidden_arguments = {
        {base, 4, "hidden_block_count_x"}, {base + 4, 4, "hidden_block_count_y"},
        {base + 8, 4, "hidden_block_count_z"}, {base + 12, 2, "hidden_group_size_x"},
        {base + 14, 2, "hidden_group_size_y"}, {base + 16, 2, "hidden_group_size_z"},
        {base + 18, 2, "hidden_remainder_x"}, {base + 20, 2, "hidden_remainder_y"},
        {base + 22, 2, "hidden_remainder_z"}, {base + 40, 8, "hidden_global_offset_x"},
        {base + 48, 8, "hidden_global_offset_y"}, {base + 56, 8, "hidden_global_offset_z"},
        {base + 64, 2, "hidden_grid_dims"}
    };
    return {"synthetic", 4096, {}, {k}};
}

void verify(const ps5hip::CodeObject &object, const std::vector<uint8_t> &bytes = {}) {
    check(object.kernels.size() == 1);
    const auto &kernel = object.kernels[0];
    const bool vector = kernel.name == "_Z10vector_addPKfS0_Pfj";
    check(vector || kernel.name == "_Z10dimensionsPj");
    const uint32_t base = vector ? 32 : 8;
    check(kernel.kernarg_size == base + 256 && kernel.kernarg_alignment == 8);
    check(kernel.arguments.size() == (vector ? 4u : 1u) && kernel.hidden_arguments.size() == 13);
    check(ps5hip::initial_launch_rejection(kernel).empty());
    const ps5hip::GpuRange module{0x200040000, 0x4000}, kernarg{0x4000000000, 0x4000};
    std::vector<ps5hip::LaunchArgument> args{{0x4000008000, 1024}};
    if (vector) args = {{0x4000008000, 1024}, {0x400000c000, 1024}, {0x4000010000, 1024}, {73, 0}};
    auto prepare = [&](const ps5hip::CodeObject &code, uint32_t groups,
                       const std::vector<ps5hip::LaunchArgument> &values) {
        return ps5hip::prepare_initial_launch(code, kernel.name, module, kernarg, groups, values);
    };
    for (uint32_t groups : {1u, 3u, UINT32_MAX / 64}) {
        const auto launch = prepare(object, groups, args);
        std::vector<uint8_t> expected(base + 256, 0);
        put(expected, 0, args[0].value, 8);
        if (vector) {
            put(expected, 8, args[1].value, 8); put(expected, 16, args[2].value, 8);
            put(expected, 24, 73, 4);
        }
        // Independent ABI oracle includes every padding/reserved byte of the implicit area.
        put(expected, base, groups, 4); put(expected, base + 4, 1, 4); put(expected, base + 8, 1, 4);
        put(expected, base + 12, 64, 2); put(expected, base + 14, 1, 2); put(expected, base + 16, 1, 2);
        put(expected, base + 64, 1, 2);
        check(launch.kernargs == expected && launch.groups_x == groups);
        check(launch.threads == std::array<uint32_t, 3>{64, 1, 1});
        check(launch.entry_address == module.address + kernel.entry_address);
        check(launch.user_sgpr_count == 6 && launch.user_sgprs[4] == 0 && launch.user_sgprs[5] == 0x40);
        const auto words = ps5hip::encode_initial_dispatch(launch);
        check(words.at(words.size() - 4) == groups);
    }
    for (uint32_t groups : {0u, UINT32_MAX / 64 + 1, UINT32_MAX})
        reject([&] { prepare(object, groups, args); });
    reject([&] { prepare(object, 1, {}); });
    auto extra = args; extra.push_back({1, 0});
    reject([&] { prepare(object, 1, extra); });
    extra = args; extra.pop_back(); reject([&] { prepare(object, 1, extra); });
    extra = args; extra[0].buffer_bytes = 0; reject([&] { prepare(object, 1, extra); });
    if (vector) {
        extra = args; extra[3].value = UINT32_MAX;
        check(prepare(object, 1, extra).kernargs.at(27) == 255);
        extra[3].value++; reject([&] { prepare(object, 1, extra); });
        extra = args; extra[3].buffer_bytes = 4; reject([&] { prepare(object, 1, extra); });
    }
    auto invalid = [&](auto mutate) {
        auto changed = object; mutate(changed.kernels[0]);
        check(!ps5hip::initial_launch_rejection(changed.kernels[0]).empty());
        reject([&] { prepare(changed, 1, args); });
    };
    for (size_t i = 0; i < kernel.hidden_arguments.size(); ++i) {
        invalid([&](auto &k) { ++k.hidden_arguments[i].offset; });
        invalid([&](auto &k) { ++k.hidden_arguments[i].size; });
        invalid([&](auto &k) { k.hidden_arguments[i].kind = "hidden_queue_ptr"; });
    }
    invalid([](auto &k) { k.hidden_arguments.pop_back(); });
    invalid([](auto &k) { k.hidden_arguments.clear(); });
    for (int delta : {-1, 1})
        invalid([&](auto &k) { k.hidden_arguments.clear(); k.kernarg_size += delta; });
    invalid([](auto &k) { k.hidden_arguments[1].kind = k.hidden_arguments[0].kind; });
    invalid([](auto &k) { k.hidden_arguments[1].offset = k.hidden_arguments[0].offset; });
    invalid([](auto &k) { k.hidden_arguments[0].offset = 0; });
    invalid([](auto &k) { k.arguments[0].offset = k.hidden_arguments[0].offset; });
    invalid([](auto &k) { k.arguments[0].offset = UINT32_MAX; });
    invalid([](auto &k) { --k.kernarg_size; });
    invalid([](auto &k) { ++k.kernarg_size; });
    invalid([](auto &k) { k.kernarg_alignment = 4; });
    invalid([](auto &k) { k.kernarg_alignment = 12; });
    auto changed = object;
    changed.kernels[0].required_workgroup = {32, 2, 1};
    reject([&] { prepare(changed, 1, args); });
    changed = object;
    std::reverse(changed.kernels[0].hidden_arguments.begin(), changed.kernels[0].hidden_arguments.end());
    check(prepare(changed, 3, args).kernargs == prepare(object, 3, args).kernargs);
    // Mutate actual metadata kind bytes without changing MessagePack or ELF framing.
    if (!bytes.empty()) for (const std::string replacement : {"hidden_group_size_q", "hidden_group_size_x"}) {
        auto malformed = bytes;
        const std::string original = "hidden_group_size_y";
        const auto at = std::search(malformed.begin(), malformed.end(), original.begin(), original.end());
        check(at != malformed.end());
        std::copy(replacement.begin(), replacement.end(), at);
        const auto parsed = ps5hip::inspect_code_object(malformed.data(), malformed.size());
        check(!ps5hip::initial_launch_rejection(parsed.kernels[0]).empty());
        reject([&] { prepare(parsed, 1, args); });
    }
    std::cout << object.target << ' ' << kernel.name << ": COv5 geometry packing and rejection checks passed\n";
}
}

int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 2);
        if (argc == 1) {
            verify(synthetic(true));
            verify(synthetic(false));
        } else {
            std::ifstream input(argv[1], std::ios::binary); check(bool(input));
            const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), {}};
            verify(ps5hip::inspect_code_object(bytes.data(), bytes.size()), bytes);
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
