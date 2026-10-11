#include "ps5hip/launch.h"
#include <fstream>
#include <iostream>
#include <iterator>

void check(bool ok) { if (!ok) throw std::runtime_error("launch assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid launch accepted");
}
int main(int argc, char **argv) {
    try {
        check(argc == 2);
        std::ifstream input(argv[1], std::ios::binary); check(bool(input));
        std::vector<uint8_t> data{std::istreambuf_iterator<char>(input), {}};
        const auto object = ps5hip::inspect_code_object(data.data(), data.size());
        using ps5hip::GpuRange;
        using ps5hip::LaunchArgument;
        const GpuRange module{0x200040000, 0x4000}, kernarg{0x4000000000, 0x4000};
        if (!object.kernels.empty() && object.kernels[0].name == "wave_full") {
            check(object.kernels.size() == 2 && object.kernels[1].name == "wave_partial");
            const std::vector<LaunchArgument> wave_args{{0x4000008000, 64 * 4 * 4}};
            for (const auto &kernel : object.kernels) {
                check(kernel.arguments.size() == 1 && kernel.arguments[0].kind == "global_buffer" &&
                      kernel.arguments[0].size == 8 && kernel.arguments[0].offset == 0 && kernel.kernarg_size == 8);
                const auto launch = ps5hip::prepare_initial_launch(object, kernel.name, module, kernarg, 1, wave_args);
                check(launch.groups_x == 1 && launch.threads == std::array<uint32_t, 3>{64, 1, 1});
                check(launch.entry_address == module.address + kernel.entry_address);
                check(launch.kernargs == std::vector<uint8_t>({0, 0x80, 0, 0, 0x40, 0, 0, 0}));
                const auto words = ps5hip::encode_initial_dispatch(launch);
                check(words.size() >= 5 && std::vector<uint32_t>(words.end() - 5, words.end()) ==
                      std::vector<uint32_t>({0xc0031502, 1, 1, 1, 0x8005}));
                reject([&] { ps5hip::prepare_initial_launch(object, kernel.name, module, kernarg, 1, {}); });
                reject([&] { ps5hip::prepare_initial_launch(object, kernel.name, module, kernarg, 1,
                                                           {wave_args[0], {64, 0}}); });
            }
            std::cout << object.target << ": both wave entry, single-pointer packing and one-group dispatch checks passed\n";
            return 0;
        }
        const std::vector<LaunchArgument> args{{0x4000008000, 256}, {64, 0}};
        auto prepare = [&](GpuRange code, GpuRange params, uint32_t groups,
                           const std::vector<LaunchArgument> &values) {
            return ps5hip::prepare_initial_launch(object, "indexed_store", code, params, groups, values);
        };
        const auto launch = prepare(module, kernarg, 1, args);
        check(launch.entry_address == module.address + object.kernels[0].entry_address);
        check(launch.user_sgpr_count == 6 && launch.user_sgprs[0] == 0 && launch.user_sgprs[3] == 0);
        check(launch.user_sgprs[4] == 0 && launch.user_sgprs[5] == 0x40);
        check(launch.kernargs == std::vector<uint8_t>({0, 0x80, 0, 0, 0x40, 0, 0, 0, 64, 0, 0, 0}));
        check(launch.threads == std::array<uint32_t, 3>{64, 1, 1});
        const auto words = ps5hip::encode_initial_dispatch(launch);
        check(words.size() == 56);
        check(std::vector<uint32_t>(words.begin(), words.begin() + 10) ==
              std::vector<uint32_t>({0xc0065800, 0, 0xffffffff, 0xff, 0, 0, 10, 0x4381, 0xc0004200, 0}));
        // Decode packet framing independently, then check semantic register values.
        std::vector<std::pair<uint32_t, uint32_t>> registers;
        size_t pos = 10;
        while (((words.at(pos) >> 8) & 255) == 0x76) {
            const uint32_t header = words.at(pos++);
            check((header >> 30) == 3 && (header & 3) == 0);
            const unsigned count = (header >> 16) & 0x3fff;
            const uint32_t address = 0xb000 + words.at(pos++) * 4;
            for (unsigned i = 0; i < count; ++i)
                registers.emplace_back(address + i * 4, words.at(pos++));
        }
        auto reg = [&](uint32_t address) {
            for (auto item : registers) if (item.first == address) return item.second;
            throw std::runtime_error("missing launch register");
        };
        check((uint64_t(reg(0xb834)) << 40 | uint64_t(reg(0xb830)) << 8) == launch.entry_address);
        check(reg(0xb848) == object.kernels[0].rsrc1 && reg(0xb84c) == object.kernels[0].rsrc2);
        check(reg(0xb858) == 0xffffffff && reg(0xb85c) == 0xffffffff);
        check(reg(0xb864) == 0 && reg(0xb868) == 0 && reg(0xb890) == 0 && reg(0xb89c) == 0);
        check(reg(0xb81c) == 64 && reg(0xb820) == 1 && reg(0xb824) == 1);
        check(reg(0xb900) == 0 && reg(0xb90c) == 0 && reg(0xb910) == 0 && reg(0xb914) == 0x40);
        check(pos + 5 == words.size() && words[pos] == 0xc0031502 && words[pos + 1] == 1);
        check(words[pos + 4] == 0x8005);
        auto invalid_launch = launch; invalid_launch.user_sgpr_count = 32;
        reject([&] { ps5hip::encode_initial_dispatch(invalid_launch); });
        invalid_launch = launch; invalid_launch.rsrc2 |= 1;
        reject([&] { ps5hip::encode_initial_dispatch(invalid_launch); });
        reject([&] { prepare({0x300000000, module.bytes}, kernarg, 1, args); });
        reject([&] { prepare({0x200040001, module.bytes}, kernarg, 1, args); });
        reject([&] { prepare({module.address, 256}, kernarg, 1, args); });
        reject([&] { prepare(module, {kernarg.address + 1, kernarg.bytes}, 1, args); });
        reject([&] { prepare(module, {kernarg.address, 8}, 1, args); });
        reject([&] { prepare(module, module, 1, args); });
        reject([&] { prepare(module, kernarg, 0, args); });
        reject([&] { prepare(module, kernarg, UINT32_MAX, args); });
        reject([&] { prepare(module, kernarg, 1, {}); });
        auto bad = args;
        for (auto pointer : {uint64_t(0), UINT64_MAX, module.address, kernarg.address, (uint64_t(1) << 47) - 8}) {
            bad[0].value = pointer;
            reject([&] { prepare(module, kernarg, 1, bad); });
        }
        bad = args; bad[0].buffer_bytes = 0;
        reject([&] { prepare(module, kernarg, 1, bad); });
        bad = args; bad[1].value = uint64_t(1) << 32;
        reject([&] { prepare(module, kernarg, 1, bad); });
        const auto vector = ps5hip::prepare_initial_launch(object, "vector_add", module, kernarg, 2,
                            {{0x4000008000, 512}, {0x4000008000, 512}, {0x400000c000, 512}, {128, 0}});
        check(vector.kernargs.size() == 28 && vector.kernargs[24] == 128 && vector.groups_x == 2);
        auto shaped = object; shaped.kernels[0].required_workgroup = {8, 8, 1};
        reject([&] { ps5hip::prepare_initial_launch(shaped, "indexed_store", module, kernarg, 1, args); });
        shaped.kernels[0].required_workgroup = {64, 1, 1};
        check(ps5hip::prepare_initial_launch(shaped, "indexed_store", module, kernarg, 1, args).threads[0] == 64);
        std::cout << object.target << ": launch addresses, packing and rejection checks passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
