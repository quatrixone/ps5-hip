#include "ps5hip/launch.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("math launch assertion"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid math launch accepted");
}
void put(std::vector<uint8_t> &bytes, size_t offset, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) bytes.at(offset + i) = uint8_t(value >> (i * 8));
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3);
        std::ifstream input(argv[1], std::ios::binary);
        std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), {}};
        const auto object = ps5hip::inspect_code_object(bytes.data(), bytes.size());
        check(object.target == argv[2] && object.kernels.size() == 1);
        const auto &k = object.kernels[0];
        check(k.name == "fp32_math" && k.kernarg_size == 280 && k.kernarg_alignment == 8);
        check(k.arguments.size() == 3 && k.hidden_arguments.size() == 13);
        check(k.rsrc2 == 0x8c && k.properties == 0x409 && k.rsrc3 == 0);
        check(k.rsrc1 == 0x60af0041 && k.entry_address == 6144 && k.preload == 0);
        check(k.private_bytes == 0 && k.group_bytes == 0 && k.wave_size == 32);
        const auto image = ps5hip::materialize_code_object(bytes.data(), bytes.size());
        const ps5hip::GpuRange module{0x200040000, image.size()}, kernarg{0x4000000000, 280};
        const std::vector<ps5hip::LaunchArgument> args{{0x4000008000, 1542 * 4}, {0x400000c000, 257 * 4}, {257}};
        const auto launch = ps5hip::prepare_initial_launch(object, k.name, module, kernarg, 5, args);
        std::vector<uint8_t> expected(280, 0);
        put(expected, 0, args[0].value, 8); put(expected, 8, args[1].value, 8); put(expected, 16, 257, 4);
        put(expected, 24, 5, 4); put(expected, 28, 1, 4); put(expected, 32, 1, 4);
        put(expected, 36, 64, 2); put(expected, 38, 1, 2); put(expected, 40, 1, 2); put(expected, 88, 1, 2);
        check(launch.kernargs == expected && launch.user_sgpr_count == 6);
        check(launch.user_sgprs == std::array<uint32_t, 6>{0, 0, 0, 0, 0, 0x40});
        check(launch.entry_address == module.address + k.entry_address && launch.groups_x == 5);
        check(launch.threads == std::array<uint32_t, 3>{64, 1, 1});
        const auto packet = ps5hip::encode_initial_dispatch(launch);
        check(packet.size() == 56 && packet[packet.size()-4] == 5 && packet.back() == 0x8005);
        std::map<uint32_t, uint32_t> registers;
        for (size_t i = 10; i < packet.size() - 5;) {
            check((packet.at(i) & 0xc000ffffu) == 0xc0007600u);
            const unsigned n = (packet.at(i) >> 16) & 0x3fffu;
            check(n && i + n + 2 <= packet.size() - 5);
            const uint32_t address = 0xb000 + packet.at(i + 1) * 4;
            for (unsigned j = 0; j < n; ++j)
                check(registers.emplace(address + j * 4, packet.at(i + 2 + j)).second);
            i += n + 2;
        }
        check(registers.at(0xb830) == uint32_t(launch.entry_address >> 8));
        check(registers.at(0xb834) == uint32_t(launch.entry_address >> 40));
        check(registers.at(0xb848) == 0x60af0041 && registers.at(0xb84c) == 0x8c && registers.at(0xb8a0) == 0);
        check(registers.at(0xb81c) == 64 && registers.at(0xb820) == 1 && registers.at(0xb824) == 1);
        for (unsigned i = 0; i < 6; ++i) check(registers.at(0xb900 + i * 4) == (i == 5 ? 0x40u : 0u));
        auto bad = args; bad.pop_back();
        reject([&] { ps5hip::prepare_initial_launch(object, k.name, module, kernarg, 5, bad); });
        bad = args; bad[2].buffer_bytes = 4;
        reject([&] { ps5hip::prepare_initial_launch(object, k.name, module, kernarg, 5, bad); });
        bad = args; bad[2].value = uint64_t(UINT32_MAX) + 1;
        reject([&] { ps5hip::prepare_initial_launch(object, k.name, module, kernarg, 5, bad); });
        reject([&] { ps5hip::prepare_initial_launch(object, k.name, module, {kernarg.address, 279}, 5, args); });
        auto changed = object; changed.kernels[0].hidden_arguments[0].offset++;
        reject([&] { ps5hip::prepare_initial_launch(changed, k.name, module, kernarg, 5, args); });
        std::cout << "math launch actual object: 280 bytes, 13 hidden fields, six SGPRs, five groups; no GPU\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
