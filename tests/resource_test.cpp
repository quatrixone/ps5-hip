#include "ps5hip/code_object.h"
#include <fstream>
#include <iostream>
#include <iterator>

void check(bool ok) { if (!ok) throw std::runtime_error("resource contract assertion failed"); }
int main(int argc, char **argv) {
    try {
        check(argc == 2);
        std::ifstream input(argv[1], std::ios::binary);
        check(bool(input));
        std::vector<uint8_t> data{std::istreambuf_iterator<char>(input), {}};
        auto object = ps5hip::inspect_code_object(data.data(), data.size());
        check(object.kernels.size() == 6);
        for (const auto &k : object.kernels) {
            check(k.wave_size == 32);
            if (k.name == "reduce_serial") {
                check(k.kernarg_size == 16 && k.group_bytes == 0 && k.private_bytes == 0);
                check(ps5hip::initial_launch_rejection(k).empty());
            } else if (k.name == "reduce_lds") {
                check(k.group_bytes == 256 && !ps5hip::initial_launch_rejection(k).empty());
            } else if (k.name == "private_array" || k.name == "private_explicit") {
                check(k.private_bytes >= 128 && !ps5hip::initial_launch_rejection(k).empty());
            } else if (k.name == "register_spill") {
                check(k.private_bytes > 0 && k.vgpr_spills > 0 && !ps5hip::initial_launch_rejection(k).empty());
            } else {
                check(k.name == "atomic_count" && k.kernarg_size == 8);
                // Metadata eligibility cannot establish instruction compatibility.
                check(k.group_bytes == 0 && k.private_bytes == 0);
            }
        }
        std::cout << object.target << ": resource requirements correctly classified\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
