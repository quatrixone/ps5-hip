#include "ps5hip/code_object.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: ps5hip-inspect object.hsaco\n"; return 2; }
    try {
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input) throw ps5hip::ObjectError("cannot open input");
        const auto bytes = input.tellg();
        if (bytes < 64 || bytes > 64 * 1024 * 1024) throw ps5hip::ObjectError("invalid object file size");
        std::vector<uint8_t> data(static_cast<size_t>(bytes));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char *>(data.data()), bytes)) throw ps5hip::ObjectError("cannot read input");
        const auto object = ps5hip::inspect_code_object(data.data(), data.size());
        std::cout << object.target << " image_bytes=" << object.image_size << " hardware_validated=false\n";
        for (const auto &k : object.kernels) {
            auto rejection = ps5hip::initial_launch_rejection(k);
            std::cout << k.name << " entry=" << k.entry_address << " wave=" << k.wave_size
                      << " kernarg=" << k.kernarg_size << " lds=" << k.group_bytes
                      << " private=" << k.private_bytes << " vgpr_spills=" << k.vgpr_spills
                      << " sgpr_spills=" << k.sgpr_spills << " initial_profile="
                      << (rejection.empty() ? "eligible" : rejection) << '\n';
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
