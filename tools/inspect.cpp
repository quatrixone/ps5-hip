#include "ps5hip/fat_binary.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char **argv) {
    const bool bundle = argc == 4 && std::string(argv[1]) == "--bundle";
    if (argc != 2 && !bundle) {
        std::cerr << "usage: ps5hip-inspect object.hsaco | --bundle gfx1013|gfx1030 file.bundle\n";
        return 2;
    }
    try {
        std::ifstream input(argv[bundle ? 3 : 1], std::ios::binary | std::ios::ate);
        if (!input) throw ps5hip::ObjectError("cannot open input");
        const auto bytes = input.tellg();
        if (bytes < 64 || bytes > 64 * 1024 * 1024) throw ps5hip::ObjectError("invalid object file size");
        std::vector<uint8_t> data(static_cast<size_t>(bytes));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char *>(data.data()), bytes)) throw ps5hip::ObjectError("cannot read input");
        const auto object = bundle ? ps5hip::read_fat_binary(data.data(), data.size(), argv[2]).object :
                                     ps5hip::inspect_code_object(data.data(), data.size());
        std::cout << object.target << " image_bytes=" << object.image_size << " hardware_validated=false\n";
        for (const auto &k : object.kernels) {
            auto rejection = ps5hip::initial_launch_rejection(k);
            std::cout << k.name << " entry=" << k.entry_address << " wave=" << k.wave_size
                      << " kernarg=" << k.kernarg_size << " lds=" << k.group_bytes
                      << " private=" << k.private_bytes << " vgpr_spills=" << k.vgpr_spills
                      << " sgpr_spills=" << k.sgpr_spills << " initial_profile="
                      << (rejection.empty() ? "eligible" : rejection) << '\n';
            std::cout << "  rsrc1=0x" << std::hex << k.rsrc1 << " rsrc2=0x" << k.rsrc2
                      << " rsrc3=0x" << k.rsrc3 << " properties=0x" << k.properties
                      << " preload=0x" << k.preload << std::dec
                      << " kernarg_alignment=" << k.kernarg_alignment
                      << " max_workgroup=" << k.max_workgroup_size << '\n';
            std::cout << "  explicit_arguments=" << k.arguments.size()
                      << " hidden_arguments=" << k.hidden_arguments.size() << '\n';
            for (const auto &arg : k.hidden_arguments)
                std::cout << "    " << arg.kind << " offset=" << arg.offset << " size=" << arg.size << '\n';
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
