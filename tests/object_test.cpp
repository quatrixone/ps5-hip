#include "ps5hip/code_object.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

using Data = std::vector<uint8_t>;
void check(bool ok) { if (!ok) throw std::runtime_error("test assertion failed"); }
void reject(const Data &data) {
    try { ps5hip::inspect_code_object(data.data(), data.size()); }
    catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("malformed object accepted");
}
void put(Data &data, size_t at, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) data.at(at + i) = uint8_t(value >> (8 * i));
}
int main(int argc, char **argv) {
    try {
        check(argc == 3);
        std::ifstream input(argv[1], std::ios::binary);
        check(bool(input));
        const Data original{std::istreambuf_iterator<char>(input), {}};
        const auto object = ps5hip::inspect_code_object(original.data(), original.size());
        check(object.target == argv[2] && object.kernels.size() == 2);
        check(object.kernels[0].name == "indexed_store" && object.kernels[0].kernarg_size == 12);
        check(object.kernels[1].name == "vector_add" && object.kernels[1].kernarg_size == 28);
        for (const auto &k : object.kernels) {
            check(k.wave_size == 32 && k.private_bytes == 0 && k.group_bytes == 0);
            check(ps5hip::initial_launch_rejection(k).empty());
            auto bad = k; bad.private_bytes = 4;
            check(!ps5hip::initial_launch_rejection(bad).empty());
            bad = k; bad.rsrc2 |= 0x40;
            check(!ps5hip::initial_launch_rejection(bad).empty());
            bad = k; bad.arguments[0].kind = "hidden_global_offset_x";
            check(!ps5hip::initial_launch_rejection(bad).empty());
        }
        const auto image = ps5hip::materialize_code_object(original.data(), original.size());
        Data expected(image.size(), 0);
        for (const auto &s : object.segments)
            std::copy_n(original.begin() + s.file_offset, s.file_size, expected.begin() + s.address);
        check(image == expected);
        for (size_t length = 0; length < original.size(); ++length)
            reject(Data(original.begin(), original.begin() + length));
        for (auto field : {0u, 4u, 5u, 7u, 8u, 16u, 18u, 20u, 52u, 54u, 58u}) {
            auto bad = original; bad[field] ^= 0xff; reject(bad);
        }
        auto bad = original; put(bad, 32, UINT64_MAX, 8); reject(bad);
        bad = original; put(bad, 40, UINT64_MAX, 8); reject(bad);
        bad = original; bad[49] |= 3; reject(bad);
        for (const auto &k : object.kernels) {
            size_t descriptor = 0;
            for (const auto &s : object.segments)
                if (k.descriptor_address >= s.address && k.descriptor_address - s.address < s.file_size)
                    descriptor = s.file_offset + k.descriptor_address - s.address;
            bad = original; bad[descriptor + 12] = 1; reject(bad);
            bad = original; put(bad, descriptor + 16, uint64_t(1) << 63, 8); reject(bad);
            bad = original; bad[descriptor + 57] ^= 4; reject(bad);
            bad = original; bad[descriptor + 8] ^= 1; reject(bad);
        }
        // Mutations may remain valid; only parser errors are expected failures.
        std::mt19937 random(0x505335);
        for (unsigned n = 0; n < 5000; ++n) {
            bad = original;
            for (unsigned i = 0; i <= n % 8; ++i) bad[random() % bad.size()] ^= uint8_t(random());
            try { ps5hip::inspect_code_object(bad.data(), bad.size()); }
            catch (const ps5hip::ObjectError &) {}
        }
        std::cout << argv[2] << ": fixture, materialization, truncation, rejection and 5000 mutation checks passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
