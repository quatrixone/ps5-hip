#include "ps5hip/fat_binary.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <utility>

namespace {
using Data = std::vector<uint8_t>;
using Item = std::pair<std::string, Data>;
void check(bool ok) { if (!ok) throw std::runtime_error("fat-binary assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid fat binary accepted");
}
Data read(const char *path) {
    std::ifstream stream(path, std::ios::binary); check(bool(stream));
    return {std::istreambuf_iterator<char>(stream), {}};
}
void put(Data &data, size_t at, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) data.at(at + i) = uint8_t(value >> (i * 8));
}
Data bundle(const std::vector<Item> &items) {
    const std::string magic = "__CLANG_OFFLOAD_BUNDLE__";
    size_t directory = 32;
    for (const auto &item : items) directory += 24 + item.first.size();
    const size_t start = (directory + 4095) & ~size_t(4095);
    Data data(start, 0);
    std::copy(magic.begin(), magic.end(), data.begin()); put(data, 24, items.size());
    size_t cursor = 32;
    for (const auto &item : items) {
        put(data, cursor, data.size()); put(data, cursor + 8, item.second.size());
        put(data, cursor + 16, item.first.size());
        std::copy(item.first.begin(), item.first.end(), data.begin() + cursor + 24);
        cursor += 24 + item.first.size();
        data.insert(data.end(), item.second.begin(), item.second.end());
    }
    return data;
}
void mutations(const Data &elf, const std::string &target) {
    const std::string host = "host-x86_64-unknown-linux--";
    const std::string device = "hipv4-amdgcn-amd-amdhsa--" + target;
    const size_t first = 32, second = first + 24 + host.size();
    Data original = bundle({{host, {}}, {device, elf}});
    auto parse = [&](const Data &data) {
        return ps5hip::read_fat_binary(data.data(), data.size(), target);
    };
    auto selected = parse(original);
    check(selected.bytes == elf && selected.object.target == target && selected.bundle_target == device);
    check(ps5hip::materialize_code_object(selected.bytes.data(), selected.bytes.size()) ==
          ps5hip::materialize_code_object(elf.data(), elf.size()));
    auto changed_source = original;
    auto owned = parse(changed_source);
    std::fill(changed_source.begin(), changed_source.end(), 0);
    check(owned.bytes == elf && owned.object.target == target);
    check(parse(bundle({{device, elf}, {host, {}}})).bytes == elf);
    check(parse(bundle({{host, {1, 2, 3}}, {device, elf}})).bytes == elf);
    check(parse(bundle({{device, elf}})).bytes == elf);
    const std::string other = "hipv4-amdgcn-amd-amdhsa--" + std::string(target == "gfx1013" ? "gfx1030" : "gfx1013");
    check(parse(bundle({{other, {1, 2, 3}}, {device, elf}})).bytes == elf);
    reject([&] { parse(bundle({{host, {}}})); });
    reject([&] { parse(bundle({{device + ":xnack-", elf}})); });
    reject([&] { parse(bundle({{"hip" + device.substr(5), elf}})); });
    reject([&] { parse(bundle({{device, {}}})); });
    reject([&] { parse(bundle({{device, elf}, {device, elf}})); });
    reject([&] { parse(bundle({{device, elf}, {host, {}}, {host, {}}})); });
    reject([&] { parse(bundle({{other, elf}})); });
    reject([&] { ps5hip::read_fat_binary(original.data(), original.size(), "gfx1010"); });
    reject([&] { ps5hip::read_fat_binary(nullptr, original.size(), target); });
    reject([&] { ps5hip::read_fat_binary(original.data(), 64u * 1024u * 1024u + 1, target); });
    for (size_t size = 0; size < original.size(); ++size)
        reject([&] { ps5hip::read_fat_binary(original.data(), size, target); });
    auto mutate = [&](size_t at, uint64_t value) {
        auto data = original; put(data, at, value); reject([&] { parse(data); });
    };
    for (uint64_t count : {uint64_t(0), uint64_t(65), UINT64_MAX}) mutate(24, count);
    for (size_t entry : {first, second}) {
        mutate(entry, UINT64_MAX); mutate(entry, original.size() + 1);
        mutate(entry, 32); mutate(entry + 8, UINT64_MAX);
        for (uint64_t length : {uint64_t(0), uint64_t(257), UINT64_MAX}) mutate(entry + 16, length);
    }
    mutate(first + 8, 1); // The zero-byte host placeholder may share an offset, not live bytes.
    auto selected_first = bundle({{device, elf}, {host, {}}});
    put(selected_first, 32 + 24 + device.size(), UINT64_MAX);
    reject([&] { parse(selected_first); }); // Selection cannot bypass later directory checks.
    auto physical = bundle({{host, {1, 2, 3}}, {device, elf}});
    auto reversed_directory = physical;
    std::copy_n(physical.begin() + second, 24 + device.size(), reversed_directory.begin() + first);
    std::copy_n(physical.begin() + first, 24 + host.size(),
                reversed_directory.begin() + first + 24 + device.size());
    check(parse(reversed_directory).bytes == elf);
    check(parse(bundle({{std::string(256, 'h'), {}}, {device, elf}})).bytes == elf);
    reject([&] { parse(bundle({{std::string(257, 'h'), {}}, {device, elf}})); });
    for (uint8_t c : {uint8_t(0), uint8_t(' '), uint8_t('\n'), uint8_t(128)}) {
        auto data = original; data[second + 24] = c; reject([&] { parse(data); });
    }
    auto compressed = original;
    std::copy_n("CCOB", 4, compressed.begin()); reject([&] { parse(compressed); });
    auto mismatch = bundle({{other, elf}});
    reject([&] { ps5hip::read_fat_binary(mismatch.data(), mismatch.size(), target == "gfx1013" ? "gfx1030" : "gfx1013"); });
    auto malformed = elf; malformed[0] = 0;
    reject([&] { parse(bundle({{device, malformed}})); });
    std::vector<Item> maximum{{device, elf}};
    for (unsigned i = 1; i < 64; ++i) maximum.push_back({"host-placeholder-" + std::to_string(i), {}});
    check(parse(bundle(maximum)).bytes == elf);
    maximum.push_back({"host-placeholder-64", {}});
    reject([&] { parse(bundle(maximum)); });
    std::mt19937 random(0x48495046);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        auto data = original;
        for (unsigned i = 0; i <= trial % 8; ++i) data[random() % data.size()] ^= uint8_t(random());
        try { check(parse(data).object.target == target); }
        catch (const ps5hip::ObjectError &) {}
    }
    std::cout << target << ": ownership, exact target, full truncation, directory/range rejection and1000 mutation checks passed\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 3) mutations(read(argv[1]), argv[2]);
        else {
            check(argc == 5 && std::string(argv[1]) == "--bundle");
            const auto data = read(argv[2]), expected = read(argv[4]);
            const auto selected = ps5hip::read_fat_binary(data.data(), data.size(), argv[3]);
            check(selected.bytes == expected);
            check(ps5hip::materialize_code_object(selected.bytes.data(), selected.bytes.size()) ==
                  ps5hip::materialize_code_object(expected.data(), expected.size()));
            std::cout << argv[3] << ": real HIP offload bundle selected, inspected and materialized unchanged\n";
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
