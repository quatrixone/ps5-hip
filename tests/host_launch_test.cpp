#include "ps5hip/host_launch.h"
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("host launch capture assertion"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid host launch capture accepted");
}
ps5hip::ReadableSpan bytes(const void *p, size_t n) { return {static_cast<const uint8_t *>(p), n}; }
void put(std::vector<uint8_t> &data, size_t at, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) data.at(at + i) = uint8_t(value >> (i * 8));
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3);
        std::ifstream input(argv[1], std::ios::binary); check(bool(input));
        std::vector<uint8_t> bundle{std::istreambuf_iterator<char>(input), {}};
        const auto owned = ps5hip::read_fat_binary(bundle.data(), bundle.size(), argv[2]);
        check(owned.object.kernels.size() == 1);
        const auto &kernel = owned.object.kernels.front();
        check(kernel.name == "_Z10vector_addPKfS0_Pfj" && kernel.kernarg_size == 288 &&
              kernel.arguments.size() == 4 && kernel.hidden_arguments.size() == 13);
        std::array<uint64_t, 3> pointers{0x4000008010, 0x400000c020, 0x4000010030};
        uint32_t n = 73;
        std::array<void *, 4> addresses{&pointers[0], &pointers[1], &pointers[2], &n};
        const auto address_span = bytes(addresses.data(), sizeof(addresses));
        std::vector<ps5hip::ReadableSpan> slots{bytes(&pointers[0], 8), bytes(&pointers[1], 8),
                                              bytes(&pointers[2], 8), bytes(&n, 4)};
        const std::vector<ps5hip::GpuRange> buffers{{0x4000008000, 512}, {0x400000c000, 512}, {0x4000010000, 512}};
        auto capture = [&] { return ps5hip::capture_initial_arguments(kernel, address_span, slots, buffers); };
        const auto values = capture();
        check(values.size() == 4 && values[0].value == pointers[0] && values[0].buffer_bytes == 496 &&
              values[1].buffer_bytes == 480 && values[2].buffer_bytes == 464 && values[3].value == 73 &&
              values[3].buffer_bytes == 0);
        const ps5hip::GpuRange module{0x200040000, 0x4000}, kernarg{0x4000000000, 0x4000};
        const auto launch = ps5hip::prepare_initial_launch(owned.object, kernel.name, module, kernarg, 2, values);
        std::vector<uint8_t> expected(288, 0);
        put(expected, 0, pointers[0], 8); put(expected, 8, pointers[1], 8); put(expected, 16, pointers[2], 8);
        put(expected, 24, 73, 4); put(expected, 32, 2, 4); put(expected, 36, 1, 4); put(expected, 40, 1, 4);
        put(expected, 44, 64, 2); put(expected, 46, 1, 2); put(expected, 48, 1, 2); put(expected, 96, 1, 2);
        check(launch.kernargs == expected && launch.threads == std::array<uint32_t, 3>{64, 1, 1});
        const auto packets = ps5hip::encode_initial_dispatch(launch);
        check(packets.size() >= 5 && std::vector<uint32_t>(packets.end() - 5, packets.end()) ==
              std::vector<uint32_t>({0xc0031502, 2, 1, 1, 0x8005}));
        const auto original_pointers = pointers;
        const auto original_addresses = addresses;
        pointers.fill(0); addresses.fill(nullptr); n = 0;
        check(values[0].value == original_pointers[0] && values[3].value == 73 &&
              launch.kernargs == expected && ps5hip::encode_initial_dispatch(launch) == packets);
        pointers = original_pointers; addresses = original_addresses; n = 73;
        std::array<uint8_t, 16> unaligned{};
        unaligned[1] = 73;
        const auto scalar_slot = slots[3];
        addresses[3] = unaligned.data() + 1; slots[3] = bytes(unaligned.data(), unaligned.size());
        check(capture()[3].value == 73);
        uint64_t scalar64 = 0xfedcba9876543210;
        addresses[3] = &scalar64; slots[3] = bytes(&scalar64, sizeof(scalar64));
        auto wide_object = owned.object; wide_object.kernels.front().arguments[3].size = 8;
        const auto wide_values = ps5hip::capture_initial_arguments(wide_object.kernels.front(), address_span, slots, buffers);
        check(wide_values[3].value == scalar64 && wide_values[3].buffer_bytes == 0);
        const auto wide_launch = ps5hip::prepare_initial_launch(wide_object, kernel.name, module, kernarg, 2, wide_values);
        auto wide_expected = expected; put(wide_expected, 24, scalar64, 8);
        check(wide_launch.kernargs == wide_expected);
        addresses = original_addresses; slots[3] = scalar_slot;
        for (uint64_t invalid : {uint64_t(0), buffers[0].address - 1, buffers[0].address + buffers[0].bytes,
                                 uint64_t(1) << 47, UINT64_MAX}) {
            pointers[0] = invalid; reject(capture);
        }
        pointers = original_pointers;
        auto retired = buffers; retired.erase(retired.begin());
        reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, slots, retired); });
        pointers[1] = pointers[0]; check(capture()[1].buffer_bytes == 496); pointers = original_pointers;
        for (auto invalid : std::vector<ps5hip::GpuRange>{{0, 512}, {1, 0}, {UINT64_MAX, 1},
                             {(uint64_t(1) << 47) - 8, 16}, {buffers[0].address + 8, 16}}) {
            auto changed = buffers; changed.push_back(invalid);
            reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, slots, changed); });
        }
        auto duplicate = buffers; duplicate.push_back(buffers[0]);
        reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, slots, duplicate); });
        for (size_t length : {size_t(0), sizeof(addresses) - 1, sizeof(addresses) + 8})
            reject([&] { ps5hip::capture_initial_arguments(kernel, bytes(addresses.data(), length), slots, buffers); });
        auto missing = slots; missing.pop_back();
        reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, missing, buffers); });
        for (const void *p : {static_cast<const void *>(nullptr), reinterpret_cast<const void *>(uintptr_t(-8))})
            reject([&] { ps5hip::capture_initial_arguments(kernel, bytes(p, sizeof(addresses)), slots, buffers); });
        for (size_t i = 0; i < addresses.size(); ++i) {
            addresses[i] = nullptr; reject(capture); addresses = original_addresses;
            auto short_slots = slots; --short_slots[i].size;
            reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, short_slots, buffers); });
            addresses[i] = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(slots[i].data) + slots[i].size);
            reject(capture); addresses = original_addresses;
        }
        auto high_slots = slots; high_slots[0] = bytes(reinterpret_cast<const void *>(uintptr_t(-8)), 16);
        reject([&] { ps5hip::capture_initial_arguments(kernel, address_span, high_slots, buffers); });
        for (unsigned mutation = 0; mutation < 5; ++mutation) {
            auto bad = kernel;
            if (mutation == 0) bad.arguments[0].size = 4;
            if (mutation == 1) bad.arguments[3].kind = "dynamic_shared_pointer";
            if (mutation == 2) bad.arguments[3].size = 2;
            if (mutation == 3) bad.arguments[1].offset = 0;
            if (mutation == 4) ++bad.hidden_arguments[0].offset;
            reject([&] { ps5hip::capture_initial_arguments(bad, {nullptr, 32}, slots, buffers); });
        }
        for (uint32_t groups : {0u, UINT32_MAX / 64 + 1, UINT32_MAX})
            reject([&] { ps5hip::prepare_initial_launch(owned.object, kernel.name, module, kernarg, groups, values); });
        auto wrong_shape = owned.object; wrong_shape.kernels.front().required_workgroup = {8, 8, 1};
        reject([&] { ps5hip::prepare_initial_launch(wrong_shape, kernel.name, module, kernarg, 2, values); });
        auto storage_alias = values; storage_alias[0] = {module.address, module.bytes};
        reject([&] { ps5hip::prepare_initial_launch(owned.object, kernel.name, module, kernarg, 2, storage_alias); });
        storage_alias[0] = {kernarg.address, kernarg.bytes};
        reject([&] { ps5hip::prepare_initial_launch(owned.object, kernel.name, module, kernarg, 2, storage_alias); });
        ps5hip::HostRegistration registry;
        int token = 0;
        reject([&] { registry.lookup(&token); });
        std::array<uint8_t, 24> wrapper{};
        auto wrap_put = [&](size_t at, uint64_t v, unsigned width) {
            for (unsigned i = 0; i < width; ++i) wrapper[at + i] = uint8_t(v >> (i * 8));
        };
        wrap_put(0, 0x48495046, 4); wrap_put(4, 1, 4); wrap_put(8, reinterpret_cast<uintptr_t>(bundle.data()), 8);
        auto handle = registry.register_module(wrapper.data(), bytes(wrapper.data(), 24), bytes(bundle.data(), bundle.size()), argv[2]);
        registry.register_function(handle, &token, kernel.name);
        check(ps5hip::capture_initial_arguments(registry.lookup(&token), address_span, slots, buffers)[3].value == 73);
        registry.unregister_module(handle); reject([&] { registry.lookup(&token); });
        std::cout << "PASS actual-vector capture, owning values, interior extents, retirement, metadata, geometry and packets; synthetic GPU addresses, no submit\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
