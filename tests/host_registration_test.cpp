#include "ps5hip/host_registration.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("registration assertion"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid registration accepted");
}
void put(std::array<uint8_t, 24> &data, size_t at, uint64_t value, unsigned count = 8) {
    for (unsigned i = 0; i < count; ++i) data[at + i] = uint8_t(value >> (i * 8));
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3);
        std::ifstream input(argv[1], std::ios::binary); check(bool(input));
        std::vector<uint8_t> bundle{std::istreambuf_iterator<char>(input), {}};
        std::array<uint8_t, 24> wrapper{};
        put(wrapper, 0, 0x48495046, 4); put(wrapper, 4, 1, 4);
        put(wrapper, 8, reinterpret_cast<uintptr_t>(bundle.data()));
        const ps5hip::ReadableSpan ws{wrapper.data(), wrapper.size()}, bs{bundle.data(), bundle.size()};
        auto read = [&] { return ps5hip::read_single_host_wrapper(wrapper.data(), ws, bs, argv[2]); };
        const auto original = wrapper;
        const auto expected = read();
        check(!expected.object.kernels.empty());
        for (size_t length : {size_t(0), size_t(23), size_t(25), size_t(48)})
            reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), {wrapper.data(), length}, bs, argv[2]); });
        for (uintptr_t address : {uintptr_t(0), reinterpret_cast<uintptr_t>(wrapper.data()) - 1,
                                 reinterpret_cast<uintptr_t>(wrapper.data()) + 1,
                                 reinterpret_cast<uintptr_t>(wrapper.data()) + 24})
            reject([&] { ps5hip::read_single_host_wrapper(reinterpret_cast<void *>(address), ws, bs, argv[2]); });
        for (uintptr_t address : {uintptr_t(0), reinterpret_cast<uintptr_t>(bundle.data()) - 1,
                                 reinterpret_cast<uintptr_t>(bundle.data()) + 1,
                                 reinterpret_cast<uintptr_t>(bundle.data()) + bundle.size(), UINTPTR_MAX}) {
            put(wrapper, 8, address); reject(read); wrapper = original;
        }
        for (size_t at : {size_t(0), size_t(4), size_t(16)}) {
            wrapper[at] ^= 2; reject(read); wrapper = original;
        }
        const auto high = reinterpret_cast<const uint8_t *>(UINTPTR_MAX - 7);
        reject([&] { ps5hip::read_single_host_wrapper(high, {high, 24}, bs, argv[2]); });
        reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), ws, {high, 32}, argv[2]); });
        reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), ws, {nullptr, 32}, argv[2]); });
        reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), ws, {wrapper.data(), 32}, argv[2]); });
        reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), ws, {bundle.data(), 31}, argv[2]); });
        reject([&] { ps5hip::read_single_host_wrapper(wrapper.data(), ws, {bundle.data(), 64u*1024u*1024u+1}, argv[2]); });
        ps5hip::HostRegistration registry;
        wrapper[0] ^= 1;
        reject([&] { registry.register_module(wrapper.data(), ws, bs, argv[2]); });
        check(registry.active_modules() == 0); wrapper = original;
        auto handle = registry.register_module(wrapper.data(), ws, bs, argv[2]);
        reject([&] { registry.register_module(wrapper.data(), ws, bs, argv[2]); });
        int token = 0, other = 0;
        const auto name = expected.object.kernels.front().name;
        reject([&] { registry.register_function(handle, &token, "absent"); });
        reject([&] { registry.register_function(handle, nullptr, name); });
        check(registry.active_functions() == 0);
        registry.register_function(handle, &token, name);
        check(registry.lookup(&token).name == name && registry.active_functions() == 1);
        reject([&] { registry.register_function(handle, &token, name); });
        reject([&] { registry.register_function(handle, &other, name); });
        reject([&] { registry.lookup(&other); });
        const auto second = bundle;
        std::fill(bundle.begin(), bundle.end(), 0);
        check(registry.module(handle).bytes == expected.bytes && registry.lookup(&token).name == name);
        registry.unregister_module(handle);
        check(registry.active_modules() == 0 && registry.active_functions() == 0);
        reject([&] { registry.module(handle); });
        reject([&] { registry.lookup(&token); });
        reject([&] { registry.unregister_module(handle); });
        reject([&] { registry.unregister_module(&other); });
        reject([&] { registry.register_function(handle, &token, name); });
        std::copy(second.begin(), second.end(), bundle.begin());
        auto next = registry.register_module(wrapper.data(), ws, bs, argv[2]);
        check(next != handle);
        reject([&] { registry.module(handle); });
        registry.unregister_module(next);
        std::cout << "PASS checked wrapper bounds, owning module, bindings, failures and retired handles\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
