#include "ps5hip/launch.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>

using namespace ps5hip;
#ifdef PS5HIP_TEST_PRIVATE_BUFFER
constexpr uint32_t private_bytes = 304, wave_stride = 10240, ring_bytes = 13107200,
                   tmpring = 0xa500, middle_frame = 160;
constexpr uint64_t carried_address = 0x4100001800;
#else
constexpr uint32_t private_bytes = 288, wave_stride = 9216, ring_bytes = 11796480,
                   tmpring = 0x9500, middle_frame = 144;
constexpr uint64_t carried_address = 0x4100001400;
#endif
void check(bool ok) { if (!ok) throw std::runtime_error("flat-call assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ObjectError &) { return; }
    throw std::runtime_error("invalid flat-call state accepted");
}
int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 2);
        CodeObject object{};
        if (argc == 2) {
            std::ifstream file(argv[1], std::ios::binary); check(bool(file));
            std::vector<uint8_t> data{std::istreambuf_iterator<char>(file), {}};
            object = inspect_code_object(data.data(), data.size());
            check(materialize_code_object(data.data(), data.size()).size() == object.image_size);
        } else {
            object.image_size = 4096;
            Kernel k{}; k.name = "nested_private"; k.entry_address = 256;
            k.kernarg_size = 16; k.kernarg_alignment = 8; k.private_bytes = private_bytes;
            k.rsrc1 = 0x60af0041; k.rsrc2 = 0x91; k.properties = 0x429;
            k.max_workgroup_size = 1024; k.wave_size = 32;
            k.arguments = {{0, 8, "global_buffer"}, {8, 8, "global_buffer"}};
            object.kernels.push_back(k);
        }
        auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                 [](const Kernel &k) { return k.name == "nested_private"; });
        check(found != object.kernels.end());
        const size_t index = size_t(found - object.kernels.begin());
        const auto &k = *found;
        check(k.private_bytes == private_bytes && k.rsrc2 == 0x91 && k.properties == 0x429);
        check(flat_call_launch_rejection(k).empty() && !initial_launch_rejection(k).empty() &&
              !buffer_scratch_launch_rejection(k).empty() && !lds_launch_rejection(k).empty());
        const GpuRange module{0x200040000, 0x100000}, kernarg{0x4000000000, 0x4000};
        const std::vector<LaunchArgument> args{{0x4002000000, 65792}, {0x4003000000, 65792}};
        // Straddles the low32 boundary to exercise raw absolute-address carry.
        const auto plan = plan_buffer_scratch(private_bytes, {0x40fffff000, ring_bytes});
        auto prepare = [&](const CodeObject &obj, const BufferScratchPlan &scratch) {
            return prepare_flat_call_launch(obj, "nested_private", module, kernarg, 257, args, scratch);
        };
        const auto launch = prepare(object, plan);
        check(plan.private_bytes == private_bytes && plan.bytes_per_wave == wave_stride &&
              plan.ring_bytes == ring_bytes && plan.tmpring_size == tmpring);
        check(launch.dispatch.user_sgpr_count == 8 && launch.dispatch.entry_address == module.address + k.entry_address);
        check(launch.flat_init == (std::array<uint32_t, 2>{0xfffff000, 0x40}));
        check(launch.dispatch.user_sgprs[1] == 0x80000040 && launch.flat_init[1] == 0x40);
        check(launch.dispatch.rsrc1 == k.rsrc1 && launch.dispatch.rsrc2 == k.rsrc2 && launch.dispatch.rsrc3 == 0);
        const auto words = encode_flat_call_dispatch(launch);
        check(words.size() == 61);
        size_t pos = 10, user_header = 0;
        std::map<uint32_t, uint32_t> regs;
        while (((words.at(pos) >> 8) & 255) == 0x76) {
            const size_t header_position = pos;
            const uint32_t header = words.at(pos++);
            check((header >> 30) == 3 && (header & 3) == 0);
            const unsigned count = (header >> 16) & 0x3fff;
            const uint32_t address = 0xb000 + words.at(pos++) * 4;
            if (address == 0xb900) { check(count == 8); user_header = header_position; }
            for (unsigned i = 0; i < count; ++i) check(regs.emplace(address + i * 4, words.at(pos++)).second);
        }
        const std::array<uint32_t, 8> expected{{0xfffff000, 0x80000040, ring_bytes, 0x21c14fac, 0, 0x40, 0xfffff000, 0x40}};
        for (unsigned i = 0; i < expected.size(); ++i) check(regs.at(0xb900 + i * 4) == expected[i]);
        check(regs.count(0xb920) == 0 && regs.count(0xb924) == 0); // System s8/s9 are not user data.
        check(regs.at(0xb860) == tmpring && regs.at(0xb84c) == 0x91 && regs.at(0xb848) == k.rsrc1);
        check(pos + 5 == words.size() && words[pos] == 0xc0031502 && words[pos + 1] == 257 &&
              words[pos + 2] == 1 && words[pos + 3] == 1 && words[pos + 4] == 0x8005);
        for (unsigned arg = 0; arg < 2; ++arg) {
            uint64_t packed = 0;
            for (unsigned byte = 0; byte < 8; ++byte) packed |= uint64_t(launch.dispatch.kernargs[arg * 8 + byte]) << (byte * 8);
            check(packed == args[arg].value);
        }
        reject([&] { encode_initial_dispatch(launch.dispatch); });
        reject([&] { encode_buffer_scratch_dispatch({launch.dispatch, plan}); });
        reject([&] { encode_lds_dispatch({launch.dispatch, plan_lds(256, 0)}); });
        reject([&] { prepare_initial_launch(object, "nested_private", module, kernarg, 257, args); });
        reject([&] { prepare_buffer_scratch_launch(object, "nested_private", module, kernarg, 257, args, plan); });
        reject([&] { prepare_lds_launch(object, "nested_private", module, kernarg, 257, args, 256); });
        // Synthetic six-slot descriptor verifies only the route/packet difference,
        // never an alternative executable ABI for the actual nested-call shader.
        auto six_object = object; six_object.kernels[index].properties = 0x409; six_object.kernels[index].rsrc2 = 0x8d;
        const auto six = prepare_buffer_scratch_launch(six_object, "nested_private", module, kernarg, 257, args, plan);
        auto reduced = words;
        reduced.erase(reduced.begin() + user_header + 8, reduced.begin() + user_header + 10);
        reduced[user_header] = 0xc0067600;
        for (size_t i = 10; i < user_header;) {
            const unsigned count = (reduced.at(i) >> 16) & 0x3fff;
            if (reduced.at(i + 1) == 0x212) reduced.at(i + 3) = 0x8d;
            i += count + 2;
        }
        check(reduced == encode_buffer_scratch_dispatch(six));
        reject([&] { prepare(six_object, plan); });
        #ifdef PS5HIP_TEST_PRIVATE_BUFFER
        reject([&] { prepare(object, plan_buffer_scratch(288, plan.storage)); });
#endif
        reject([&] { prepare(object, plan_buffer_scratch(144, {plan.storage.address, plan.storage.bytes})); });
        for (GpuRange bad : {GpuRange{plan.storage.address, plan.ring_bytes - 1},
                            GpuRange{plan.storage.address + 1, plan.ring_bytes},
                            GpuRange{(uint64_t(1) << 47) - 4096, plan.ring_bytes},
                            GpuRange{UINT64_MAX, plan.ring_bytes}})
            reject([&] { plan_buffer_scratch(private_bytes, bad); });
        reject([&] { plan_buffer_scratch(UINT32_MAX, plan.storage); });
        for (uint64_t address : {module.address, kernarg.address, args[0].value, args[1].value})
            reject([&] { prepare(object, plan_buffer_scratch(private_bytes, {address, plan.ring_bytes})); });
        for (unsigned mutation = 0; mutation < 13; ++mutation) {
            auto bad = object; auto &kernel = bad.kernels[index];
            if (mutation == 0) kernel.private_bytes = 0;
            if (mutation == 1) kernel.properties |= 0x800;
            if (mutation == 2) kernel.properties |= 0x40;
            if (mutation == 3) kernel.hidden_arguments.push_back({16, 8, "hidden_queue_ptr"});
            if (mutation == 4) kernel.group_bytes = 256;
            if (mutation == 5) kernel.rsrc3 = 1;
            if (mutation == 6) kernel.preload = 1;
            if (mutation == 7) kernel.wave_size = 64;
            if (mutation == 8) kernel.required_workgroup = {32, 2, 1};
            if (mutation == 9) kernel.rsrc1 |= 1u << 20;
            if (mutation == 10) kernel.arguments[0].kind = "dynamic_shared_pointer";
            if (mutation == 11) kernel.kernarg_size += 8;
            if (mutation == 12) kernel.arguments[1].offset = 0;
            reject([&] { prepare(bad, plan); });
        }
        for (unsigned bit = 0; bit < 32; ++bit) {
            auto bad = launch; bad.dispatch.rsrc2 ^= 1u << bit;
            reject([&] { encode_flat_call_dispatch(bad); });
            auto bad_object = object; bad_object.kernels[index].rsrc2 ^= 1u << bit;
            reject([&] { prepare(bad_object, plan); });
        }
        for (unsigned field = 0; field < 9; ++field) {
            auto bad = launch;
            if (field < 4) bad.scratch.descriptor[field] ^= 1;
            if (field == 4) bad.scratch.bytes_per_wave += 1024;
            if (field == 5) --bad.scratch.ring_bytes;
            if (field == 6) bad.scratch.tmpring_size ^= 1;
            if (field == 7) bad.flat_init[0] >>= 8; // Wrong legacy GFX7/8 convention.
            if (field == 8) bad.flat_init[1] = bad.scratch.descriptor[1]; // V# flags are not address bits.
            reject([&] { encode_flat_call_dispatch(bad); });
        }
        auto bad = launch; bad.dispatch.user_sgpr_count = 6;
        reject([&] { encode_flat_call_dispatch(bad); });
        bad = launch; bad.dispatch.user_sgprs[0] ^= 4096;
        reject([&] { encode_flat_call_dispatch(bad); });
        // Source-derived frame/capacity arithmetic only, not private-aperture emulation.
        uint32_t stack = 144 * 32; const uint32_t frame = stack; stack += middle_frame * 32;
        check(frame == 0x1200 && stack == private_bytes * 32); // Leaf adds zero.
        stack -= middle_frame * 32; check(stack == frame);
        std::vector<bool> addresses(plan.bytes_per_wave, false);
        for (unsigned lane = 0; lane < 32; ++lane) for (unsigned byte = 0; byte < private_bytes; ++byte) {
            const unsigned swizzled = (byte / 4) * 128 + lane * 4 + byte % 4;
            check(swizzled < plan.bytes_per_wave && !addresses[swizzled]); addresses[swizzled] = true;
        }
        check(std::count(addresses.begin(), addresses.end(), true) == private_bytes * 32);
        check(std::none_of(addresses.begin() + private_bytes * 32, addresses.end(), [](bool value) { return value; }));
        for (uint32_t slot : {0u, 1u, 1279u}) {
            const uint64_t wave_offset = uint64_t(slot) * plan.bytes_per_wave;
            check(wave_offset + plan.bytes_per_wave <= plan.ring_bytes);
            const uint64_t low_sum = uint64_t(launch.flat_init[0]) + wave_offset;
            const uint64_t reconstructed = (uint64_t(launch.flat_init[1] + (low_sum >> 32)) << 32) | uint32_t(low_sum);
            check(reconstructed == plan.storage.address + wave_offset);
            if (slot == 1) check(reconstructed == carried_address);
        }
        check(uint64_t(1280) * plan.bytes_per_wave == plan.ring_bytes);
        std::cout << "flat-call: structural packing, raw base, bounded frame model and negative checks passed; execution unqualified\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
