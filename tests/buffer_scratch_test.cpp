#include "ps5hip/launch.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>

using namespace ps5hip;
void check(bool ok) { if (!ok) throw std::runtime_error("buffer scratch assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ObjectError &) { return; }
    throw std::runtime_error("invalid buffer scratch accepted");
}
int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 2 || (argc == 3 && std::string(argv[2]) == "--o1-profile"));
        const bool o1_profile = argc == 3;
        const uint32_t private_bytes = o1_profile ? 712 : 212;
        const uint32_t vgpr_spills = o1_profile ? 177 : 52;
        const uint32_t wave_bytes = o1_profile ? 23552 : 7168;
        const uint32_t ring_bytes = o1_profile ? 30146560 : 9175040;
        const uint32_t tmpring_size = o1_profile ? 0x17500 : 0x7500;
        CodeObject object{};
        if (argc >= 2) {
            std::ifstream file(argv[1], std::ios::binary); check(bool(file));
            std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
            object = inspect_code_object(bytes.data(), bytes.size());
        } else {
            check(argc == 1);
            object.image_size = 4096;
            Kernel k{};
            k.name = "register_spill"; k.entry_address = 256;
            k.kernarg_size = 16; k.kernarg_alignment = 8; k.private_bytes = 212;
            k.rsrc1 = 0x60af0041; k.rsrc2 = 0x8d; k.properties = 0x409;
            k.max_workgroup_size = 64; k.wave_size = 32; k.vgpr_spills = 52;
            k.arguments = {{0, 8, "global_buffer"}, {8, 8, "global_buffer"}};
            object.kernels.push_back(k);
        }
        const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                       [](const Kernel &k) { return k.name == "register_spill"; });
        check(found != object.kernels.end());
        const auto &kernel = *found;
        check(kernel.private_bytes == private_bytes && kernel.vgpr_spills == vgpr_spills);
        check(!initial_launch_rejection(kernel).empty() && buffer_scratch_launch_rejection(kernel).empty());
        const GpuRange module{0x200040000, 0x100000}, kernarg{0x4000000000, 0x4000};
        const GpuRange storage{0x4000100000, ring_bytes};
        const std::vector<LaunchArgument> args{{0x4002000000, 16384}, {0x4003000000, 16384}};
        const auto plan = plan_buffer_scratch(private_bytes, storage);
        check(plan.bytes_per_wave == wave_bytes && plan.ring_bytes == ring_bytes && plan.tmpring_size == tmpring_size);
        check(plan.descriptor[0] == 0x100000 && plan.descriptor[1] == 0x80000040);
        check(plan.descriptor[2] == ring_bytes);
        // Decode independent fields, rather than comparing only the packed constant.
        const uint32_t word = plan.descriptor[3];
        check((word & 7) == 4 && ((word >> 3) & 7) == 5 && ((word >> 6) & 7) == 6 &&
              ((word >> 9) & 7) == 7 && ((word >> 12) & 127) == 20 &&
              ((word >> 21) & 3) == 2 && ((word >> 23) & 1) == 1 &&
              ((word >> 24) & 1) == 1 && ((word >> 28) & 3) == 2 && (word >> 30) == 0);
        check((word & 0x0e180000) == 0); // Reserved bits.
        auto prepare = [&](const CodeObject &obj, const BufferScratchPlan &scratch) {
            return prepare_buffer_scratch_launch(obj, "register_spill", module, kernarg, 1, args, scratch);
        };
        const auto launch = prepare(object, plan);
        check(launch.dispatch.user_sgpr_count == 6 && launch.dispatch.user_sgprs[4] == 0 &&
              launch.dispatch.user_sgprs[5] == 0x40 && launch.dispatch.kernargs.size() == 16);
        const auto words = encode_buffer_scratch_dispatch(launch);
        check(words.size() == 59);
        size_t pos = 10;
        std::map<uint32_t, uint32_t> regs;
        while (((words.at(pos) >> 8) & 255) == 0x76) {
            const uint32_t header = words.at(pos++);
            check((header >> 30) == 3 && (header & 3) == 0);
            const unsigned count = (header >> 16) & 0x3fff;
            const uint32_t address = 0xb000 + words.at(pos++) * 4;
            for (unsigned i = 0; i < count; ++i) check(regs.emplace(address + i * 4, words.at(pos++)).second);
        }
        check(regs.at(0xb860) == tmpring_size && regs.at(0xb84c) == 0x8d);
        for (unsigned i = 0; i < 4; ++i) check(regs.at(0xb900 + i * 4) == plan.descriptor[i]);
        check(regs.at(0xb910) == 0 && regs.at(0xb914) == 0x40);
        check(pos + 5 == words.size() && words[pos] == 0xc0031502 && words[pos + 1] == 1 &&
              words[pos + 2] == 1 && words[pos + 3] == 1 && words[pos + 4] == 0x8005);
        reject([&] { prepare_initial_launch(object, "register_spill", module, kernarg, 1, args); });
        reject([&] { encode_initial_dispatch(launch.dispatch); });
        for (uint32_t n : {0u, UINT32_MAX, 104833u}) reject([&] { buffer_scratch_ring_bytes(n); });
        check(buffer_scratch_ring_bytes(1) == 1310720 && buffer_scratch_ring_bytes(32) == 1310720);
        check(buffer_scratch_ring_bytes(33) == 2621440);
        check(buffer_scratch_ring_bytes(104832) == 4293918720u);
        for (GpuRange range : {GpuRange{0, storage.bytes}, GpuRange{storage.address + 1, storage.bytes},
                               GpuRange{storage.address, storage.bytes - 1}, GpuRange{storage.address, 0},
                               GpuRange{(uint64_t(1) << 47) - 4096, storage.bytes},
                               GpuRange{UINT64_MAX, storage.bytes}})
            reject([&] { plan_buffer_scratch(private_bytes, range); });
        for (unsigned field = 0; field < 8; ++field) {
            auto bad = plan;
            if (field < 4) bad.descriptor[field] ^= 1;
            if (field == 4) ++bad.bytes_per_wave;
            if (field == 5) ++bad.ring_bytes;
            if (field == 6) ++bad.tmpring_size;
            if (field == 7) ++bad.private_bytes;
            reject([&] { prepare(object, bad); });
            auto bad_launch = launch; bad_launch.scratch = bad;
            // Private byte counts with identical rounded stride are valid plans,
            // but preparation must still match the kernel's exact requirement.
            if (field != 7) reject([&] { encode_buffer_scratch_dispatch(bad_launch); });
        }
        for (uint64_t address : {module.address, kernarg.address, args[0].value, args[1].value}) {
            const auto alias = plan_buffer_scratch(private_bytes, {address, storage.bytes});
            reject([&] { prepare(object, alias); });
        }
        for (unsigned mutation = 0; mutation < 11; ++mutation) {
            auto bad = object;
            auto &k = bad.kernels[size_t(found - object.kernels.begin())];
            if (mutation == 0) k.wave_size = 64;
            if (mutation == 1) k.properties |= 0x20; // Flat-scratch initialization.
            if (mutation == 2) k.properties |= 0x800; // Dynamic stack.
            if (mutation == 3) k.properties &= ~1;
            if (mutation == 4) k.rsrc2 &= ~1u;
            if (mutation == 5) k.rsrc2 |= 0x100; // Extra system SGPR.
            if (mutation == 6) k.group_bytes = 256;
            if (mutation == 7) k.rsrc3 = 1;
            if (mutation == 8) k.required_workgroup = {8, 8, 1};
            if (mutation == 9) k.hidden_arguments.push_back({16, 8, "hidden_hostcall_buffer"});
            if (mutation == 10) k.private_bytes = 0;
            reject([&] { prepare(bad, plan); });
        }
        auto bad_launch = launch; bad_launch.dispatch.user_sgprs[0] ^= 4096;
        reject([&] { encode_buffer_scratch_dispatch(bad_launch); });
        bad_launch = launch; bad_launch.dispatch.rsrc2 &= ~1u;
        reject([&] { encode_buffer_scratch_dispatch(bad_launch); });
        std::cout << "buffer scratch: real ABI, ring/descriptor packing, packets and rejection checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
