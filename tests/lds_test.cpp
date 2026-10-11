#include "ps5hip/launch.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>

using namespace ps5hip;
void check(bool ok) { if (!ok) throw std::runtime_error("LDS assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ObjectError &) { return; }
    throw std::runtime_error("invalid LDS launch accepted");
}
void verify_probes(const CodeObject &object) {
    check(object.kernels.size() == 3);
    const GpuRange module{0x200040000, 0x100000}, kernarg{0x4000000000, 0x4000};
    for (const auto &k : object.kernels) {
        const bool dynamic = k.name != "lds_static";
        const uint32_t fixed = k.name == "lds_dynamic" ? 0 : 256;
        check(k.name == "lds_static" || k.name == "lds_dynamic" || k.name == "lds_mixed");
        check(k.group_bytes == fixed && k.private_bytes == 0 && k.kernarg_size == (dynamic ? 20u : 16u) &&
              k.hidden_arguments.empty() && k.rsrc2 == 0x8c && k.properties == 0x409 &&
              k.arguments.size() == (dynamic ? 3u : 2u));
        check(lds_launch_rejection(k).empty() && !buffer_scratch_launch_rejection(k).empty());
        // The default route cannot infer extern-shared requirements from group_bytes=0;
        // it retains its existing structural eligibility, never a semantic certification.
        check(initial_launch_rejection(k).empty() == (fixed == 0));
        for (uint32_t stride : {1u, 5u, 16u}) {
            std::vector<LaunchArgument> args{{0x4002000000, 257 * 64 * 4}, {0x4003000000, 257 * 4}};
            if (dynamic) args.push_back({stride, 0});
            const uint32_t requested = dynamic ? 256 * stride : 0;
            const auto launch = prepare_lds_launch(object, k.name, module, kernarg, 257, args, requested);
            const uint32_t allocated = dynamic ? (fixed + requested <= 1024 ? 1024 :
                (fixed + requested <= 2048 ? 2048 : (fixed ? 5120 : 4096))) : 1024;
            check(launch.lds.static_bytes == fixed && launch.lds.dynamic_bytes == requested &&
                  launch.lds.allocated_bytes == allocated && launch.dispatch.rsrc1 == k.rsrc1 &&
                  launch.dispatch.rsrc2 == (0x8c | ((allocated / 512) << 15)) &&
                  launch.dispatch.entry_address == module.address + k.entry_address);
            check(launch.dispatch.kernargs.size() == (dynamic ? 20u : 16u));
            if (dynamic) check(launch.dispatch.kernargs[16] == stride && launch.dispatch.kernargs[17] == 0 &&
                               launch.dispatch.kernargs[18] == 0 && launch.dispatch.kernargs[19] == 0);
            check(encode_lds_dispatch(launch).size() == 56);
            reject([&] { encode_initial_dispatch(launch.dispatch); });
        }
    }
}
int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 2 || (argc == 3 && std::string(argv[2]) == "--probes"));
        CodeObject object{};
        if (argc >= 2) {
            std::ifstream file(argv[1], std::ios::binary); check(bool(file));
            std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
            object = inspect_code_object(bytes.data(), bytes.size());
        } else {
            object.image_size = 4096;
            Kernel k{};
            k.name = "reduce_lds"; k.entry_address = 256;
            k.kernarg_size = 16; k.kernarg_alignment = 8; k.group_bytes = 256;
            k.rsrc1 = 0x60af0040; k.rsrc2 = 0x8c; k.properties = 0x409;
            k.max_workgroup_size = 1024; k.wave_size = 32;
            k.arguments = {{0, 8, "global_buffer"}, {8, 8, "global_buffer"}};
            object.kernels.push_back(k);
        }
        if (argc == 3) {
            verify_probes(object);
            std::cout << "LDS: actual static/dynamic/mixed entries and stride allocations passed\n";
            return 0;
        }
        const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                       [](const Kernel &k) { return k.name == "reduce_lds"; });
        check(found != object.kernels.end());
        const size_t index = size_t(found - object.kernels.begin());
        const auto &k = *found;
        check(k.group_bytes == 256 && k.private_bytes == 0 && k.rsrc2 == 0x8c && k.properties == 0x409);
        check(lds_launch_rejection(k).empty() && !initial_launch_rejection(k).empty() &&
              !buffer_scratch_launch_rejection(k).empty());
        const GpuRange module{0x200040000, 0x100000}, kernarg{0x4000000000, 0x4000};
        const std::vector<LaunchArgument> args{{0x4002000000, 257 * 64 * 4}, {0x4003000000, 257 * 4}};
        auto prepare = [&](const CodeObject &obj, uint32_t dynamic = 0) {
            return prepare_lds_launch(obj, "reduce_lds", module, kernarg, 257, args, dynamic);
        };
        const auto launch = prepare(object);
        check(launch.lds.static_bytes == 256 && launch.lds.dynamic_bytes == 0 &&
              launch.lds.allocated_bytes == 1024 && launch.lds.rsrc2_lds == 0x10000);
        check(launch.dispatch.entry_address == module.address + k.entry_address &&
              launch.dispatch.rsrc1 == k.rsrc1 && launch.dispatch.rsrc2 == 0x1008c && launch.dispatch.rsrc3 == 0);
        check(launch.dispatch.user_sgpr_count == 6 && launch.dispatch.user_sgprs ==
              (std::array<uint32_t, 6>{0, 0, 0, 0, 0, 0x40}));
        check(launch.dispatch.kernargs.size() == 16);
        for (size_t arg = 0; arg < 2; ++arg) {
            uint64_t packed = 0;
            for (unsigned byte = 0; byte < 8; ++byte) packed |= uint64_t(launch.dispatch.kernargs[arg * 8 + byte]) << (byte * 8);
            check(packed == args[arg].value);
        }
        const auto words = encode_lds_dispatch(launch);
        check(words.size() == 56);
        size_t pos = 10;
        std::map<uint32_t, uint32_t> regs;
        while (((words.at(pos) >> 8) & 255) == 0x76) {
            const uint32_t header = words.at(pos++);
            check((header >> 30) == 3 && (header & 3) == 0);
            const unsigned count = (header >> 16) & 0x3fff;
            const uint32_t address = 0xb000 + words.at(pos++) * 4;
            for (unsigned i = 0; i < count; ++i) check(regs.emplace(address + i * 4, words.at(pos++)).second);
        }
        check(regs.at(0xb84c) == 0x1008c && ((regs.at(0xb84c) >> 15) & 511) * 512 == 1024);
        check(regs.at(0xb848) == k.rsrc1 && regs.at(0xb854) == 0 && regs.count(0xb860) == 0);
        check(regs.at(0xb81c) == 64 && regs.at(0xb820) == 1 && regs.at(0xb824) == 1);
        check(pos + 5 == words.size() && words[pos] == 0xc0031502 && words[pos + 1] == 257 &&
              words[pos + 2] == 1 && words[pos + 3] == 1 && words[pos + 4] == 0x8005);
        // Identical initial packet stream apart from the independently decoded LDS field.
        auto plain = launch.dispatch; plain.rsrc2 = k.rsrc2;
        auto baseline = encode_initial_dispatch(plain);
        size_t differences = 0;
        for (size_t i = 0; i < words.size(); ++i) if (words[i] != baseline.at(i)) {
            check(words[i] == 0x1008c && baseline[i] == 0x8c); ++differences;
        }
        check(differences == 1);
        reject([&] { prepare_initial_launch(object, "reduce_lds", module, kernarg, 257, args); });
        reject([&] { encode_initial_dispatch(launch.dispatch); });
        for (const auto boundary : std::vector<std::pair<uint32_t, uint32_t>>{
                 {0, 0}, {1, 1024}, {256, 1024}, {512, 1024}, {513, 1024},
                 {1024, 1024}, {1025, 2048}, {65535, 65536}, {65536, 65536}}) {
            for (const auto sizes : {std::pair<uint32_t, uint32_t>{boundary.first, 0}, {0, boundary.first}}) {
                const auto plan = plan_lds(sizes.first, sizes.second);
                check(plan.allocated_bytes == boundary.second &&
                      ((plan.rsrc2_lds >> 15) & 511) * 512 == boundary.second && !(plan.rsrc2_lds & ~0x00ff8000));
            }
        }
        check(prepare(object, 768).lds.allocated_bytes == 1024);
        check(prepare(object, 769).lds.allocated_bytes == 2048);
        check(prepare(object, 65280).dispatch.rsrc2 == 0x40008c);
        reject([&] { prepare(object, 65281); });
        for (auto sizes : {std::pair<uint32_t, uint32_t>{65537, 0}, {65536, 1}, {UINT32_MAX, 1},
                           {1, UINT32_MAX}, {UINT32_MAX, UINT32_MAX}})
            reject([&] { plan_lds(sizes.first, sizes.second); });
        auto no_static = object; no_static.kernels[index].group_bytes = 0;
        reject([&] { prepare(no_static); });
        check(prepare(no_static, 256).lds.allocated_bytes == 1024); // Reservation only, no new argument ABI.
        for (unsigned mutation = 0; mutation < 19; ++mutation) {
            auto bad = object; auto &kernel = bad.kernels[index];
            if (mutation == 0) kernel.wave_size = 64;
            if (mutation == 1) kernel.private_bytes = 4;
            if (mutation == 2) kernel.properties |= 0x20;
            if (mutation == 3) kernel.properties |= 0x800;
            if (mutation == 4) kernel.rsrc2 |= 1;
            if (mutation == 5) kernel.rsrc2 |= 0x100;
            if (mutation == 6) kernel.rsrc2 |= 0x8000;
            if (mutation == 7) kernel.rsrc3 = 1;
            if (mutation == 8) kernel.preload = 1;
            if (mutation == 9) kernel.required_workgroup = {8, 8, 1};
            if (mutation == 10) kernel.max_workgroup_size = 32;
            if (mutation == 11) kernel.hidden_arguments.push_back({136, 4, "hidden_dynamic_lds_size"});
            if (mutation == 12) kernel.arguments[0].kind = "dynamic_shared_pointer";
            if (mutation == 13) kernel.arguments[1].offset = 0;
            if (mutation == 14) kernel.group_bytes = 65537;
            if (mutation == 15) kernel.properties &= ~1;
            if (mutation == 16) kernel.kernarg_size += 8;
            if (mutation == 17) kernel.arguments[0].offset = 1;
            if (mutation == 18) kernel.rsrc1 |= 1u << 20;
            reject([&] { prepare(bad); });
        }
        for (uint32_t groups : {0u, UINT32_MAX / 64 + 1})
            reject([&] { prepare_lds_launch(object, "reduce_lds", module, kernarg, groups, args); });
        reject([&] { prepare_lds_launch(object, "reduce_lds", module, {kernarg.address + 1, kernarg.bytes}, 1, args); });
        for (unsigned field = 0; field < 4; ++field) {
            auto bad = launch;
            if (field == 0) bad.lds.static_bytes = UINT32_MAX;
            if (field == 1) bad.lds.dynamic_bytes = UINT32_MAX;
            if (field == 2) ++bad.lds.allocated_bytes;
            if (field == 3) bad.lds.rsrc2_lds ^= 0x8000;
            reject([&] { encode_lds_dispatch(bad); });
        }
        for (unsigned bit = 0; bit < 32; ++bit) {
            auto bad = launch; bad.dispatch.rsrc2 ^= 1u << bit;
            reject([&] { encode_lds_dispatch(bad); });
        }
        for (unsigned slot = 0; slot < 4; ++slot) {
            auto bad = launch; bad.dispatch.user_sgprs[slot] = 1;
            reject([&] { encode_lds_dispatch(bad); });
        }
        auto bad = launch; bad.dispatch.threads = {32, 2, 1};
        reject([&] { encode_lds_dispatch(bad); });
        bad = launch; bad.lds = plan_lds(0, 0); bad.dispatch.rsrc2 = 0x8c;
        reject([&] { encode_lds_dispatch(bad); });
        std::cout << "LDS: allocation boundaries, real entry ABI, packets and unsupported-state checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
