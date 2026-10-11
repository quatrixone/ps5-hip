#include "ps5hip/launch.h"
#include <algorithm>

namespace ps5hip {
namespace {
constexpr uint64_t gpu_limit = uint64_t(1) << 47;
constexpr uint64_t window_begin = uint64_t(2) << 32;
constexpr uint64_t window_end = uint64_t(3) << 32;
void require(bool ok, const char *message) { if (!ok) throw ObjectError(message); }
void check_range(GpuRange range) {
    require(range.address != 0 && range.address < gpu_limit && range.bytes != 0 &&
            range.bytes <= gpu_limit - range.address, "invalid GPU allocation range");
}
bool overlaps(GpuRange a, GpuRange b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}

InitialLaunch prepare_initial_launch(const CodeObject &object, const std::string &name,
                                    GpuRange module, GpuRange kernarg_storage,
                                    uint32_t groups_x, const std::vector<LaunchArgument> &arguments) {
    const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                   [&](const Kernel &k) { return k.name == name; });
    require(found != object.kernels.end(), "kernel not found");
    const auto &k = *found;
    const auto rejection = initial_launch_rejection(k);
    if (!rejection.empty()) throw ObjectError(rejection);
    require(k.max_workgroup_size >= 64, "kernel cannot support the initial 64-thread group");
    require(k.required_workgroup == std::array<uint32_t, 3>{} ||
            k.required_workgroup == std::array<uint32_t, 3>{64, 1, 1}, "required workgroup shape mismatch");
    require(groups_x != 0 && groups_x <= UINT32_MAX / 64, "invalid initial dispatch dimensions");
    check_range(module); check_range(kernarg_storage);
    require(module.address >= window_begin && module.address < window_end &&
            module.bytes <= window_end - module.address, "shader allocation outside PS5 code window");
    require(object.image_size != 0 && object.image_size <= module.bytes && k.entry_address < object.image_size,
            "module allocation does not cover the code object");
    uint64_t alignment = 256;
    for (const auto &s : object.segments) alignment = std::max(alignment, s.alignment);
    require(module.address % alignment == 0, "module base violates load alignment");
    require(k.kernarg_alignment != 0 && kernarg_storage.address % k.kernarg_alignment == 0 &&
            kernarg_storage.bytes >= k.kernarg_size, "invalid kernarg allocation");
    require(!overlaps(module, kernarg_storage), "module and kernarg allocations overlap");
    require(arguments.size() == k.arguments.size(), "kernel argument count mismatch");
    InitialLaunch result{};
    result.entry_address = module.address + k.entry_address;
    require(result.entry_address % 256 == 0, "unaligned executable entry");
    result.rsrc1 = k.rsrc1; result.rsrc2 = k.rsrc2; result.rsrc3 = k.rsrc3;
    result.groups_x = groups_x; result.threads = {64, 1, 1};
    result.user_sgpr_count = (k.rsrc2 >> 1) & 31;
    const unsigned kernarg_sgpr = (k.properties & 1) ? 4 : 0;
    result.user_sgprs[kernarg_sgpr] = uint32_t(kernarg_storage.address);
    result.user_sgprs[kernarg_sgpr + 1] = uint32_t(kernarg_storage.address >> 32);
    result.kernargs.resize(k.kernarg_size, 0);
    for (size_t i = 0; i < arguments.size(); ++i) {
        const auto &arg = arguments[i];
        const auto &layout = k.arguments[i];
        require(layout.offset <= result.kernargs.size() && layout.size <= result.kernargs.size() - layout.offset,
                "argument outside kernarg storage");
        if (layout.kind == "global_buffer") {
            GpuRange buffer{arg.value, arg.buffer_bytes};
            check_range(buffer);
            require(!overlaps(buffer, module) && !overlaps(buffer, kernarg_storage),
                    "argument buffer aliases launch storage");
        } else {
            require(arg.buffer_bytes == 0 && (layout.size == 8 || arg.value <= UINT32_MAX),
                    "invalid scalar argument bits");
        }
        for (unsigned byte = 0; byte < layout.size; ++byte)
            result.kernargs[layout.offset + byte] = uint8_t(arg.value >> (byte * 8));
    }
    return result;
}

std::vector<uint32_t> encode_initial_dispatch(const InitialLaunch &launch) {
    require(launch.entry_address >= window_begin && launch.entry_address < window_end &&
            launch.entry_address % 256 == 0, "invalid dispatch entry address");
    require(launch.threads == std::array<uint32_t, 3>{64, 1, 1} && launch.groups_x != 0 &&
            launch.groups_x <= UINT32_MAX / 64, "invalid dispatch shape");
    require(launch.user_sgpr_count == 2 || launch.user_sgpr_count == 6, "invalid dispatch user-SGPR count");
    require(((launch.rsrc2 >> 1) & 31) == launch.user_sgpr_count && !(launch.rsrc2 & ~0xbeu) &&
            (launch.rsrc2 & 0x80) && launch.rsrc3 == 0, "unsupported dispatch resources");
    require(!(launch.rsrc1 & ((1u << 20) | (1u << 22) | (1u << 25))), "privileged dispatch unsupported");
    // GFX10 ACQUIRE_MEM: invalidate instruction, scalar, vector, L1 and L2;
    // CPU upload flushes precede submission and earlier submissions are retired.
    std::vector<uint32_t> words{0xc0065800u, 0, 0xffffffffu, 0xff, 0, 0, 10, 0x4381u,
                                0xc0004200u, 0}; // PFP_SYNC_ME
    auto registers = [&](uint32_t address, const uint32_t *values, unsigned count) {
        words.push_back(0xc0007600u | (count << 16)); // SET_SH_REG
        words.push_back((address - 0xb000u) / 4);
        words.insert(words.end(), values, values + count);
    };
    const uint32_t program[] = {uint32_t(launch.entry_address >> 8), uint32_t(launch.entry_address >> 40)};
    const uint32_t resources[] = {launch.rsrc1, launch.rsrc2};
    const uint32_t destinations[] = {0xffffffffu, 0xffffffffu};
    const uint32_t zeros[] = {0, 0, 0, 0};
    registers(0xb858, destinations, 2);
    registers(0xb864, zeros, 2);
    registers(0xb890, zeros, 4);
    registers(0xb830, program, 2);
    registers(0xb848, resources, 2);
    registers(0xb8a0, &launch.rsrc3, 1);
    // Two waves per group, no occupancy cap, one CU group: Mesa's GFX10 formula gives zero.
    const uint32_t limits = 0;
    registers(0xb854, &limits, 1);
    registers(0xb81c, launch.threads.data(), 3);
    registers(0xb900, launch.user_sgprs.data(), launch.user_sgpr_count);
    words.insert(words.end(), {0xc0031502u, launch.groups_x, 1, 1, 0x8005u});
    // Initiator: shader enable, force start at zero, wave32; initial in-order dispatch.
    return words;
}
} // namespace ps5hip
