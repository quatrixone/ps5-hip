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
void check_geometry(const GeometryShape &shape) {
    require((shape.dimensions == 2 && shape.threads == std::array<uint32_t,3>{8,8,1} && shape.groups[2] == 1) ||
            (shape.dimensions == 3 && shape.threads == std::array<uint32_t,3>{8,4,2}), "unsupported measured geometry shape");
    uint64_t groups = 1;
    for (unsigned axis = 0; axis < 3; ++axis) {
        require(shape.groups[axis] != 0 && shape.groups[axis] <= UINT32_MAX / shape.threads[axis],
                "invalid geometry axis extent");
        require(groups <= (UINT32_MAX / 64) / shape.groups[axis], "geometry grid product exceeds limit");
        groups *= shape.groups[axis];
    }
}
}

static InitialLaunch prepare_launch(const CodeObject &object, const std::string &name,
                                    GpuRange module, GpuRange kernarg_storage,
                                    uint32_t groups_x, const std::vector<LaunchArgument> &arguments,
                                    const BufferScratchPlan *scratch, bool lds = false, bool flat_call = false,
                                    const GeometryShape *geometry = nullptr) {
    const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                   [&](const Kernel &k) { return k.name == name; });
    require(found != object.kernels.end(), "kernel not found");
    const auto &k = *found;
    const auto rejection = geometry ? geometry_launch_rejection(k) : flat_call ? flat_call_launch_rejection(k) : lds ? lds_launch_rejection(k) :
        (scratch ? buffer_scratch_launch_rejection(k) : initial_launch_rejection(k));
    if (!rejection.empty()) throw ObjectError(rejection);
    const auto threads = geometry ? geometry->threads : std::array<uint32_t,3>{64,1,1};
    const auto groups = geometry ? geometry->groups : std::array<uint32_t,3>{groups_x,1,1};
    if (geometry) check_geometry(*geometry);
    require(k.max_workgroup_size >= 64, "kernel cannot support the initial 64-thread group");
    require(k.required_workgroup == std::array<uint32_t, 3>{} ||
            k.required_workgroup == threads, "required workgroup shape mismatch");
    if (!geometry) require(groups_x != 0 && groups_x <= UINT32_MAX / 64, "invalid initial dispatch dimensions");
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
    if (scratch) {
        require(k.private_bytes == scratch->private_bytes, "scratch plan does not match kernel private storage");
        require(!overlaps(scratch->storage, module) && !overlaps(scratch->storage, kernarg_storage),
                "scratch aliases launch storage");
    }
    InitialLaunch result{};
    result.entry_address = module.address + k.entry_address;
    require(result.entry_address % 256 == 0, "unaligned executable entry");
    result.rsrc1 = k.rsrc1; result.rsrc2 = k.rsrc2; result.rsrc3 = k.rsrc3;
    result.groups_x = groups_x; result.threads = threads;
    result.user_sgpr_count = (k.rsrc2 >> 1) & 31;
    const unsigned kernarg_sgpr = (k.properties & 1) ? 4 : 0;
    result.user_sgprs[kernarg_sgpr] = uint32_t(kernarg_storage.address);
    result.user_sgprs[kernarg_sgpr + 1] = uint32_t(kernarg_storage.address >> 32);
    if (scratch) std::copy(scratch->descriptor.begin(), scratch->descriptor.end(), result.user_sgprs.begin());
    result.kernargs.resize(k.kernarg_size, 0);
    if (!k.hidden_arguments.empty()) {
        const uint32_t base = k.kernarg_size - 256;
        const std::array<uint64_t, 13> values{{groups[0], groups[1], groups[2], threads[0], threads[1], threads[2], 0, 0, 0, 0, 0, 0, geometry ? geometry->dimensions : 1}};
        for (size_t i = 0; i < detail::cov5_geometry.size(); ++i) {
            const auto &field = detail::cov5_geometry[i];
            for (unsigned byte = 0; byte < field.size; ++byte)
                result.kernargs[base + field.offset + byte] = uint8_t(values[i] >> (byte * 8));
        }
    }
    for (size_t i = 0; i < arguments.size(); ++i) {
        const auto &arg = arguments[i];
        const auto &layout = k.arguments[i];
        require(layout.offset <= result.kernargs.size() && layout.size <= result.kernargs.size() - layout.offset,
                "argument outside kernarg storage");
        if (layout.kind == "global_buffer") {
            GpuRange buffer{arg.value, arg.buffer_bytes};
            check_range(buffer);
            require(!scratch || !overlaps(buffer, scratch->storage), "argument buffer aliases scratch");
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

static std::vector<uint32_t> encode_dispatch(const InitialLaunch &launch, const BufferScratchPlan *scratch, const LdsPlan *lds = nullptr,
                                             const std::array<uint32_t, 2> *flat_init = nullptr,
                                             const GeometryShape *geometry = nullptr) {
    require(launch.entry_address >= window_begin && launch.entry_address < window_end &&
            launch.entry_address % 256 == 0, "invalid dispatch entry address");
    if (geometry) {
        check_geometry(*geometry);
        require(launch.threads == geometry->threads && launch.groups_x == geometry->groups[0], "geometry plan shape mismatch");
        require(launch.kernargs.size() >= 256 && launch.kernargs.size() <= 65536,
                "invalid geometry implicit storage");
        std::array<uint8_t, 256> expected{};
        const std::array<uint64_t, 13> values{{geometry->groups[0], geometry->groups[1], geometry->groups[2],
            geometry->threads[0], geometry->threads[1], geometry->threads[2], 0, 0, 0, 0, 0, 0, geometry->dimensions}};
        for (size_t i = 0; i < detail::cov5_geometry.size(); ++i) {
            const auto &field = detail::cov5_geometry[i];
            for (unsigned byte = 0; byte < field.size; ++byte)
                expected[field.offset + byte] = uint8_t(values[i] >> (byte * 8));
        }
        require(std::equal(expected.begin(), expected.end(), launch.kernargs.end() - 256),
                "geometry implicit arguments disagree with dispatch shape");
    } else require(launch.threads == std::array<uint32_t, 3>{64, 1, 1} && launch.groups_x != 0 &&
                   launch.groups_x <= UINT32_MAX / 64, "invalid dispatch shape");
    require(flat_init ? launch.user_sgpr_count == 8 :
            (launch.user_sgpr_count == 2 || launch.user_sgpr_count == 6), "invalid dispatch user-SGPR count");
    require(((launch.rsrc2 >> 1) & 31) == launch.user_sgpr_count &&
            (geometry ? launch.rsrc2 == 0x138cu : !(launch.rsrc2 & ~((scratch ? 0xbfu : 0xbeu) | (lds ? 0x00ff8000u : 0u)))) &&
            (launch.rsrc2 & 0x80) && launch.rsrc3 == 0, "unsupported dispatch resources");
    if (scratch) {
        require(launch.rsrc2 == (flat_init ? 0x91u : 0x8du) && launch.user_sgpr_count == (flat_init ? 8u : 6u) &&
                std::equal(scratch->descriptor.begin(), scratch->descriptor.end(), launch.user_sgprs.begin()),
                "scratch dispatch entry state mismatch");
    }
    if (flat_init) {
        require(scratch && (*flat_init)[0] == uint32_t(scratch->storage.address) &&
                (*flat_init)[1] == uint32_t(scratch->storage.address >> 32),
                "flat initialization must contain the raw scratch base");
    }
    if (lds) {
        require(launch.rsrc2 == (0x8c | lds->rsrc2_lds) && launch.user_sgpr_count == 6 &&
                std::all_of(launch.user_sgprs.begin(), launch.user_sgprs.begin() + 4,
                            [](uint32_t value) { return value == 0; }), "LDS dispatch entry state mismatch");
    }
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
    if (scratch) registers(0xb860, &scratch->tmpring_size, 1);
    registers(0xb864, zeros, 2);
    registers(0xb890, zeros, 4);
    registers(0xb830, program, 2);
    registers(0xb848, resources, 2);
    registers(0xb8a0, &launch.rsrc3, 1);
    // Two waves per group, no occupancy cap, one CU group: Mesa's GFX10 formula gives zero.
    const uint32_t limits = 0;
    registers(0xb854, &limits, 1);
    registers(0xb81c, launch.threads.data(), 3);
    if (flat_init) {
        std::array<uint32_t, 8> words{};
        std::copy(launch.user_sgprs.begin(), launch.user_sgprs.end(), words.begin());
        std::copy(flat_init->begin(), flat_init->end(), words.begin() + 6);
        registers(0xb900, words.data(), unsigned(words.size()));
    } else registers(0xb900, launch.user_sgprs.data(), launch.user_sgpr_count);
    words.insert(words.end(), {0xc0031502u, launch.groups_x, geometry ? geometry->groups[1] : 1, geometry ? geometry->groups[2] : 1, 0x8005u});
    // Initiator: shader enable, force start at zero, wave32; initial in-order dispatch.
    return words;
}

InitialLaunch prepare_initial_launch(const CodeObject &object, const std::string &name,
                                    GpuRange module, GpuRange kernarg_storage,
                                    uint32_t groups_x, const std::vector<LaunchArgument> &arguments) {
    return prepare_launch(object, name, module, kernarg_storage, groups_x, arguments, nullptr);
}
std::vector<uint32_t> encode_initial_dispatch(const InitialLaunch &launch) {
    return encode_dispatch(launch, nullptr);
}

uint32_t buffer_scratch_ring_bytes(uint32_t private_bytes) {
    require(private_bytes != 0, "zero private storage does not require scratch");
    const uint64_t per_wave = (uint64_t(private_bytes) * 32 + 1023) & ~uint64_t(1023);
    // Mesa's provisional PS5 topology rounds 9 active CUs/SA to 10: 32*10*2*2.
    const uint64_t bytes = per_wave * 1280;
    require((per_wave >> 10) <= 0x1fff && bytes <= UINT32_MAX,
            "scratch ring exceeds register capacity");
    return uint32_t(bytes);
}

BufferScratchPlan plan_buffer_scratch(uint32_t private_bytes, GpuRange storage) {
    check_range(storage);
    const uint32_t bytes = buffer_scratch_ring_bytes(private_bytes);
    const uint32_t per_wave = bytes / 1280;
    require(storage.address % 4096 == 0 && bytes <= storage.bytes,
            "scratch ring exceeds backing or base alignment");
    // ROCr GFX10 V#, with CP's wave32 INDEX_STRIDE completed here (ACO uses 2).
    return {storage, private_bytes, per_wave, bytes, 1280 | ((per_wave >> 10) << 12),
            {uint32_t(storage.address), uint32_t(storage.address >> 32) | 0x80000000u,
             bytes, 0x21c14facu}};
}

static void validate_scratch_plan(const BufferScratchPlan &scratch) {
    const auto expected = plan_buffer_scratch(scratch.private_bytes, scratch.storage);
    require(scratch.bytes_per_wave == expected.bytes_per_wave && scratch.ring_bytes == expected.ring_bytes &&
            scratch.tmpring_size == expected.tmpring_size && scratch.descriptor == expected.descriptor,
            "inconsistent scratch plan");
}
BufferScratchLaunch prepare_buffer_scratch_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, const BufferScratchPlan &scratch) {
    validate_scratch_plan(scratch);
    return {prepare_launch(object, name, module, kernarg_storage, groups_x, arguments, &scratch), scratch};
}
std::vector<uint32_t> encode_buffer_scratch_dispatch(const BufferScratchLaunch &launch) {
    validate_scratch_plan(launch.scratch);
    return encode_dispatch(launch.dispatch, &launch.scratch);
}
LdsPlan plan_lds(uint32_t static_bytes, uint32_t dynamic_bytes) {
    const uint64_t total = uint64_t(static_bytes) + dynamic_bytes;
    const uint64_t allocated = (total + 1023) & ~uint64_t(1023);
    // Use pinned PS5 Mesa's GFX10.3 allocation policy, not the compiler target ID.
    require(total <= 65536 && allocated <= 65536 && allocated / 512 <= 511,
            "LDS allocation exceeds per-group capacity");
    return {static_bytes, dynamic_bytes, uint32_t(allocated), uint32_t(allocated / 512) << 15};
}

LdsLaunch prepare_lds_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, uint32_t dynamic_bytes) {
    auto dispatch = prepare_launch(object, name, module, kernarg_storage, groups_x, arguments, nullptr, true);
    const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
                                   [&](const Kernel &k) { return k.name == name; });
    const auto lds = plan_lds(found->group_bytes, dynamic_bytes);
    require(lds.allocated_bytes != 0, "zero LDS allocation does not require LDS launch");
    dispatch.rsrc2 |= lds.rsrc2_lds;
    return {std::move(dispatch), lds};
}
std::vector<uint32_t> encode_lds_dispatch(const LdsLaunch &launch) {
    const auto expected = plan_lds(launch.lds.static_bytes, launch.lds.dynamic_bytes);
    require(expected.allocated_bytes != 0 && launch.lds.allocated_bytes == expected.allocated_bytes &&
            launch.lds.rsrc2_lds == expected.rsrc2_lds, "inconsistent LDS plan");
    return encode_dispatch(launch.dispatch, nullptr, &launch.lds);
}

FlatCallLaunch prepare_flat_call_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, const BufferScratchPlan &scratch) {
    validate_scratch_plan(scratch);
    return {prepare_launch(object, name, module, kernarg_storage, groups_x, arguments, &scratch, false, true),
            scratch, {uint32_t(scratch.storage.address), uint32_t(scratch.storage.address >> 32)}};
}
std::vector<uint32_t> encode_flat_call_dispatch(const FlatCallLaunch &launch) {
    validate_scratch_plan(launch.scratch);
    return encode_dispatch(launch.dispatch, &launch.scratch, nullptr, &launch.flat_init);
}

GeometryLaunch prepare_geometry_launch(const CodeObject &object, const std::string &name,
    GpuRange module, GpuRange kernarg_storage, const GeometryShape &shape,
    const std::vector<LaunchArgument> &arguments) {
    check_geometry(shape);
    return {prepare_launch(object, name, module, kernarg_storage, shape.groups[0], arguments,
                           nullptr, false, false, &shape), shape};
}
std::vector<uint32_t> encode_geometry_dispatch(const GeometryLaunch &launch) {
    return encode_dispatch(launch.state, nullptr, nullptr, nullptr, &launch.shape);
}

} // namespace ps5hip
