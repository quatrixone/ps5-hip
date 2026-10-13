#include "ps5hip/native_launch.h"
#include <algorithm>
#include <cstring>

namespace ps5hip {
namespace {
void require(bool yes, const char *detail) { if (!yes) throw ObjectError(detail); }
bool same_layout(const std::vector<KernelArgument> &a, const std::vector<KernelArgument> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].offset != b[i].offset || a[i].size != b[i].size || a[i].kind != b[i].kind) return false;
    return true;
}
bool same_kernel(const Kernel &a, const Kernel &b) {
    return a.name == b.name && a.descriptor_address == b.descriptor_address && a.entry_address == b.entry_address &&
        a.kernarg_size == b.kernarg_size && a.kernarg_alignment == b.kernarg_alignment &&
        a.group_bytes == b.group_bytes && a.private_bytes == b.private_bytes &&
        a.rsrc1 == b.rsrc1 && a.rsrc2 == b.rsrc2 && a.rsrc3 == b.rsrc3 &&
        a.max_workgroup_size == b.max_workgroup_size && a.wave_size == b.wave_size &&
        a.sgpr_spills == b.sgpr_spills && a.vgpr_spills == b.vgpr_spills &&
        a.properties == b.properties && a.preload == b.preload && a.required_workgroup == b.required_workgroup &&
        same_layout(a.arguments, b.arguments) && same_layout(a.hidden_arguments, b.hidden_arguments);
}
bool overlaps(GpuRange a, GpuRange b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
void close_native_launch_temporaries(PreparedNativeLaunch &launch, NativeDataRegistry &data) {
    // Never release part of a launch whose GPU ownership is still unresolved.
    for (const auto &memory : launch.temporaries)
        if (!memory || memory->gpu_in_use()) {
            data.poison();
            throw NativeMemoryCleanupError("native launch temporary still in use");
        }
    bool cleaned = true;
    for (auto i = launch.temporaries.rbegin(); i != launch.temporaries.rend(); ++i)
        if (!(*i)->close()) cleaned = false;
    if (!cleaned) {
        data.poison();
        throw NativeMemoryCleanupError("native launch cleanup retained backing");
    }
}
PreparedNativeLaunch prepare_native_launch(const BundledCodeObject &bundle, const Kernel &kernel,
    const std::vector<LaunchArgument> &values, NativeDataRegistry &data, const NativeLaunchRequest &request) {
    // Reinspect immutable owned bytes rather than trusting mutable copied metadata.
    const auto object = inspect_code_object(bundle.bytes.data(), bundle.bytes.size());
    require(object.target == "gfx1013", "unsupported native target");
    const auto selected = std::find_if(bundle.object.kernels.begin(), bundle.object.kernels.end(),
        [&](const Kernel &candidate) { return &candidate == &kernel; });
    require(selected != bundle.object.kernels.end(), "native kernel is not a selected object reference");
    const auto found = std::find_if(object.kernels.begin(), object.kernels.end(),
        [&](const Kernel &candidate) { return candidate.name == kernel.name; });
    require(found != object.kernels.end() && same_kernel(kernel, *found), "native kernel metadata changed");
    const auto &k = *found;
    std::string rejection;
    switch (request.route) {
    case NativeLaunchRoute::Initial: rejection = initial_launch_rejection(k); break;
    case NativeLaunchRoute::Lds: rejection = lds_launch_rejection(k); break;
    case NativeLaunchRoute::BufferScratch: rejection = buffer_scratch_launch_rejection(k); break;
    case NativeLaunchRoute::Geometry: rejection = geometry_launch_rejection(k); break;
    case NativeLaunchRoute::GeometryLds: rejection = geometry_lds_launch_rejection(k); break;
    default: throw ObjectError("unknown native launch route");
    }
    if (!rejection.empty()) throw ObjectError(rejection);
    require(request.route == NativeLaunchRoute::Lds || request.dynamic_lds_bytes == 0,
        "dynamic LDS specified for another native route");
    require(values.size() == k.arguments.size(), "native argument count mismatch");
    // Encode through the established pure routes once actual backing is known.
    auto plan = [&](GpuRange code, GpuRange args, GpuRange scratch) {
        if (request.route != NativeLaunchRoute::Geometry && request.route != NativeLaunchRoute::GeometryLds)
            require(request.shape.dimensions == 1 && request.shape.threads == std::array<uint32_t,3>{64,1,1} &&
                request.shape.groups[1] == 1 && request.shape.groups[2] == 1, "unsupported native 1D shape");
        const auto groups = request.shape.groups[0];
        switch (request.route) {
        case NativeLaunchRoute::Initial: {
            auto p = prepare_initial_launch(object, k.name, code, args, groups, values);
            return std::make_pair(encode_initial_dispatch(p), std::move(p.kernargs));
        }
        case NativeLaunchRoute::Lds: {
            auto p = prepare_lds_launch(object, k.name, code, args, groups, values, request.dynamic_lds_bytes);
            return std::make_pair(encode_lds_dispatch(p), std::move(p.dispatch.kernargs));
        }
        case NativeLaunchRoute::BufferScratch: {
            auto p = prepare_buffer_scratch_launch(object, k.name, code, args, groups, values,
                plan_buffer_scratch(k.private_bytes, scratch));
            return std::make_pair(encode_buffer_scratch_dispatch(p), std::move(p.dispatch.kernargs));
        }
        case NativeLaunchRoute::Geometry: {
            auto p = prepare_geometry_launch(object, k.name, code, args, request.shape, values);
            return std::make_pair(encode_geometry_dispatch(p), std::move(p.state.kernargs));
        }
        case NativeLaunchRoute::GeometryLds: {
            auto p = prepare_geometry_lds_launch(object, k.name, code, args, request.shape, values);
            return std::make_pair(encode_geometry_lds_dispatch(p), std::move(p.geometry.state.kernargs));
        }
        }
        throw ObjectError("unknown native launch route");
    };
    PreparedNativeLaunch result;
    result.resources.reserve(values.size() + 3);
    result.temporaries.reserve(3);
    // Resolve before any temporary allocation; pointer ranges are actual current
    // logical extents, including interior offsets, never caller-invented sizes.
    for (size_t i = 0; i < values.size(); ++i) {
        if (k.arguments[i].kind != "global_buffer") {
            require(values[i].buffer_bytes == 0 && (k.arguments[i].size == 8 || values[i].value <= UINT32_MAX),
                "invalid native scalar bits");
            continue;
        }
        auto hold = data.resolve(reinterpret_cast<const void *>(values[i].value), values[i].buffer_bytes);
        require(values[i].buffer_bytes && values[i].buffer_bytes == hold.remaining, "native captured extent is stale or invented");
        if (std::find(result.resources.begin(), result.resources.end(), hold.memory) == result.resources.end())
            result.resources.push_back(std::move(hold.memory));
    }
    // Validate the complete permitted inventory, not just referenced subranges.
    const auto live = data.ranges();
    std::vector<GpuRange> backing;
    backing.reserve(live.size());
    for (auto range : live) backing.push_back(data.resolve(reinterpret_cast<void *>(range.address), range.bytes).memory->range());
    // Validate dimensions/capacity without invented GPU allocations.
    if (request.route == NativeLaunchRoute::Geometry || request.route == NativeLaunchRoute::GeometryLds) {
        const auto &s = request.shape;
        require(request.route != NativeLaunchRoute::GeometryLds || s.dimensions == 2,
                "combined native geometry/LDS requires 2D");
        require((s.dimensions == 2 && s.threads == std::array<uint32_t,3>{8,8,1} && s.groups[2] == 1) ||
                (s.dimensions == 3 && s.threads == std::array<uint32_t,3>{8,4,2}), "unsupported native geometry shape");
        uint64_t groups = 1;
        for (unsigned axis = 0; axis < 3; ++axis) {
            require(s.groups[axis] && s.groups[axis] <= UINT32_MAX / s.threads[axis] &&
                groups <= (UINT32_MAX / 64) / s.groups[axis], "native geometry extent exceeds limit");
            groups *= s.groups[axis];
        }
    } else {
        require(request.shape.dimensions == 1 && request.shape.threads == std::array<uint32_t,3>{64,1,1} &&
            request.shape.groups[1] == 1 && request.shape.groups[2] == 1 && request.shape.groups[0] &&
            request.shape.groups[0] <= UINT32_MAX / 64, "unsupported native 1D shape");
    }
    require(k.max_workgroup_size >= 64 && (k.required_workgroup == std::array<uint32_t,3>{} ||
        k.required_workgroup == request.shape.threads), "native required workgroup mismatch");
    const uint32_t scratch_bytes = request.route == NativeLaunchRoute::BufferScratch ? buffer_scratch_ring_bytes(k.private_bytes) : 0;
    if (request.route == NativeLaunchRoute::Lds)
        require(plan_lds(k.group_bytes, request.dynamic_lds_bytes).allocated_bytes != 0, "zero native LDS allocation");
    try {
        const auto image = materialize_code_object(bundle.bytes.data(), bundle.bytes.size());
        uint64_t alignment = 256;
        for (const auto &segment : object.segments) alignment = std::max(alignment, segment.alignment);
        result.temporaries.push_back(std::make_shared<NativeMemory>(image.size(), alignment, true));
        result.temporaries.push_back(std::make_shared<NativeMemory>(std::max(1u,k.kernarg_size), k.kernarg_alignment, false));
        if (scratch_bytes) result.temporaries.push_back(std::make_shared<NativeMemory>(scratch_bytes, 4096, false));
        const auto &code = result.temporaries[0], &kernarg = result.temporaries[1];
        const auto code_range = code->range(), kernarg_range = kernarg->range();
        bool conflict = false;
        for (size_t i = 0; i < result.temporaries.size(); ++i) {
            const auto range = result.temporaries[i]->range();
            for (auto other : backing) conflict = conflict || overlaps(range,other);
            for (size_t j = 0; j < i; ++j) conflict = conflict || overlaps(range,result.temporaries[j]->range());
        }
        if (conflict) {
            data.poison();
            throw NativeDataError(NativeDataStatus::NativeFailure, "native provider returned overlapping launch/data backing");
        }
        auto launch = plan(code_range, kernarg_range, scratch_bytes ? result.temporaries[2]->range() : GpuRange{});
        result.commands = std::move(launch.first);
        require(launch.second.size() == k.kernarg_size, "native kernarg size changed");
        std::memcpy(code->cpu(), image.data(), image.size());
        if (!launch.second.empty()) std::memcpy(kernarg->cpu(), launch.second.data(), launch.second.size());
        result.resources.insert(result.resources.begin(), result.temporaries.begin(), result.temporaries.end());
        return result;
    } catch (...) {
        // No backend or GPU use has begun. Explicitly check partial setup cleanup;
        // destructor-only cleanup could hide a retained mapping from the caller.
        close_native_launch_temporaries(result, data);
        try { throw; }
        catch (const NativeMemorySetupError &error) { if (error.retained_backing) data.poison(); throw; }
    }
}
PreparedNativeLaunch prepare_native_vector(const BundledCodeObject &bundle, const Kernel &kernel,
    const std::vector<LaunchArgument> &values, NativeDataRegistry &data) {
    require(inspect_code_object(bundle.bytes.data(),bundle.bytes.size()).kernels.size() == 1,
        "native vector object must contain exactly one kernel");
    require(bundle.object.kernels.size() == 1 && &kernel == &bundle.object.kernels.front() &&
        kernel.name == "_Z10vector_addPKfS0_Pfj" && kernel.kernarg_size == 288 && kernel.arguments.size() == 4 &&
        kernel.hidden_arguments.size() == 13 && initial_launch_rejection(kernel).empty(), "unmeasured native vector profile");
    for (size_t i = 0; i < 4; ++i)
        require(kernel.arguments[i].offset == i * 8 && kernel.arguments[i].size == (i == 3 ? 4 : 8) &&
            kernel.arguments[i].kind == (i == 3 ? "by_value" : "global_buffer"), "unmeasured native vector argument layout");
    require(values.size() == 4 && values[3].value == 73 && values[3].buffer_bytes == 0, "unmeasured native vector element count");
    for (size_t i = 0; i < 3; ++i) {
        require(values[i].value % sizeof(float) == 0, "unaligned native vector pointer");
        const auto hold = data.resolve(reinterpret_cast<const void *>(values[i].value), 73 * sizeof(float));
        require(values[i].buffer_bytes == hold.remaining, "native vector captured extent is stale or invented");
    }
    return prepare_native_launch(bundle, kernel, values, data, {NativeLaunchRoute::Initial, {{2,1,1},{64,1,1},1}, 0});
}
} // namespace ps5hip
