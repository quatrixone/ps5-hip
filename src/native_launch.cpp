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
PreparedNativeLaunch prepare_native_vector(const BundledCodeObject &bundle, const Kernel &kernel,
    const std::vector<LaunchArgument> &values, NativeDataRegistry &data) {
    // Reinspect immutable owned bytes rather than trusting mutable copied metadata.
    const auto object = inspect_code_object(bundle.bytes.data(), bundle.bytes.size());
    require(object.target == "gfx1013" && object.kernels.size() == 1 && bundle.object.kernels.size() == 1 &&
        &kernel == &bundle.object.kernels.front() && same_kernel(kernel, object.kernels.front()),
        "native vector kernel is not the unchanged selected object kernel");
    const auto &k = object.kernels.front();
    require(k.name == "_Z10vector_addPKfS0_Pfj" && k.kernarg_size == 288 && k.arguments.size() == 4 &&
        k.hidden_arguments.size() == 13 && initial_launch_rejection(k).empty(), "unmeasured native vector profile");
    for (size_t i = 0; i < 4; ++i)
        require(k.arguments[i].offset == i * 8 && k.arguments[i].size == (i == 3 ? 4 : 8) &&
            k.arguments[i].kind == (i == 3 ? "by_value" : "global_buffer"), "unmeasured native vector argument layout");
    require(values.size() == 4 && values[3].value == 73 && values[3].buffer_bytes == 0,
        "unmeasured native vector element count");
    PreparedNativeLaunch result;
    result.resources.reserve(5);
    // Resolve before any temporary allocation; pointer ranges are actual current
    // logical extents, including interior offsets, never caller-invented sizes.
    for (size_t i = 0; i < 3; ++i) {
        require(values[i].value % sizeof(float) == 0, "unaligned native vector pointer");
        auto hold = data.resolve(reinterpret_cast<const void *>(values[i].value), 73 * sizeof(float));
        require(values[i].buffer_bytes == hold.remaining, "native vector captured extent is stale or invented");
        if (std::find(result.resources.begin(), result.resources.end(), hold.memory) == result.resources.end())
            result.resources.push_back(std::move(hold.memory));
    }
    // Validate the complete permitted inventory, not just referenced subranges.
    const auto live = data.ranges();
    std::vector<GpuRange> backing;
    backing.reserve(live.size());
    for (auto range : live) backing.push_back(data.resolve(reinterpret_cast<void *>(range.address), range.bytes).memory->range());
    std::shared_ptr<NativeMemory> code, kernarg;
    try {
        const auto image = materialize_code_object(bundle.bytes.data(), bundle.bytes.size());
        uint64_t alignment = 256;
        for (const auto &segment : object.segments) alignment = std::max(alignment, segment.alignment);
        code = std::make_shared<NativeMemory>(image.size(), alignment, true);
        kernarg = std::make_shared<NativeMemory>(288, k.kernarg_alignment, false);
        const auto code_range = code->range(), kernarg_range = kernarg->range();
        bool conflict = overlaps(code_range, kernarg_range);
        for (auto range : backing)
            conflict = conflict || overlaps(range, code_range) || overlaps(range, kernarg_range);
        if (conflict) {
            data.poison();
            throw NativeDataError(NativeDataStatus::NativeFailure, "native provider returned overlapping launch/data backing");
        }
        const auto launch = prepare_initial_launch(object, k.name, code_range, kernarg_range, 2, values);
        result.commands = encode_initial_dispatch(launch);
        require(launch.kernargs.size() == 288, "native vector kernarg size changed");
        std::memcpy(code->cpu(), image.data(), image.size());
        std::memcpy(kernarg->cpu(), launch.kernargs.data(), launch.kernargs.size());
        result.resources.insert(result.resources.begin(), {code, kernarg});
        return result;
    } catch (...) {
        // No backend or GPU use has begun. Explicitly check partial setup cleanup;
        // destructor-only cleanup could hide a retained mapping from the caller.
        bool cleaned = true;
        if (kernarg && !kernarg->close()) cleaned = false;
        if (code && !code->close()) cleaned = false;
        if (!cleaned) {
            data.poison();
            throw NativeMemoryCleanupError("native vector preparation cleanup retained backing");
        }
        try { throw; }
        catch (const NativeMemorySetupError &error) { if (error.retained_backing) data.poison(); throw; }
    }
}
} // namespace ps5hip
