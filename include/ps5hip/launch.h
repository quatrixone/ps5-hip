#pragma once

#include "ps5hip/code_object.h"
#include <array>

namespace ps5hip {

struct GpuRange { uint64_t address, bytes; };

// Scalars carry their raw bits and zero buffer_bytes; pointers carry the bytes
// accessible from their value. The native allocator must supply those extents.
struct LaunchArgument { uint64_t value, buffer_bytes = 0; };

struct InitialLaunch {
    uint64_t entry_address;
    uint32_t rsrc1, rsrc2, rsrc3, groups_x, user_sgpr_count;
    std::array<uint32_t, 3> threads;
    std::array<uint32_t, 6> user_sgprs{};
    std::vector<uint8_t> kernargs;
};

// Pure host preparation for the 64x1x1, wave32, scratch/LDS-free launch profile.
// Object must come from inspect_code_object; ranges must be live GPU allocations.
// Arguments are explicit only; supported COv5 geometry fields are runtime-generated.
// This neither maps memory nor submits work, and cannot verify mapping ownership.
InitialLaunch prepare_initial_launch(const CodeObject &object, const std::string &name,
                                    GpuRange module, GpuRange kernarg_storage,
                                    uint32_t groups_x, const std::vector<LaunchArgument> &arguments);

// Encodes the initial cache-invalidation prefix, compute state and DISPATCH_DIRECT.
// A native submission still needs AGC initialization, CPU upload visibility,
// compute drain/completion, bounded waiting and ownership. Never submit alone.
std::vector<uint32_t> encode_initial_dispatch(const InitialLaunch &launch);

} // namespace ps5hip
