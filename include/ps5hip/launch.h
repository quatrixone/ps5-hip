#pragma once

#include "ps5hip/code_object.h"
#include <array>

namespace ps5hip {

// Explicit checked routes; private-call entry is deliberately excluded.
enum class NativeLaunchRoute { Initial, Lds, BufferScratch, Geometry, GeometryLds };

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

struct BufferScratchPlan {
    GpuRange storage;
    uint32_t private_bytes, bytes_per_wave, ring_bytes, tmpring_size;
    std::array<uint32_t, 4> descriptor;
};

// Offline GFX10 wave32 plan: fixed 1280-slot conservative Mesa-derived capacity,
// 1 KiB wave stride granularity, 4 KiB base alignment. No allocation/submission.
uint32_t buffer_scratch_ring_bytes(uint32_t private_bytes);
BufferScratchPlan plan_buffer_scratch(uint32_t private_bytes, GpuRange storage);

struct BufferScratchLaunch {
    InitialLaunch dispatch;
    BufferScratchPlan scratch;
};

// Experimental six-user-SGPR buffer-scratch contract, 64x1x1 threads.
// Caller must retain storage in SubmissionQueue's resources until completion;
// timeout/failure retains it in quarantine. These functions cannot prove ownership.
BufferScratchLaunch prepare_buffer_scratch_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, const BufferScratchPlan &scratch);
std::vector<uint32_t> encode_buffer_scratch_dispatch(const BufferScratchLaunch &launch);

struct LdsPlan {
    uint32_t static_bytes, dynamic_bytes, allocated_bytes, rsrc2_lds;
};

// PS5 Mesa GFX10.3 allocation policy: 1 KiB allocation granularity, 512-byte
// register units, at most 64 KiB per group. No backing allocation is needed.
LdsPlan plan_lds(uint32_t static_bytes, uint32_t dynamic_bytes);
struct LdsLaunch {
    InitialLaunch dispatch;
    LdsPlan lds;
};

// Experimental explicit-argument 64x1x1 wave32 contract. Dynamic bytes reserve
// capacity only: shared-pointer/hidden dynamic-size argument ABIs are unsupported.
LdsLaunch prepare_lds_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, uint32_t dynamic_bytes = 0);
std::vector<uint32_t> encode_lds_dispatch(const LdsLaunch &launch);

struct FlatCallLaunch {
    InitialLaunch dispatch; // First six user words; the selected route requires count eight.
    BufferScratchPlan scratch;
    std::array<uint32_t, 2> flat_init;
};

// OFFLINE experimental absolute-flat-scratch entry, 64x1x1 wave32. Reuses the
// cumulative private-size ring plan; raw base words follow the six-word prefix.
// Structural checks do not certify shader/call-graph bounds or private aperture.
// This planner supplies no ownership, aperture setup or execution qualification;
// the separate private304 fixture binds an independently audited compiler graph.
FlatCallLaunch prepare_flat_call_launch(
    const CodeObject &object, const std::string &name, GpuRange module,
    GpuRange kernarg_storage, uint32_t groups_x,
    const std::vector<LaunchArgument> &arguments, const BufferScratchPlan &scratch);
std::vector<uint32_t> encode_flat_call_dispatch(const FlatCallLaunch &launch);

struct GeometryShape {
    std::array<uint32_t, 3> groups, threads;
    uint32_t dimensions;
};
struct GeometryLaunch { InitialLaunch state; GeometryShape shape; };

// Only measured uniform64-thread shapes8x8x1(dim2) or8x4x2(dim3).
// Complete blocks, zero global offsets/remainders; logical tails live in kernels.
GeometryLaunch prepare_geometry_launch(const CodeObject &object, const std::string &name,
    GpuRange module, GpuRange kernarg_storage, const GeometryShape &shape,
    const std::vector<LaunchArgument> &arguments);
std::vector<uint32_t> encode_geometry_dispatch(const GeometryLaunch &launch);

struct GeometryLdsLaunch { GeometryLaunch geometry; LdsPlan lds; };
// Measured XY entry, 8x8x1, static512 LDS and the complete COv5 geometry area.
// No dynamic LDS, scratch or explicit-only geometry entry in this profile.
GeometryLdsLaunch prepare_geometry_lds_launch(const CodeObject &object, const std::string &name,
    GpuRange module, GpuRange kernarg_storage, const GeometryShape &shape,
    const std::vector<LaunchArgument> &arguments);
std::vector<uint32_t> encode_geometry_lds_dispatch(const GeometryLdsLaunch &launch);


} // namespace ps5hip
