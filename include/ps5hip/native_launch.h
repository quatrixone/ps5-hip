#pragma once
#include "ps5hip/fat_binary.h"
#include "ps5hip/launch.h"
#include "ps5hip/native_data.h"

namespace ps5hip {
// All preparation precedes backend initialization. Temporaries are owned code,
// kernarg and optional scratch; resources contains these and unique live data owners.
// The serialized caller must retain this result through submit/wait and check
// explicit temporary close outcomes after completion. An uncertain queue owns
// its GPU leases and must be retained; never retry, free or unregister on poison.
struct PreparedNativeLaunch {
    std::vector<uint32_t> commands;
    std::vector<std::shared_ptr<NativeMemory>> resources;
    std::vector<std::shared_ptr<NativeMemory>> temporaries;
};

struct NativeLaunchRequest {
    NativeLaunchRoute route = NativeLaunchRoute::Initial;
    GeometryShape shape{{1,1,1},{64,1,1},1};
    uint32_t dynamic_lds_bytes = 0;
};

// Checked structural preparation for gfx1013; no runtime capability advertisement.
// Kernel must be an actual reference in bundle.object and match reinspected bytes.
// Values already own their scalar bits; each pointer extent must equal the live
// registry remainder. The caller still proves kernel-specific indexing bounds.
// No private-call/flat entry or combined scratch+LDS/geometry admission.
PreparedNativeLaunch prepare_native_launch(const BundledCodeObject &, const Kernel &,
    const std::vector<LaunchArgument> &, NativeDataRegistry &, const NativeLaunchRequest &);

// Explicitly closes only launch temporaries after known completion/no submission.
// Refuses in-use temporaries, poisons data on any failure and reports typed failure.
void close_native_launch_temporaries(PreparedNativeLaunch &, NativeDataRegistry &);

// Only the measured gfx1013 vector-add ABI: 73 floats, two groups of 64 threads,
// three explicit global buffers and one u32, 288-byte COv5 kernarg. Kernel must
// be the actual selected bundle's kernel reference; arguments are owned snapshots
// from checked capture. Host/backend execution and GPU visibility are not proved.
PreparedNativeLaunch prepare_native_vector(const BundledCodeObject &, const Kernel &,
    const std::vector<LaunchArgument> &, NativeDataRegistry &);
} // namespace ps5hip
