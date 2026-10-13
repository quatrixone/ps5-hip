#pragma once
#include "ps5hip/fat_binary.h"
#include "ps5hip/launch.h"
#include "ps5hip/native_data.h"

namespace ps5hip {
// All preparation precedes backend initialization. Resources[0:2] are owned
// code/kernarg temporaries; the remaining entries are unique live data owners.
// The serialized caller must retain this result through submit/wait and check
// explicit temporary close outcomes after completion. An uncertain queue owns
// its GPU leases and must be retained; never retry, free or unregister on poison.
struct PreparedNativeLaunch {
    std::vector<uint32_t> commands;
    std::vector<std::shared_ptr<NativeMemory>> resources;
};

// Only the measured gfx1013 vector-add ABI: 73 floats, two groups of 64 threads,
// three explicit global buffers and one u32, 288-byte COv5 kernarg. Kernel must
// be the actual selected bundle's kernel reference; arguments are owned snapshots
// from checked capture. Host/backend execution and GPU visibility are not proved.
PreparedNativeLaunch prepare_native_vector(const BundledCodeObject &, const Kernel &,
    const std::vector<LaunchArgument> &, NativeDataRegistry &);
} // namespace ps5hip
