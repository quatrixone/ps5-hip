#pragma once
#include "ps5hip/native_launch.h"

namespace ps5hip {
// Serialized bounded-module profile for trusted compiler programs. Structural
// admission and live pointer extents do not prove a kernel's indexing memory-safe.
// Only the existing ABI classes/shapes are supported; no private-call admission.
NativeLaunchRequest select_hip_launch_profile(const Kernel &, std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads, size_t dynamic_lds_bytes);
}
