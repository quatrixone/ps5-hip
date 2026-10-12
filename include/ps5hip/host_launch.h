#pragma once
#include "ps5hip/host_registration.h"
#include "ps5hip/launch.h"

namespace ps5hip {
// The caller certifies readable, immutable argument-array/value storage for this
// call and supplies only permitted live data allocations (never launch storage).
// This measured profile has at least one explicit argument. Pointer values must
// resolve uniquely inside that inventory. Copies own their
// bits, not GPU allocations; this cannot detect address reuse or prove mappings.
std::vector<LaunchArgument> capture_initial_arguments(
    const Kernel &, ReadableSpan argument_addresses,
    const std::vector<ReadableSpan> &value_storage,
    const std::vector<GpuRange> &live_data_buffers);
}
