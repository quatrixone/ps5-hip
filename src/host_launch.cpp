#include "ps5hip/host_launch.h"
#include <cstring>
#include <limits>

namespace ps5hip {
namespace {
void require(bool value, const char *message) { if (!value) throw ObjectError(message); }
uintptr_t checked_span(ReadableSpan span) {
    const auto base = reinterpret_cast<uintptr_t>(span.data);
    require(base && span.size && span.size <= std::numeric_limits<uintptr_t>::max() - base,
            "invalid trusted argument storage");
    return base;
}
}
std::vector<LaunchArgument> capture_initial_arguments(
    const Kernel &kernel, ReadableSpan addresses, const std::vector<ReadableSpan> &storage,
    const std::vector<GpuRange> &buffers) {
    static_assert(sizeof(uintptr_t) == 8, "measured compiler argument ABI is 64-bit");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "measured compiler argument ABI is little endian"
#endif
    const auto rejection = initial_launch_rejection(kernel);
    if (!rejection.empty()) throw ObjectError(rejection);
    const size_t count = kernel.arguments.size();
    require(count && count <= std::numeric_limits<size_t>::max() / sizeof(uintptr_t) &&
            addresses.size == count * sizeof(uintptr_t) && storage.size() == count,
            "compiler argument count or array extent mismatch");
    const auto array_base = checked_span(addresses);
    constexpr uint64_t limit = uint64_t(1) << 47;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto range = buffers[i];
        require(range.address && range.address < limit && range.bytes && range.bytes <= limit - range.address,
                "invalid live data allocation");
        for (size_t j = 0; j < i; ++j)
            require(range.address >= buffers[j].address + buffers[j].bytes ||
                    buffers[j].address >= range.address + range.bytes, "ambiguous live data allocations");
    }
    std::vector<LaunchArgument> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto &layout = kernel.arguments[i];
        const auto base = checked_span(storage[i]);
        uintptr_t address = 0;
        std::memcpy(&address, reinterpret_cast<const void *>(array_base + i * sizeof(uintptr_t)), sizeof(address));
        require(address >= base && address - base <= storage[i].size &&
                layout.size <= storage[i].size - (address - base), "argument value outside trusted storage");
        LaunchArgument value{};
        std::memcpy(&value.value, reinterpret_cast<const void *>(address), layout.size);
        if (layout.kind == "global_buffer") {
            bool found = false;
            for (const auto range : buffers)
                if (value.value >= range.address && value.value - range.address < range.bytes) {
                    require(!found, "ambiguous global pointer");
                    found = true;
                    value.buffer_bytes = range.bytes - (value.value - range.address);
                }
            require(found, "global pointer is not in a permitted live data allocation");
        }
        result.push_back(value);
    }
    return result;
}
}
