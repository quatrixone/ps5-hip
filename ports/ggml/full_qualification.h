#pragma once
#include <array>
#include <cstdint>
namespace m0 {
struct FullStatistics {
    uint64_t requests=0, completed_nodes=0, launches=0;
    unsigned instances_created=0, instances_freed=0;
};
// Enable before llama creates its backends; disable after their destruction.
void enable_full(const std::array<const char*,6>& modules);
void disable_full();
// Caller must first synchronize and clear llama's actual KV memory.
void reset_full();
FullStatistics full_statistics();
}
