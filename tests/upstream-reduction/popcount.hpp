// Local host-only companion: upstream example references an absent header.
// Not claimed to be upstream source. Width-specific overloads preserve masks.
#pragma once
#include <cstdint>
inline int popcount(std::uint32_t v) { return __builtin_popcount(v); }
inline int popcount(std::uint64_t v) { return __builtin_popcountll(v); }
