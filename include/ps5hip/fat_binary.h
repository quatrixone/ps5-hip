#pragma once

#include "ps5hip/code_object.h"

namespace ps5hip {

struct BundledCodeObject {
    std::string bundle_target;
    std::vector<uint8_t> bytes;
    CodeObject object;
};

// Read an explicitly bounded, uncompressed Clang offload bundle. The selected
// linked AMDHSA V5 object is copied and inspected before return. Target selection
// is exact; this does not register host stubs, allocate GPU memory or qualify ISA.
// Input must remain immutable for the duration of this call.
BundledCodeObject read_fat_binary(const uint8_t *data, size_t size,
                                 const std::string &target);

} // namespace ps5hip
