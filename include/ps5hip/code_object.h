#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5hip {

struct ObjectError : std::runtime_error { using std::runtime_error::runtime_error; };

struct KernelArgument {
    uint32_t offset, size;
    std::string kind;
};

struct Kernel {
    std::string name;
    uint64_t descriptor_address, entry_address;
    uint32_t kernarg_size, kernarg_alignment, group_bytes, private_bytes;
    uint32_t rsrc1, rsrc2, rsrc3, max_workgroup_size, wave_size;
    uint32_t sgpr_spills, vgpr_spills;
    uint16_t properties, preload;
    std::array<uint32_t, 3> required_workgroup{};
    std::vector<KernelArgument> arguments;
};

struct LoadSegment {
    uint64_t file_offset, address, file_size, memory_size, alignment;
    uint32_t flags;
};

struct CodeObject {
    std::string target;
    uint64_t image_size = 0;
    std::vector<LoadSegment> segments;
    std::vector<Kernel> kernels;
};

// Structural inspection supports the initial linked AMDHSA V5 objects. It does
// not validate arbitrary instructions or certify execution on a PS5.
CodeObject inspect_code_object(const uint8_t *data, size_t size);

// Materialize validated PT_LOAD contents at their relative virtual addresses.
// The original input must remain unchanged between inspection and materializing.
std::vector<uint8_t> materialize_code_object(const uint8_t *data, size_t size);

// Returns the first unsupported requirement for the initial scratch-free launch
// profile. An empty string means structurally eligible, never hardware-qualified.
std::string initial_launch_rejection(const Kernel &kernel);

} // namespace ps5hip
