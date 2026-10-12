#pragma once
#include "ps5hip/fat_binary.h"
#include <memory>

namespace ps5hip {
struct ReadableSpan { const uint8_t *data; size_t size; };

// Spans are trusted readable, immutable host-image extents supplied by the linker
// or caller. This validates membership; it cannot discover mapped memory bounds.
BundledCodeObject read_single_host_wrapper(const void *wrapper, ReadableSpan wrappers,
                                          ReadableSpan bundle, const std::string &target);

// Metadata-only, serialized single-module control. No GPU resources or work exist.
// Retired handles stay allocated until registry destruction to reject stale IDs.
class HostRegistration {
public:
    using Handle = void *;
    Handle register_module(const void *, ReadableSpan, ReadableSpan, const std::string &);
    void register_function(Handle, const void *host_token, const std::string &device_name);
    const Kernel &lookup(const void *host_token) const;
    const BundledCodeObject &module(Handle) const;
    void unregister_module(Handle);
    size_t active_modules() const;
    size_t active_functions() const;
private:
    struct Binding { const void *token; std::string name; };
    struct Module { bool active = true; BundledCodeObject code; std::vector<Binding> bindings; };
    Module &find(Handle) const;
    std::vector<std::unique_ptr<Module>> modules_;
};
}
