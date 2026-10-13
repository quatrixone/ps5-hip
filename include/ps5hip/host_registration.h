#pragma once
#include "ps5hip/fat_binary.h"
#include <memory>

namespace ps5hip {
struct ReadableSpan { const uint8_t *data; size_t size; };
struct HostModuleSpan { ReadableSpan wrapper, bundle; };
// Each 32-byte entry holds little-endian wrapper begin/end offsets followed by
// bundle begin/end offsets, relative to the respective immutable linker spans.
HostModuleSpan select_host_module(const void *, ReadableSpan table,
                                 ReadableSpan wrappers, ReadableSpan bundles);

// Spans are trusted readable, immutable host-image extents supplied by the linker
// or caller. This validates membership; it cannot discover mapped memory bounds.
BundledCodeObject read_single_host_wrapper(const void *wrapper, ReadableSpan wrappers,
                                          ReadableSpan bundle, const std::string &target);

// Metadata-only serialized control; one active module by default, at most 32.
// Retired handles stay allocated until registry destruction to reject stale IDs.
class HostRegistration {
public:
    using Handle = void *;
    explicit HostRegistration(size_t limit = 1) : limit_(limit) {}
    struct Lookup { Handle owner; const Kernel &kernel; };
    Handle register_module(const void *, ReadableSpan, ReadableSpan, const std::string &);
    void register_function(Handle, const void *host_token, const std::string &device_name);
    const Kernel &lookup(const void *host_token) const;
    Lookup lookup_owned(const void *host_token) const;
    const BundledCodeObject &module(Handle) const;
    void unregister_module(Handle);
    size_t active_modules() const;
    size_t active_functions() const;
    bool modules_ready() const;
private:
    struct Binding { const void *token; std::string name; };
    struct Module { bool active = true; const void *wrapper=nullptr; BundledCodeObject code; std::vector<Binding> bindings; };
    Module &find(Handle) const;
    std::vector<std::unique_ptr<Module>> modules_;
    size_t limit_, admitted_bytes_ = 0;
};
}
