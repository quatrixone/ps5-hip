#include "ps5hip/host_registration.h"
#include <cstring>
#include <limits>

namespace ps5hip {
namespace {
void require(bool value, const char *message) { if (!value) throw ObjectError(message); }
uintptr_t checked_span(ReadableSpan span) {
    const auto base = reinterpret_cast<uintptr_t>(span.data);
    require(base && span.size <= std::numeric_limits<uintptr_t>::max() - base,
            "invalid host-image span");
    return base;
}
uint64_t little(const uint8_t *p, unsigned count) {
    uint64_t result = 0;
    for (unsigned i = 0; i < count; ++i) result |= uint64_t(p[i]) << (i * 8);
    return result;
}
}
BundledCodeObject read_single_host_wrapper(const void *wrapper, ReadableSpan wrappers,
                                          ReadableSpan bundle, const std::string &target) {
    static_assert(sizeof(uintptr_t) == 8, "measured HIP wrapper is a 64-bit host ABI");
    const auto wrapper_base = checked_span(wrappers), bundle_base = checked_span(bundle);
    require(wrappers.size == 24 && reinterpret_cast<uintptr_t>(wrapper) == wrapper_base,
            "expected exact single host wrapper");
    require(bundle.size >= 32 && bundle.size <= 64u * 1024u * 1024u,
            "invalid linked bundle extent");
    require(wrapper_base + wrappers.size <= bundle_base || bundle_base + bundle.size <= wrapper_base,
            "host wrapper overlaps bundle");
    const auto *p = wrappers.data;
    require(little(p, 4) == 0x48495046 && little(p + 4, 4) == 1,
            "unsupported HIP host wrapper");
    require(little(p + 16, 8) == 0, "unsupported HIP wrapper reserved field");
    require(little(p + 8, 8) == bundle_base, "wrapper does not name exact linked bundle");
    return read_fat_binary(bundle.data, bundle.size, target);
}
HostRegistration::Handle HostRegistration::register_module(const void *wrapper, ReadableSpan wrappers,
                                                          ReadableSpan bundle, const std::string &target) {
    require(active_modules() == 0, "single-module control already registered");
    auto entry = std::make_unique<Module>();
    entry->code = read_single_host_wrapper(wrapper, wrappers, bundle, target);
    const auto handle = entry.get();
    modules_.push_back(std::move(entry));
    return handle;
}
HostRegistration::Module &HostRegistration::find(Handle handle) const {
    for (const auto &entry : modules_) if (entry.get() == handle && entry->active) return *entry;
    throw ObjectError("invalid or retired registration handle");
}
void HostRegistration::register_function(Handle handle, const void *token, const std::string &name) {
    auto &entry = find(handle);
    require(token != nullptr, "null host function token");
    bool found = false;
    for (const auto &kernel : entry.code.object.kernels) if (kernel.name == name) found = true;
    require(found, "registered device symbol is not a kernel");
    for (const auto &binding : entry.bindings)
        require(binding.token != token && binding.name != name, "duplicate kernel binding");
    entry.bindings.push_back({token, name});
}
const Kernel &HostRegistration::lookup(const void *token) const {
    for (const auto &entry : modules_) if (entry->active)
        for (const auto &binding : entry->bindings) if (binding.token == token)
            for (const auto &kernel : entry->code.object.kernels)
                if (kernel.name == binding.name) return kernel;
    throw ObjectError("unregistered host function token");
}
const BundledCodeObject &HostRegistration::module(Handle handle) const { return find(handle).code; }
void HostRegistration::unregister_module(Handle handle) {
    auto &entry = find(handle);
    entry.active = false;
    entry.bindings.clear();
    entry.code = {};
}
size_t HostRegistration::active_modules() const {
    size_t count = 0; for (const auto &entry : modules_) count += entry->active; return count;
}
size_t HostRegistration::active_functions() const {
    size_t count = 0; for (const auto &entry : modules_) if (entry->active) count += entry->bindings.size(); return count;
}
}
