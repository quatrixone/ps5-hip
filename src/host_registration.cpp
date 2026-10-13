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
HostModuleSpan select_host_module(const void *wrapper, ReadableSpan table,
                                 ReadableSpan wrappers, ReadableSpan bundles) {
    checked_span(table); const auto wb=checked_span(wrappers), bb=checked_span(bundles);
    require(table.size && table.size%32==0 && table.size/32<=32, "invalid module span table");
    HostModuleSpan selected{}; unsigned matches=0;
    auto pair = [&](size_t at, unsigned offset, uintptr_t base, size_t size) {
        const auto lo=little(table.data+at+offset,8), hi=little(table.data+at+offset+8,8);
        require(hi>=lo && hi<=size, "module outside trusted image");
        return ReadableSpan{reinterpret_cast<const uint8_t *>(base+lo),size_t(hi-lo)};
    };
    for(size_t at=0;at<table.size;at+=32) {
        const auto w=pair(at,0,wb,wrappers.size), b=pair(at,16,bb,bundles.size);
        require(w.size==24 && b.size>=32 && b.size<=64u*1024u*1024u,"invalid module extents");
        const uintptr_t wl=reinterpret_cast<uintptr_t>(w.data), bl=reinterpret_cast<uintptr_t>(b.data);
        require(wl+w.size<=bl || bl+b.size<=wl,"overlapping module spans");
        for(size_t prior=0;prior<at;prior+=32) {
            const auto pw=pair(prior,0,wb,wrappers.size), pb=pair(prior,16,bb,bundles.size);
            for(auto x:{w,b}) for(auto y:{pw,pb}) {
                const auto a=reinterpret_cast<uintptr_t>(x.data), c=reinterpret_cast<uintptr_t>(y.data);
                require(a+x.size<=c || c+y.size<=a,"duplicate or overlapping module spans");
            }
        }
        if(w.data==wrapper) { selected={w,b}; ++matches; }
    }
    require(matches==1,"wrapper absent from module span table"); return selected;
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
    require(limit_ && limit_<=32 && active_modules()<limit_ && modules_.size()<32, "module limit exceeded");
    for(const auto &module:modules_) require(!module->active || module->wrapper!=wrapper,"duplicate module wrapper");
    require(bundle.size<=64u*1024u*1024u-admitted_bytes_,"aggregate module bytes exceeded");
    auto entry = std::make_unique<Module>();
    entry->wrapper=wrapper;
    entry->code = read_single_host_wrapper(wrapper, wrappers, bundle, target);
    const auto handle = entry.get();
    modules_.push_back(std::move(entry));
    admitted_bytes_+=bundle.size;
    return handle;
}
HostRegistration::Module &HostRegistration::find(Handle handle) const {
    for (const auto &entry : modules_) if (entry.get() == handle && entry->active) return *entry;
    throw ObjectError("invalid or retired registration handle");
}
void HostRegistration::register_function(Handle handle, const void *token, const std::string &name) {
    auto &entry = find(handle);
    require(token != nullptr, "null host function token");
    require(active_functions()<256,"global function limit exceeded");
    for(const auto &module:modules_) if(module->active)
        for(const auto &binding:module->bindings) require(binding.token!=token,"duplicate global host token");
    bool found = false;
    for (const auto &kernel : entry.code.object.kernels) if (kernel.name == name) found = true;
    require(found, "registered device symbol is not a kernel");
    for (const auto &binding : entry.bindings)
        require(binding.token != token && binding.name != name, "duplicate kernel binding");
    entry.bindings.push_back({token, name});
}
HostRegistration::Lookup HostRegistration::lookup_owned(const void *token) const {
    for (const auto &entry : modules_) if (entry->active)
        for (const auto &binding : entry->bindings) if (binding.token == token)
            for (const auto &kernel : entry->code.object.kernels)
                if (kernel.name == binding.name) return {entry.get(),kernel};
    throw ObjectError("unregistered host function token");
}
const Kernel &HostRegistration::lookup(const void *token) const { return lookup_owned(token).kernel; }
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
bool HostRegistration::modules_ready() const {
    if(!active_modules()) return false;
    for(const auto &entry:modules_) if(entry->active && entry->bindings.empty()) return false;
    return true;
}
}
