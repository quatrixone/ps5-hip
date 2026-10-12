// Offline control only: genuine compiler registration, deliberately unsupported dispatch.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/host_registration.h"
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
#include "ps5hip/host_launch.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
extern const uint8_t __ps5hip_wrappers_begin[], __ps5hip_wrappers_end[];
extern const uint8_t __ps5hip_bundle_begin[], __ps5hip_bundle_end[];
extern const uint8_t __ps5hip_names_begin[], __ps5hip_names_end[];
}
namespace {
constexpr bool fail_symbol =
#ifdef PS5HIP_REGISTRATION_FAIL_SYMBOL
    true;
#else
    false;
#endif
ps5hip::ReadableSpan extent(const uint8_t *begin, const uint8_t *end) {
    const auto lo = reinterpret_cast<uintptr_t>(begin), hi = reinterpret_cast<uintptr_t>(end);
    if (hi < lo) throw ps5hip::ObjectError("reversed linked extent");
    return {begin, hi - lo};
}
std::string name_string(const char *text) {
    const auto span = extent(__ps5hip_names_begin, __ps5hip_names_end);
    const auto lo = reinterpret_cast<uintptr_t>(span.data), p = reinterpret_cast<uintptr_t>(text);
    if (p < lo || p - lo >= span.size) throw ps5hip::ObjectError("name outside linked read-only extent");
    const auto remaining = span.size - (p - lo);
    for (size_t i = 0; i < remaining && i < 1024; ++i) if (!text[i]) return {text, i};
    throw ps5hip::ObjectError("unterminated or oversized kernel name");
}
struct Configuration { dim3 grid, block; size_t shared; hipStream_t stream; };
struct State {
    ps5hip::HostRegistration registry;
    void *handle = nullptr;
    unsigned registered = 0, functions = 0, removed = 0, failures = 0, launches = 0;
    bool main_finished = false;
    hipError_t last_error = hipSuccess;
    std::vector<Configuration> configurations; // Serialized control only, not HIP thread semantics.
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
    std::vector<ps5hip::LaunchArgument> captured;
    ps5hip::InitialLaunch prepared{};
    std::vector<uint32_t> packets;
    // Synthetic addresses only: these are not allocated, mapped or submitted.
    const std::vector<ps5hip::GpuRange> live_buffers{{0x4000008000, 512}, {0x400000c000, 512}, {0x4000010000, 512}};
#endif
    ~State() {
        const bool valid = main_finished && registered == 1 && removed == 1 &&
            functions == (fail_symbol ? 0u : 1u) && failures == (fail_symbol ? 1u : 0u) &&
            launches == (fail_symbol ? 0u : 1u) && registry.active_modules() == 0 && configurations.empty();
        std::printf("REGISTRY_FINAL active=%zu functions=%zu valid=%u hardware_validated=false\n",
                    registry.active_modules(), registry.active_functions(), unsigned(valid));
        if (!valid) std::abort();
    }
};
State &state() { static State value; return value; }
}
extern "C" void **__hipRegisterFatBinary(const void *wrapper) {
    auto &s = state();
    try {
        s.handle = s.registry.register_module(wrapper, extent(__ps5hip_wrappers_begin, __ps5hip_wrappers_end),
            extent(__ps5hip_bundle_begin, __ps5hip_bundle_end), "gfx1013");
        ++s.registered;
        std::printf("COMPILER_REGISTER module_owned=true\n");
        return reinterpret_cast<void **>(s.handle);
    } catch (...) { ++s.failures; s.last_error = hipErrorInvalidValue; return nullptr; }
}
extern "C" void __hipRegisterFunction(void **handle, const void *host, char *device_function,
    const char *device_name, unsigned limit, uint3 *tid, uint3 *bid, dim3 *block, dim3 *grid, int *warp) {
    auto &s = state();
    try {
        if (limit != UINT32_MAX || tid || bid || block || grid || warp ||
            name_string(device_function) != name_string(device_name))
            throw ps5hip::ObjectError("unmeasured registration arguments");
        s.registry.register_function(handle, host, fail_symbol ? "deliberately_absent" : name_string(device_name));
        ++s.functions;
        std::printf("COMPILER_FUNCTION %s host_token_bound=true\n", device_name);
    } catch (...) {
        ++s.failures; s.last_error = hipErrorInvalidDeviceFunction;
        // Failed partial registration owns no work: release its owned bytes now.
        try { s.registry.unregister_module(handle); } catch (...) {}
    }
}
extern "C" void __hipUnregisterFatBinary(void **handle) {
    auto &s = state();
    try {
        if (!s.main_finished || handle != s.handle) throw ps5hip::ObjectError("unexpected unregister lifecycle");
        if (!fail_symbol) s.registry.unregister_module(handle);
        else if (s.registry.active_modules() != 0) throw ps5hip::ObjectError("partial registration retained storage");
        ++s.removed;
        std::printf("COMPILER_UNREGISTER active=%zu after_main=true\n", s.registry.active_modules());
    } catch (...) { ++s.failures; s.last_error = hipErrorInvalidValue; }
}
extern "C" hipError_t __hipPushCallConfiguration(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
    try { state().configurations.push_back({grid, block, shared, stream}); return hipSuccess; }
    catch (...) { return state().last_error = hipErrorOutOfMemory; }
}
extern "C" hipError_t __hipPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, hipStream_t *stream) {
    if (!grid || !block || !shared || !stream) return state().last_error = hipErrorInvalidValue;
    *grid = dim3(0); *block = dim3(0); *shared = 0; *stream = nullptr;
    auto &configurations = state().configurations;
    if (configurations.empty()) return state().last_error = hipErrorInvalidValue;
    const auto c = configurations.back(); configurations.pop_back();
    *grid = c.grid; *block = c.block; *shared = c.shared; *stream = c.stream;
    return hipSuccess;
}
extern "C" hipError_t hipLaunchKernel(const void *host, dim3 grid, dim3 block, void **args,
                                      size_t shared, hipStream_t stream) {
    auto &s = state();
    try {
        const auto &kernel = s.registry.lookup(host);
        unsigned n = 0;
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
        if (!args || kernel.arguments.size() != 4 || !ps5hip::initial_launch_rejection(kernel).empty())
            throw ps5hip::ObjectError("unexpected compiler argument layout");
#else
        if (!args || !args[3]) throw ps5hip::ObjectError("missing compiler arguments");
        std::memcpy(&n, args[3], sizeof(n));
#endif
        if (kernel.name != "_Z10vector_addPKfS0_Pfj" || grid.x != 2 || grid.y != 1 || grid.z != 1 ||
            block.x != 64 || block.y != 1 || block.z != 1 || shared || stream)
            throw ps5hip::ObjectError("unexpected compiler launch arguments");
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
        const ps5hip::GpuRange module{0x200040000, 0x4000}, kernarg{0x4000000000, 0x4000};
        // The inventory is permitted data storage only, including each whole allocation.
        for (const auto range : s.live_buffers)
            for (const auto reserved : {module, kernarg})
                if (range.address < reserved.address + reserved.bytes && reserved.address < range.address + range.bytes)
                    throw ps5hip::ObjectError("data inventory aliases launch storage");
        std::vector<ps5hip::ReadableSpan> slots;
        for (size_t i = 0; i < kernel.arguments.size(); ++i)
            slots.push_back({reinterpret_cast<const uint8_t *>(args[i]), kernel.arguments[i].size});
        // Readability of the exact array and slots is trusted only for this pinned compiler stub.
        s.captured = ps5hip::capture_initial_arguments(kernel,
            {reinterpret_cast<const uint8_t *>(args), kernel.arguments.size() * sizeof(void *)}, slots, s.live_buffers);
        n = static_cast<unsigned>(s.captured.at(3).value);
        if (n != 73) throw ps5hip::ObjectError("unexpected vector element count");
        for (size_t i = 0; i < 3; ++i)
            if (s.captured[i].buffer_bytes < uint64_t(n) * sizeof(float))
                throw ps5hip::ObjectError("vector extent is shorter than its element count");
        s.prepared = ps5hip::prepare_initial_launch(s.registry.module(s.handle).object, kernel.name,
                                                  module, kernarg, grid.x, s.captured);
        s.packets = ps5hip::encode_initial_dispatch(s.prepared);
        std::printf("COMPILER_CAPTURE arguments=4 kernarg_bytes=%zu packet_words=%zu synthetic_gpu_addresses=true gpu_calls=0\n",
                    s.prepared.kernargs.size(), s.packets.size());
#endif
        if (n != 73) throw ps5hip::ObjectError("unexpected vector element count");
        ++s.launches;
        std::printf("COMPILER_LAUNCH unsupported=true groups=2 block=64 n=73 gpu_calls=0\n");
        return s.last_error = hipErrorNotSupported;
    } catch (...) { ++s.failures; return s.last_error = hipErrorInvalidValue; }
}
void launch_vector_add(const float *, const float *, float *, unsigned);
int main() {
    auto &s = state();
    if (s.registered != 1 || s.functions != (fail_symbol ? 0u : 1u)) return 1;
    if (fail_symbol) {
        if (s.registry.active_modules() || s.last_error != hipErrorInvalidDeviceFunction) return 2;
    } else {
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
        const float *a = reinterpret_cast<const float *>(uintptr_t(0x4000008010));
        const float *b = reinterpret_cast<const float *>(uintptr_t(0x400000c020));
        float *out = reinterpret_cast<float *>(uintptr_t(0x4000010030));
        unsigned n = 73;
        launch_vector_add(a, b, out, n);
        a = nullptr; b = nullptr; out = nullptr; n = 0;
        if (a || b || out || n || s.captured.size() != 4 || s.captured[0].value != 0x4000008010 ||
            s.captured[1].value != 0x400000c020 || s.captured[2].value != 0x4000010030 ||
            s.captured[3].value != 73 || s.captured[0].buffer_bytes != 496 ||
            s.captured[1].buffer_bytes != 480 || s.captured[2].buffer_bytes != 464 ||
            s.prepared.kernargs.size() != 288 || s.prepared.kernargs.at(24) != 73 ||
            s.prepared.kernargs.at(32) != 2 || s.packets.empty()) return 4;
        std::printf("CAPTURE_OWNED after_stub_return=true caller_values_changed=true synthetic_model=true\n");
#else
        launch_vector_add(nullptr, nullptr, nullptr, 73);
#endif
        if (s.last_error != hipErrorNotSupported || s.launches != 1 || !s.configurations.empty()) return 3;
    }
    s.main_finished = true;
#ifdef PS5HIP_ARGUMENT_CAPTURE_CONTROL
    std::printf("MAIN_PASS partial_failure=%u launch_preparation_only=true hardware_validated=false\n", unsigned(fail_symbol));
#else
    std::printf("MAIN_PASS partial_failure=%u registration_only=true hardware_validated=false\n", unsigned(fail_symbol));
#endif
}
