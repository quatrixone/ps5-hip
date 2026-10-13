// Serialized bounded HIP runtime; client policy is a separately linked contract.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/host_registration.h"
#include "ps5hip/host_launch.h"
#include "ps5hip/native_launch.h"
#include "ps5hip/submission.h"
#include "ps5hip/hip_runtime_control.h"
#if !defined(PS5HIP_VECTOR_MODEL) && !defined(PS5HIP_RUNTIME_MODEL)
#include "ps5hip/agc_backend.h"

#endif
#include <cstdio>
#include <cstring>
#include <new>

extern "C" {
extern const uint8_t __ps5hip_wrappers_begin[], __ps5hip_wrappers_end[];
extern const uint8_t __ps5hip_bundle_begin[], __ps5hip_bundle_end[];
extern const uint8_t __ps5hip_names_begin[], __ps5hip_names_end[];
}
namespace {
struct Configuration { dim3 grid, block; size_t shared; hipStream_t stream; };
bool fixed_diagnostic() noexcept {
    return ps5hip::hip_runtime_contract().mode == ps5hip::HipRuntimeMode::FixedDiagnostic;
}
bool completion_expected(unsigned completed) noexcept {
    return !fixed_diagnostic() || completed == ps5hip::hip_runtime_contract().expected_completions;
}
bool functions_ready(size_t count) noexcept {
    return count != 0 && (!fixed_diagnostic() || count == ps5hip::hip_runtime_contract().expected_functions);
}

// Trivial process-lifetime observation; callbacks never access a destroyed State.
struct FinalObservation { bool checked, destroyed; ps5hip::HipRuntimeObservation snapshot; } final_observation{};
struct DestructionSentinel { ~DestructionSentinel() { final_observation.destroyed=true; } };
struct State {

    DestructionSentinel destruction_sentinel; // First member is destroyed last.
    ps5hip::HostRegistration registration;
    ps5hip::NativeDataRegistry data;
    std::unique_ptr<ps5hip::SubmissionBackend> backend;
    std::unique_ptr<ps5hip::SubmissionQueue> queue; // Destroyed before its backend.
    void *handle = nullptr;
    const void *host_token = nullptr;
    std::vector<std::pair<std::string,const void *>> tokens;
    unsigned function_attempts=0;
    std::vector<Configuration> configurations;
    hipError_t error = hipSuccess;
    bool terminal = false, init_attempted = false, partial = false, main_finished = false;
    unsigned registered = 0, removed = 0, completed = 0;
    ~State() {

        try {
            final_observation.snapshot={registration.active_modules(),registration.active_functions(),completed,init_attempted};
            final_observation.checked = registered==1 && removed==1 && completion_expected(completed) &&
                !registration.active_modules() && !registration.active_functions() && !partial && !terminal && !data.poisoned() &&
                data.ranges().empty() && configurations.empty() && main_finished &&
                (queue ? !queue->poisoned() : (!fixed_diagnostic() && !init_attempted));
            if(final_observation.checked) { queue.reset(); backend.reset(); }
        } catch (...) { final_observation.checked=false; }
        std::printf("VECTOR_FINAL registered=%u removed=%u completed=%u active=%zu terminal=%u main_finished=%u hardware_validated=false\n",
            registered, removed, completed, registration.active_modules(), unsigned(terminal), unsigned(main_finished));
    }
};
State &state() { static State s; return s; }
hipError_t record(hipError_t error) { return state().error = error; }
void quarantine_if_uncertain(bool submission_boundary) noexcept {
    auto &s = state();
    if ((submission_boundary && s.queue && s.queue->poisoned()) || s.data.poisoned()) {
        s.terminal = true;
        s.data.poison();
    }
}
template<class F> hipError_t checked(F action, bool submission_boundary = false) {
    try {
        auto &s = state();
        if (s.terminal || s.data.poisoned()) return record(hipErrorUnknown);
        if (s.partial) return record(hipErrorInvalidDeviceFunction);
        action();
        return record(hipSuccess);
    } catch (const ps5hip::NativeDataError &error) {
        quarantine_if_uncertain(submission_boundary);
        std::fprintf(stderr, "VECTOR_ERROR %s\n", error.what());
        using S = ps5hip::NativeDataStatus;
        if (error.status() == S::InvalidValue) return record(hipErrorInvalidValue);
        if (error.status() == S::InvalidPointer) return record(hipErrorInvalidDevicePointer);
        if (error.status() == S::InUse) return record(hipErrorNotReady);
    } catch (const ps5hip::NativeMemoryCleanupError &error) {
        state().terminal=true; state().data.poison();
        std::fprintf(stderr,"VECTOR_CLEANUP_FAILURE gpu_unresolved=false detail=%s\n",error.what());
    } catch (const ps5hip::NativeMemorySetupError &error) {
        if (error.retained_backing) { state().terminal=true; state().data.poison(); }
        quarantine_if_uncertain(submission_boundary);
        std::fprintf(stderr, "VECTOR_ERROR %s\n", error.what());
    } catch (const std::bad_alloc &) {
        quarantine_if_uncertain(submission_boundary);
        return record(state().terminal ? hipErrorUnknown : hipErrorOutOfMemory);
    } catch (const std::exception &error) {
        quarantine_if_uncertain(submission_boundary);
        std::fprintf(stderr, "VECTOR_ERROR %s\n", error.what());
    } catch (...) {
        quarantine_if_uncertain(submission_boundary);
        std::fprintf(stderr, "VECTOR_ERROR unknown exception\n");
    }
    return record(hipErrorUnknown);
}
ps5hip::ReadableSpan extent(const uint8_t *begin, const uint8_t *end) {
    const auto lo = reinterpret_cast<uintptr_t>(begin), hi = reinterpret_cast<uintptr_t>(end);
    if (hi < lo) throw ps5hip::ObjectError("reversed linked extent");
    return {begin, hi - lo};
}
std::string name_string(const char *text) {
    const auto span = extent(__ps5hip_names_begin, __ps5hip_names_end);
    const auto lo = reinterpret_cast<uintptr_t>(span.data), p = reinterpret_cast<uintptr_t>(text);
    if (p < lo || p - lo >= span.size) throw ps5hip::ObjectError("name outside linked extent");
    for (size_t i = 0; i < span.size - (p-lo) && i < 1024; ++i)
        if (!text[i]) return {text, i};
    throw ps5hip::ObjectError("unbounded name");
}
void initialize_backend() {
    auto &s = state();
    if (s.queue) return;
    if (s.init_attempted) throw ps5hip::ObjectError("backend initialization already failed");
    s.init_attempted = true;
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
    s.backend = ps5hip_runtime_model_backend();
#else
    s.backend = std::make_unique<ps5hip::AgcBackend>();
#endif
    s.queue = std::make_unique<ps5hip::SubmissionQueue>(*s.backend);
}
}
extern "C" void **__hipRegisterFatBinary(const void *wrapper) {
    try {
        auto &s = state();
        if(s.terminal || s.partial || s.registered || s.removed)
            throw ps5hip::ObjectError("registration blocked by lifecycle");
        s.handle = s.registration.register_module(wrapper, extent(__ps5hip_wrappers_begin,__ps5hip_wrappers_end),
            extent(__ps5hip_bundle_begin,__ps5hip_bundle_end), "gfx1013");
        ++s.registered;
        std::puts("VECTOR_REGISTER owned=true gpu_calls=0");
        return reinterpret_cast<void **>(s.handle);
    } catch (...) {
        if(!fixed_diagnostic()) {
            auto &s=state(); s.partial=true;
            try { if(!s.terminal && !s.data.poisoned() && s.registration.active_modules()) s.registration.unregister_module(s.handle); } catch (...) {}
            s.tokens.clear();
        }
        (void)record(hipErrorInvalidValue); return nullptr;
    }
}
extern "C" void __hipRegisterFunction(void **handle, const void *host, char *function,
    const char *name, unsigned limit, uint3 *tid, uint3 *bid, dim3 *block, dim3 *grid, int *warp) {
    try {
        auto &s = state();
        if(s.terminal || s.partial || s.removed || handle!=s.handle || !s.handle)
            throw ps5hip::ObjectError("function registration blocked by lifecycle");
        ++s.function_attempts;
        if (limit != UINT32_MAX || tid || bid || block || grid || warp || name_string(function) != name_string(name))
            throw ps5hip::ObjectError("unsupported registration ABI");
#if defined(PS5HIP_VECTOR_PARTIAL) || defined(PS5HIP_RUNTIME_PARTIAL)
        if(s.function_attempts==(fixed_diagnostic() ? ps5hip::hip_runtime_contract().expected_functions : 2u))
            s.registration.register_function(handle, host, "deliberately_absent");
        else
#endif
        s.registration.register_function(handle, host, name_string(name));
        if(s.registration.active_functions()>(fixed_diagnostic() ? ps5hip::hip_runtime_contract().expected_functions : 256u))
            throw ps5hip::ObjectError("too many registered functions");
        s.tokens.emplace_back(name_string(name),host);
        s.host_token = host;
        std::puts("VECTOR_FUNCTION bound=true");
    } catch (...) {
        auto &s = state(); s.partial = true; s.error = hipErrorInvalidDeviceFunction;
        try { if(!s.terminal && !s.data.poisoned() && s.registration.active_modules()) s.registration.unregister_module(s.handle); } catch (...) {}
        s.tokens.clear();
    }
}
extern "C" void __hipUnregisterFatBinary(void **handle) {
    auto &s = state();
    try {
        if (s.terminal || s.removed || !s.main_finished || handle != s.handle || !s.configurations.empty())
            throw ps5hip::ObjectError("unregister blocked by lifecycle or terminal failure");
        if (!s.partial) s.registration.unregister_module(handle);
        else if (s.registration.active_modules()) throw ps5hip::ObjectError("partial registration not unwound");
        s.tokens.clear();
        ++s.removed;
        std::puts("VECTOR_UNREGISTER after_main=true active=0");
    } catch (...) { s.error = hipErrorUnknown; std::puts("VECTOR_UNREGISTER rejected=true"); }
}
extern "C" hipError_t __hipPushCallConfiguration(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
    return checked([&] { state().configurations.push_back({grid,block,shared,stream}); });
}
extern "C" hipError_t __hipPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, hipStream_t *stream) {
    if (!grid || !block || !shared || !stream) return record(hipErrorInvalidValue);
    *grid=dim3(0); *block=dim3(0); *shared=0; *stream=nullptr;
    if (state().configurations.empty()) return record(hipErrorInvalidValue);
    return checked([&] {
        auto &v=state().configurations; const auto c=v.back();v.pop_back();
        *grid=c.grid;*block=c.block;*shared=c.shared;*stream=c.stream;
    });
}
extern "C" hipError_t hipLaunchKernel(const void *host, dim3 grid, dim3 block, void **args, size_t shared, hipStream_t stream) {
    auto &s=state();
    if (s.terminal || s.data.poisoned()) return record(hipErrorUnknown);
    if(stream) return record(hipErrorNotSupported);
    if(s.partial || !functions_ready(s.registration.active_functions()))
        return record(hipErrorInvalidDeviceFunction);
    std::vector<ps5hip::LaunchArgument> values;
    const ps5hip::Kernel *kernel=nullptr;
    ps5hip::NativeLaunchRequest request;
    try { kernel=&s.registration.lookup(host); }
    catch (const std::bad_alloc &) { return record(hipErrorOutOfMemory); }
    catch (...) { return record(hipErrorInvalidValue); }
    try {
        request=ps5hip::hip_runtime_request(*kernel,{grid.x,grid.y,grid.z},{block.x,block.y,block.z},shared);
    } catch (const std::bad_alloc &) { return record(hipErrorOutOfMemory); }
    catch (...) { return record(hipErrorNotSupported); }
    try {
        ps5hip::validate_capture_layout(*kernel,request.route);
        if(!args) throw ps5hip::ObjectError("missing compiler argument array");
        const size_t count=kernel->arguments.size();
        std::vector<ps5hip::ReadableSpan> slots;
        // Only genuine compiler-stub arrays/values are trusted readable here.
        // Metadata was completely validated before reading any args[i].
        for(size_t i=0;i<count;++i)
            slots.push_back({reinterpret_cast<const uint8_t *>(args[i]),kernel->arguments[i].size});
        values=ps5hip::capture_launch_arguments(*kernel,request.route,
            {reinterpret_cast<const uint8_t *>(args),count*sizeof(void *)},slots,s.data.ranges());
        ps5hip::hip_runtime_validate_arguments(*kernel,request,values);
    } catch(const std::bad_alloc &) { return record(hipErrorOutOfMemory); }
    catch (...) { return record(hipErrorInvalidValue); }
    return checked([&] {
        auto launch=ps5hip::hip_runtime_contract().vector_compatibility ?
            ps5hip::prepare_native_vector(s.registration.module(s.handle),*kernel,values,s.data) :
            ps5hip::prepare_native_launch(s.registration.module(s.handle),*kernel,values,s.data,request);
        auto close_temporaries = [&] {
            try { ps5hip::close_native_launch_temporaries(launch,s.data); }
            catch (...) { s.terminal=true;throw; }
        };
        try {
            initialize_backend();
            const auto result=s.queue->submit_and_wait(launch.commands,launch.resources,2000);
            if(result.status!=ps5hip::SubmissionStatus::Complete) {
                std::fprintf(stderr,"VECTOR_SUBMISSION status=%u driver=0x%08x\n",unsigned(result.status),unsigned(result.driver_error));
                throw ps5hip::ObjectError("vector submission incomplete");
            }
        } catch (...) {
            if(!s.queue || !s.queue->poisoned()) close_temporaries();
            throw;
        }
        close_temporaries();
        ++s.completed;
        if(ps5hip::hip_runtime_contract().vector_compatibility)
            std::printf("VECTOR_COMPLETE sequence=%u n=73 blocking=true\n",s.completed);
        else std::printf("RUNTIME_COMPLETE sequence=%u kernel=%s blocking=true\n",s.completed,kernel->name.c_str());
    }, true);
}
extern "C" hipError_t hipMalloc(void **pointer,size_t bytes) {
    if(!pointer)return record(hipErrorInvalidValue);
    *pointer=nullptr;
    return checked([&]{if(bytes)*pointer=state().data.allocate(bytes);});
}
extern "C" hipError_t hipFree(void *pointer) { return checked([&]{if(pointer)state().data.free(pointer);}); }
extern "C" hipError_t hipMemcpy(void *destination,const void *source,size_t bytes,hipMemcpyKind kind) {
    if(kind!=hipMemcpyHostToDevice && kind!=hipMemcpyDeviceToHost)return record(hipErrorNotSupported);
    return checked([&]{if(!bytes)return; if(kind==hipMemcpyHostToDevice)state().data.copy_to_device(destination,source,bytes);
        else state().data.copy_from_device(destination,source,bytes);});
}
extern "C" hipError_t hipDeviceSynchronize() {
    auto &s=state();
    if(s.terminal || s.data.poisoned())return record(hipErrorUnknown);
    if(s.partial)return record(hipErrorInvalidDeviceFunction);
    if(s.queue && s.queue->poisoned())return record(hipErrorNotReady);
    // Every supported launch has already waited; no stream work is deferred.
    return record(hipSuccess);
}
extern "C" hipError_t hipPeekAtLastError(){return state().error;}
extern "C" hipError_t hipGetLastError(){auto e=state().error;state().error=hipSuccess;return e;}
extern "C" const char *hipGetErrorString(hipError_t e){
    switch(e){case hipSuccess:return "success";case hipErrorInvalidValue:return "invalid vector value";
    case hipErrorInvalidDevicePointer:return "invalid device pointer";case hipErrorInvalidDeviceFunction:return "invalid device function";
    case hipErrorNotReady:return "data GPU-owned";case hipErrorNotSupported:return "unmeasured operation";
    case hipErrorOutOfMemory:return "host allocation failed";default:return "native/control failure";}
}
extern "C" void ps5hip_runtime_finish(){state().main_finished=true;}
extern "C" bool ps5hip_runtime_terminal(){return state().terminal;}
extern "C" const void *ps5hip_runtime_token(){return state().host_token;}
extern "C" void **ps5hip_runtime_handle(){return reinterpret_cast<void **>(state().handle);}
extern "C" bool ps5hip_runtime_gpu_unresolved(){return state().queue && state().queue->poisoned();}
extern "C" const void *ps5hip_runtime_token_for(const char *name) {
    if(!name)return nullptr;
    auto &s=state();
    if(s.partial || s.terminal || s.removed)return nullptr;
    for(const auto &item:s.tokens) if(item.first==name) {
        try { (void)s.registration.lookup(item.second);return item.second; } catch(...) { return nullptr; }
    }
    return nullptr;
}


extern "C" bool ps5hip_runtime_registration_ready() noexcept {
    try {
        const auto &s=state();
        if(s.registered!=1 || !s.host_token || s.partial || s.error!=hipSuccess ||
           s.registration.active_modules()!=1 || !functions_ready(s.registration.active_functions()) || s.terminal || s.data.poisoned() ||
           !s.data.ranges().empty() || s.backend || s.queue || s.init_attempted) return false;
        if(ps5hip::hip_runtime_contract().vector_compatibility)
            std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0");
        return true;
    } catch (...) { return false; }
}
extern "C" bool ps5hip_runtime_ready() noexcept {
    try {
        const auto &s=state();
        return s.registered==1 && !s.partial && !s.removed && completion_expected(s.completed) && s.main_finished &&
            !s.terminal && !s.data.poisoned() && s.data.ranges().empty() &&
            s.configurations.empty() && (s.queue ? !s.queue->poisoned() : (!fixed_diagnostic() && !s.init_attempted)) &&
            s.registration.active_modules()==1 && functions_ready(s.registration.active_functions());
    } catch (...) { return false; }
}
extern "C" bool ps5hip_runtime_finalized() noexcept {
    return final_observation.checked && final_observation.destroyed;
}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation() noexcept {
    if(final_observation.destroyed)return final_observation.snapshot;
    const auto &s=state();
    return {s.registration.active_modules(),s.registration.active_functions(),s.completed,s.init_attempted};
}
