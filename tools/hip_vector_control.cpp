// Bounded serialized vector control; genuine compiler ABI, no constructor GPU work.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/host_registration.h"
#include "ps5hip/host_launch.h"
#include "ps5hip/native_launch.h"
#include "ps5hip/submission.h"
#ifndef PS5HIP_VECTOR_MODEL
#include "ps5hip/agc_backend.h"
#else
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_vector_model_backend();
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
#ifdef PS5HIP_VECTOR_TITLE
// Trivial process-lifetime observation; callbacks never access a destroyed State.
struct FinalObservation { bool checked, destroyed; } final_observation{};
struct DestructionSentinel { ~DestructionSentinel() { final_observation.destroyed=true; } };
#endif
struct State {
#ifdef PS5HIP_VECTOR_TITLE
    DestructionSentinel destruction_sentinel; // First member is destroyed last.
#endif
    ps5hip::HostRegistration registration;
    ps5hip::NativeDataRegistry data;
    std::unique_ptr<ps5hip::SubmissionBackend> backend;
    std::unique_ptr<ps5hip::SubmissionQueue> queue; // Destroyed before its backend.
    void *handle = nullptr;
    const void *host_token = nullptr;
    std::vector<Configuration> configurations;
    hipError_t error = hipSuccess;
    bool terminal = false, init_attempted = false, partial = false, main_finished = false;
    unsigned registered = 0, removed = 0, completed = 0;
    ~State() {
#ifdef PS5HIP_VECTOR_TITLE
        try {
            final_observation.checked = registered==1 && removed==1 && completed==2 &&
                !registration.active_modules() && !registration.active_functions() && !terminal && !data.poisoned() &&
                data.ranges().empty() && configurations.empty() && main_finished &&
                queue && !queue->poisoned();
            if(final_observation.checked) { queue.reset(); backend.reset(); }
        } catch (...) { final_observation.checked=false; }
#endif
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
#ifdef PS5HIP_VECTOR_MODEL
    s.backend = ps5hip_vector_model_backend();
#else
    s.backend = std::make_unique<ps5hip::AgcBackend>();
#endif
    s.queue = std::make_unique<ps5hip::SubmissionQueue>(*s.backend);
}
}
extern "C" void **__hipRegisterFatBinary(const void *wrapper) {
    try {
        auto &s = state();
        s.handle = s.registration.register_module(wrapper, extent(__ps5hip_wrappers_begin,__ps5hip_wrappers_end),
            extent(__ps5hip_bundle_begin,__ps5hip_bundle_end), "gfx1013");
        ++s.registered;
        std::puts("VECTOR_REGISTER owned=true gpu_calls=0");
        return reinterpret_cast<void **>(s.handle);
    } catch (...) { (void)record(hipErrorInvalidValue); return nullptr; }
}
extern "C" void __hipRegisterFunction(void **handle, const void *host, char *function,
    const char *name, unsigned limit, uint3 *tid, uint3 *bid, dim3 *block, dim3 *grid, int *warp) {
    try {
        auto &s = state();
        if (limit != UINT32_MAX || tid || bid || block || grid || warp || name_string(function) != name_string(name))
            throw ps5hip::ObjectError("unsupported registration ABI");
#ifdef PS5HIP_VECTOR_PARTIAL
        s.registration.register_function(handle, host, "deliberately_absent");
#else
        s.registration.register_function(handle, host, name_string(name));
#endif
        s.host_token = host;
        std::puts("VECTOR_FUNCTION bound=true");
    } catch (...) {
        auto &s = state(); s.partial = true; s.error = hipErrorInvalidDeviceFunction;
        try { s.registration.unregister_module(handle); } catch (...) {}
    }
}
extern "C" void __hipUnregisterFatBinary(void **handle) {
    auto &s = state();
    try {
        if (s.terminal || !s.main_finished || handle != s.handle || !s.configurations.empty())
            throw ps5hip::ObjectError("unregister blocked by lifecycle or terminal failure");
        if (!s.partial) s.registration.unregister_module(handle);
        else if (s.registration.active_modules()) throw ps5hip::ObjectError("partial registration not unwound");
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
    if (grid.x!=2 || grid.y!=1 || grid.z!=1 || block.x!=64 || block.y!=1 || block.z!=1 || shared || stream)
        return record(hipErrorNotSupported);
    // Validate metadata and trusted compiler-stub argument storage before allocation/AGC.
    std::vector<ps5hip::LaunchArgument> values;
    const ps5hip::Kernel *kernel=nullptr;
    try {
        kernel=&s.registration.lookup(host);
        if (!args || kernel->name!="_Z10vector_addPKfS0_Pfj" || kernel->arguments.size()!=4 ||
            !ps5hip::initial_launch_rejection(*kernel).empty()) throw ps5hip::ObjectError("unmeasured vector ABI");
        std::vector<ps5hip::ReadableSpan> slots;
        for (size_t i=0;i<4;++i) slots.push_back({reinterpret_cast<const uint8_t *>(args[i]),kernel->arguments[i].size});
        values=ps5hip::capture_initial_arguments(*kernel,{reinterpret_cast<const uint8_t *>(args),4*sizeof(void *)},slots,s.data.ranges());
        if (values[3].value!=73) throw ps5hip::ObjectError("unmeasured element count");
        for (size_t i=0;i<3;++i) if(values[i].buffer_bytes<73*4) throw ps5hip::ObjectError("short vector buffer");
    } catch(const std::bad_alloc &) { return record(hipErrorOutOfMemory); }
    catch (...) { return record(hipErrorInvalidValue); }
    return checked([&] {
        auto launch=ps5hip::prepare_native_vector(s.registration.module(s.handle),*kernel,values,s.data);
        auto close_temporaries = [&] {
            bool closed=true;
            for(size_t i=0;i<2;++i) if(!launch.resources.at(i)->close()) closed=false;
            if(!closed) {
                s.terminal=true;s.data.poison();
                throw ps5hip::NativeMemoryCleanupError("vector temporary cleanup failed");
            }
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
        std::printf("VECTOR_COMPLETE sequence=%u n=73 blocking=true\n",s.completed);
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
extern "C" void ps5hip_vector_finish(){state().main_finished=true;}
extern "C" bool ps5hip_vector_terminal(){return state().terminal;}
#ifdef PS5HIP_VECTOR_MODEL
extern "C" const void *ps5hip_vector_token(){return state().host_token;}
extern "C" bool ps5hip_vector_gpu_unresolved(){return state().queue && state().queue->poisoned();}
#endif

#ifdef PS5HIP_VECTOR_TITLE
extern "C" bool ps5hip_vector_title_registration() noexcept {
    try {
        const auto &s=state();
        if(s.registered!=1 || !s.host_token || s.partial || s.error!=hipSuccess ||
           s.registration.active_modules()!=1 || s.registration.active_functions()!=1 || s.terminal || s.data.poisoned() ||
           !s.data.ranges().empty() || s.backend || s.queue || s.init_attempted) return false;
        std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0");
        return true;
    } catch (...) { return false; }
}
extern "C" bool ps5hip_vector_title_ready() noexcept {
    try {
        const auto &s=state();
        return s.registered==1 && !s.removed && s.completed==2 && s.main_finished &&
            !s.terminal && !s.data.poisoned() && s.data.ranges().empty() &&
            s.configurations.empty() && s.queue && !s.queue->poisoned() &&
            s.registration.active_modules()==1 && s.registration.active_functions()==1;
    } catch (...) { return false; }
}
extern "C" bool ps5hip_vector_title_finalized() noexcept {
    return final_observation.checked && final_observation.destroyed;
}
#endif
