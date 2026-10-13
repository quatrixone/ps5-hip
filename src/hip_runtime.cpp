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
#include <atomic>
#include <pthread.h>
#include <array>
#include <algorithm>

#ifdef PS5HIP_THREAD_MODEL
extern "C" bool ps5hip_thread_model_fail(unsigned point) noexcept;
#endif

extern "C" {
extern const uint8_t __ps5hip_wrappers_begin[], __ps5hip_wrappers_end[];
extern const uint8_t __ps5hip_bundle_begin[], __ps5hip_bundle_end[];
extern const uint8_t __ps5hip_names_begin[], __ps5hip_names_end[];
extern const uint8_t __ps5hip_module_spans_begin[] __attribute__((weak));
extern const uint8_t __ps5hip_module_spans_end[] __attribute__((weak));
}
namespace {
struct Configuration { dim3 grid, block; size_t shared; hipStream_t stream; };
struct ThreadContext {
    hipError_t error = hipSuccess;
    std::vector<Configuration> configurations;
};
enum class Lifetime { Open, Closing, Destroyed };
struct Gate {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_key_t key{};
    bool key_created = false, abandoned = false, owner_set = false, registration_failed = false;
    pthread_t owner{};
    Lifetime lifetime = Lifetime::Open;
    size_t contexts = 0, pending = 0;
} gate;
static_assert(std::atomic<unsigned>::is_always_lock_free, "fault latch must not allocate or lock");
std::atomic<unsigned> infrastructure_fault{0};
const unsigned char oom_marker = 0;

bool take_gate() noexcept {
    if (infrastructure_fault.load()) return false;
    if (pthread_mutex_lock(&gate.mutex)) {
        infrastructure_fault.store(1); return false;
    }
    if (!infrastructure_fault.load()) return true;
    if (pthread_mutex_unlock(&gate.mutex)) infrastructure_fault.store(1);
    return false;
}
bool release_gate() noexcept {
    if (pthread_mutex_unlock(&gate.mutex)) infrastructure_fault.store(1);
    return !infrastructure_fault.load();
}
bool is_marker(const void *value) noexcept { return value == &oom_marker; }
void *key_value() noexcept { return gate.key_created ? pthread_getspecific(gate.key) : nullptr; }
bool set_key(void *value) noexcept {
    if (!pthread_setspecific(gate.key, value)) return true;
    infrastructure_fault.store(1); return false;
}
void thread_destructor(void *value) noexcept {
    if (!value || is_marker(value)) return;
    // Never consult another TLS key, State or a HIP error helper during key teardown.
    if (!take_gate()) return;
    auto *context = static_cast<ThreadContext *>(value);
    if (!context->configurations.empty()) gate.abandoned = true;
    gate.pending -= context->configurations.size();
    --gate.contexts;
    if (release_gate()) delete context;
}
ThreadContext *ensure_context(hipError_t &error) noexcept {
    if (!gate.key_created) {
        if (pthread_key_create(&gate.key, thread_destructor)) {
            infrastructure_fault.store(1); error = hipErrorUnknown; return nullptr;
        }
        gate.key_created = true;
    }
    void *value = key_value();
    if (value && !is_marker(value)) return static_cast<ThreadContext *>(value);
    ThreadContext *context = nullptr;
#ifdef PS5HIP_THREAD_MODEL
    if (!ps5hip_thread_model_fail(1))
#endif
        context = new (std::nothrow) ThreadContext;
    if (!context) {
        error = set_key(const_cast<unsigned char *>(&oom_marker)) ? hipErrorOutOfMemory : hipErrorUnknown;
        return nullptr;
    }
    if (is_marker(value)) context->error = hipErrorOutOfMemory;
    // Failed installation leaves the new storage retained: the key state is uncertain.
    if (!set_key(context)) { error = hipErrorUnknown; return nullptr; }
    ++gate.contexts;
    return context;
}
bool clear_current_context() noexcept {
    void *value = key_value();
    if (!value) return true;
    if (!set_key(nullptr)) return false;
    if (!is_marker(value)) {
        auto *context = static_cast<ThreadContext *>(value);
        if (!context->configurations.empty()) gate.abandoned = true;
        gate.pending -= context->configurations.size();
        --gate.contexts;
        delete context;
    }
    return true;
}
template<class F> hipError_t call(F action, bool registration = false) noexcept {
    if (!take_gate()) return hipErrorUnknown;
    hipError_t result = hipErrorDeinitialized;
    if (gate.lifetime == Lifetime::Open && gate.abandoned) result = hipErrorUnknown;
    else if (gate.lifetime == Lifetime::Open) {
        auto *context = ensure_context(result);
        if (context) {
            try {
                if (!registration && gate.registration_failed) result = hipErrorInvalidDeviceFunction;
                else result = action();
            }
            catch (const std::bad_alloc &) { result = hipErrorOutOfMemory; }
            catch (...) { result = hipErrorUnknown; }
            context->error = result;
        }
    }
    if (registration && result != hipSuccess) gate.registration_failed = true;
    // Unlock status must be known before the ABI's return value is selected.
    return release_gate() ? result : hipErrorUnknown;
}
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
    ps5hip::HostRegistration registration{ps5hip::hip_runtime_contract().mode==ps5hip::HipRuntimeMode::ReusableModules ? 32u : 1u};
    ps5hip::NativeDataRegistry data;
    std::unique_ptr<ps5hip::SubmissionBackend> backend;
    std::unique_ptr<ps5hip::SubmissionQueue> queue; // Destroyed before its backend.
    void *handle = nullptr;
    const void *host_token = nullptr;
    struct Token { std::string name; const void *host; void *owner; };
    struct Module { void *handle=nullptr; bool callback=false; };
    std::array<Module,32> modules{};
    std::vector<Token> tokens;
    bool work_started=false, registration_complete=false;
    unsigned function_attempts=0;
    bool terminal = false, init_attempted = false, partial = false, main_finished = false;
    unsigned registered = 0, removed = 0, completed = 0;
    ~State() {

        try {
            final_observation.snapshot={registration.active_modules(),registration.active_functions(),completed,init_attempted};
            final_observation.checked = registration_complete && registered>0 && removed==registered && completion_expected(completed) &&
                !registration.active_modules() && !registration.active_functions() && !partial && !terminal && !data.poisoned() &&
                data.ranges().empty() && !gate.pending && !gate.contexts && !gate.abandoned && !gate.registration_failed && main_finished &&
                (queue ? !queue->poisoned() : (!fixed_diagnostic() && !init_attempted));
            if(final_observation.checked) { queue.reset(); backend.reset(); }
        } catch (...) { final_observation.checked=false; }
        std::printf("VECTOR_FINAL registered=%u removed=%u completed=%u active=%zu terminal=%u main_finished=%u hardware_validated=false\n",
            registered, removed, completed, registration.active_modules(), unsigned(terminal), unsigned(main_finished));
    }
};
State *state_pointer = nullptr;
struct StateOwner {
    ~StateOwner() {
        if (!take_gate()) return;
        // Main's pthread-key destructor is not guaranteed to run on process exit.
        const bool owner = gate.owner_set && pthread_equal(gate.owner, pthread_self());
        if (owner) clear_current_context();
        bool safe = false;
        if (!infrastructure_fault.load() && state_pointer)
            final_observation.snapshot = {state_pointer->registration.active_modules(),
                state_pointer->registration.active_functions(),state_pointer->completed,state_pointer->init_attempted};
        if (!infrastructure_fault.load() && owner && !gate.contexts && !gate.pending && state_pointer) {
            try {
                safe = gate.lifetime == Lifetime::Closing && !gate.abandoned && !gate.registration_failed &&
                    state_pointer->main_finished && state_pointer->registration_complete && state_pointer->registered > 0 && state_pointer->removed == state_pointer->registered &&
                    !state_pointer->registration.active_modules() && !state_pointer->registration.active_functions() &&
                    completion_expected(state_pointer->completed) && !state_pointer->partial &&
                    !state_pointer->terminal && !state_pointer->data.poisoned() &&
                    state_pointer->data.ranges().empty() &&
                    (state_pointer->queue ? !state_pointer->queue->poisoned() : (!fixed_diagnostic() && !state_pointer->init_attempted));
            } catch (...) {}
        }
        // No automatic owning member may destroy retained, potentially live State.
        if (safe) { delete state_pointer; state_pointer = nullptr; }
        else if (!infrastructure_fault.load() && state_pointer)
            std::printf("RUNTIME_RETAINED registered=%u removed=%u completed=%u active=%zu terminal=%u main_finished=%u contexts=%zu pending=%zu abandoned=%u registration_failed=%u hardware_validated=false\n",
                state_pointer->registered,state_pointer->removed,state_pointer->completed,
                state_pointer->registration.active_modules(),unsigned(state_pointer->terminal),unsigned(state_pointer->main_finished),
                gate.contexts,gate.pending,unsigned(gate.abandoned),unsigned(gate.registration_failed));
        else std::puts("RUNTIME_RETAINED infrastructure_unavailable=true hardware_validated=false");
        gate.lifetime = Lifetime::Destroyed;
        (void)release_gate();
    }
};
State &create_state() {
    if (!state_pointer) {
        static StateOwner owner;
        (void)owner;
#ifdef PS5HIP_THREAD_MODEL
        if (ps5hip_thread_model_fail(3)) throw std::bad_alloc();
#endif
        state_pointer = new State;
        gate.owner = pthread_self(); gate.owner_set = true;
    }
    return *state_pointer;
}
State &state() {
    if (!state_pointer) throw ps5hip::ObjectError("runtime State was not registered");
    return *state_pointer;
}
hipError_t record(hipError_t error) {
    auto *value = key_value();
    if (value && !is_marker(value)) static_cast<ThreadContext *>(value)->error = error;
    return error;
}
void quarantine_if_uncertain(bool submission_boundary) noexcept {
    // Error handling must never retry allocation of a failed initial State.
    if (!state_pointer) return;
    auto &s = *state_pointer;
    if ((submission_boundary && s.queue && s.queue->poisoned()) || s.data.poisoned()) {
        s.terminal = true;
        s.data.poison();
    }
}
template<class F> hipError_t checked(F action, bool submission_boundary = false) {
    try {
        if (gate.registration_failed) return record(hipErrorInvalidDeviceFunction);
        auto &s = state();
        if (s.terminal || s.data.poisoned()) return record(hipErrorUnknown);
        if (s.partial || gate.registration_failed) return record(hipErrorInvalidDeviceFunction);
        s.work_started=true;
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
        return record(state_pointer && state_pointer->terminal ? hipErrorUnknown : hipErrorOutOfMemory);
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
void unwind_registration(State &s) noexcept {
    s.partial=true;
    if(!s.terminal && !s.data.poisoned())
        for(unsigned i=0;i<s.registered;++i)
            try { s.registration.unregister_module(s.modules[i].handle); } catch (...) {}
    s.tokens.clear(); s.host_token=nullptr;
}
ps5hip::HostModuleSpan module_spans(const void *wrapper) {
    const auto wrappers=extent(__ps5hip_wrappers_begin,__ps5hip_wrappers_end);
    const auto bundles=extent(__ps5hip_bundle_begin,__ps5hip_bundle_end);
    if(__ps5hip_module_spans_begin || __ps5hip_module_spans_end) {
        if(ps5hip::hip_runtime_contract().mode!=ps5hip::HipRuntimeMode::ReusableModules ||
           !__ps5hip_module_spans_begin || !__ps5hip_module_spans_end)
            throw ps5hip::ObjectError("unsupported module span table");
        return ps5hip::select_host_module(wrapper,extent(__ps5hip_module_spans_begin,__ps5hip_module_spans_end),wrappers,bundles);
    }
    return {wrappers,bundles};
}
}
static void **__hipRegisterFatBinary_locked(const void *wrapper) {
    try {
        if (gate.registration_failed) { (void)record(hipErrorInvalidDeviceFunction); return nullptr; }
        auto &s = create_state();
        if(s.terminal || s.partial || gate.registration_failed || s.work_started || s.removed || s.registered>=s.modules.size())
            throw ps5hip::ObjectError("registration blocked by lifecycle");
        const auto spans=module_spans(wrapper);
        auto *handle=s.registration.register_module(wrapper,spans.wrapper,spans.bundle,"gfx1013");
        s.modules[s.registered].handle=handle;
        s.handle=s.registered ? nullptr : handle;
        ++s.registered;
        std::puts("VECTOR_REGISTER owned=true gpu_calls=0");
        return reinterpret_cast<void **>(handle);
    } catch (const std::bad_alloc &) {
        if (!fixed_diagnostic() && state_pointer) {
            unwind_registration(*state_pointer);
        }
        (void)record(hipErrorOutOfMemory); return nullptr;
    } catch (...) {
        if(!fixed_diagnostic() && state_pointer) {
            unwind_registration(*state_pointer);
        }
        (void)record(hipErrorInvalidValue); return nullptr;
    }
}
static void __hipRegisterFunction_locked(void **handle, const void *host, char *function,
    const char *name, unsigned limit, uint3 *tid, uint3 *bid, dim3 *block, dim3 *grid, int *warp) {
    if (!state_pointer) { (void)record(hipErrorInvalidDeviceFunction); return; }
    try {
        auto &s = state();
        if(s.terminal || s.partial || gate.registration_failed || s.work_started || s.removed || !handle)
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
        s.tokens.push_back({name_string(name),host,handle});
        s.host_token = host;
        std::puts("VECTOR_FUNCTION bound=true");
    } catch (...) {
        auto &s = state(); (void)record(hipErrorInvalidDeviceFunction); unwind_registration(s);
    }
}
static void __hipUnregisterFatBinary_locked(void **handle) {
    auto &s = state();
    try {
        auto module=std::find_if(s.modules.begin(),s.modules.begin()+s.registered,[&](const State::Module &m){return m.handle==handle;});
        if (s.terminal || !s.main_finished || module==s.modules.begin()+s.registered || module->callback || gate.pending || gate.contexts || gate.abandoned)
            throw ps5hip::ObjectError("unregister blocked by lifecycle or terminal failure");
        if (!s.partial) s.registration.unregister_module(handle);
        else if (s.registration.active_modules()) throw ps5hip::ObjectError("partial registration not unwound");
        s.tokens.erase(std::remove_if(s.tokens.begin(),s.tokens.end(),[&](const State::Token &t){return t.owner==handle;}),s.tokens.end());
        module->callback=true;
        ++s.removed;
        std::printf("VECTOR_UNREGISTER after_main=true active=%zu\n",s.registration.active_modules());
    } catch (...) { (void)record(hipErrorUnknown); std::puts("VECTOR_UNREGISTER rejected=true"); }
}
static hipError_t __hipPushCallConfiguration_locked(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
    return checked([&] {
#ifdef PS5HIP_THREAD_MODEL
        if (ps5hip_thread_model_fail(2)) throw std::bad_alloc();
#endif
        static_cast<ThreadContext *>(key_value())->configurations.push_back({grid,block,shared,stream});
        ++gate.pending;
    });
}
static hipError_t __hipPopCallConfiguration_locked(dim3 *grid, dim3 *block, size_t *shared, hipStream_t *stream) {
    if (!grid || !block || !shared || !stream) return record(hipErrorInvalidValue);
    *grid=dim3(0); *block=dim3(0); *shared=0; *stream=nullptr;
    if (static_cast<ThreadContext *>(key_value())->configurations.empty()) return record(hipErrorInvalidValue);
    return checked([&] {
        auto &v=static_cast<ThreadContext *>(key_value())->configurations; const auto c=v.back();v.pop_back(); --gate.pending;
        *grid=c.grid;*block=c.block;*shared=c.shared;*stream=c.stream;
    });
}
static hipError_t hipLaunchKernel_locked(const void *host, dim3 grid, dim3 block, void **args, size_t shared, hipStream_t stream) {
    auto &s=state();
    if (s.terminal || s.data.poisoned()) return record(hipErrorUnknown);
    if(stream) return record(hipErrorNotSupported);
    if(s.partial || gate.registration_failed || !s.registration.modules_ready() || !functions_ready(s.registration.active_functions()))
        return record(hipErrorInvalidDeviceFunction);
    std::vector<ps5hip::LaunchArgument> values;
    const ps5hip::Kernel *kernel=nullptr;
    void *owner=nullptr;
    ps5hip::NativeLaunchRequest request;
    try { const auto found=s.registration.lookup_owned(host); kernel=&found.kernel; owner=found.owner; }
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
            ps5hip::prepare_native_vector(s.registration.module(owner),*kernel,values,s.data) :
            ps5hip::prepare_native_launch(s.registration.module(owner),*kernel,values,s.data,request);
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
static hipError_t hipMalloc_locked(void **pointer,size_t bytes) {
    if(!pointer)return record(hipErrorInvalidValue);
    *pointer=nullptr;
    return checked([&]{if(bytes)*pointer=state().data.allocate(bytes);});
}
static hipError_t hipFree_locked(void *pointer) { return checked([&]{if(pointer)state().data.free(pointer);}); }
static hipError_t hipMemcpy_locked(void *destination,const void *source,size_t bytes,hipMemcpyKind kind) {
    if(kind!=hipMemcpyHostToDevice && kind!=hipMemcpyDeviceToHost && kind!=hipMemcpyDeviceToDevice)return record(hipErrorNotSupported);
    return checked([&]{if(!bytes)return; if(kind==hipMemcpyHostToDevice)state().data.copy_to_device(destination,source,bytes);
        else if(kind==hipMemcpyDeviceToHost)state().data.copy_from_device(destination,source,bytes);
        else state().data.copy_device(destination,source,bytes);});
}
static hipError_t hipDeviceSynchronize_locked() {
    auto &s=state();
    if(s.terminal || s.data.poisoned())return record(hipErrorUnknown);
    if(s.partial || gate.registration_failed)return record(hipErrorInvalidDeviceFunction);
    if(s.queue && s.queue->poisoned())return record(hipErrorNotReady);
    // Every supported launch has already waited; no stream work is deferred.
    return record(hipSuccess);
}
extern "C" void **__hipRegisterFatBinary(const void *wrapper) {
    void **result = nullptr;
    const auto status = call([&] { result = __hipRegisterFatBinary_locked(wrapper);
        return result ? hipSuccess : static_cast<ThreadContext *>(key_value())->error; }, true);
    return status == hipSuccess ? result : nullptr;
}
extern "C" void __hipRegisterFunction(void **handle, const void *host, char *function,
    const char *name, unsigned limit, uint3 *tid, uint3 *bid, dim3 *block, dim3 *grid, int *warp) {
    (void)call([&] {
        __hipRegisterFunction_locked(handle,host,function,name,limit,tid,bid,block,grid,warp);
        return !state_pointer || state_pointer->partial || gate.registration_failed ? hipErrorInvalidDeviceFunction : hipSuccess;
    }, true);
}
extern "C" void __hipUnregisterFatBinary(void **handle) {
    if (!take_gate()) { std::puts("VECTOR_UNREGISTER rejected=true"); return; }
    if (gate.lifetime == Lifetime::Closing && state_pointer)
        __hipUnregisterFatBinary_locked(handle);
    else std::puts("VECTOR_UNREGISTER rejected=true");
    (void)release_gate();
}
extern "C" hipError_t __hipPushCallConfiguration(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
    return call([&] { return __hipPushCallConfiguration_locked(grid,block,shared,stream); });
}
extern "C" hipError_t __hipPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, hipStream_t *stream) {
    return call([&] { return __hipPopCallConfiguration_locked(grid,block,shared,stream); });
}
extern "C" hipError_t hipLaunchKernel(const void *host, dim3 grid, dim3 block, void **args, size_t shared, hipStream_t stream) {
    return call([&] { return hipLaunchKernel_locked(host,grid,block,args,shared,stream); });
}
template<class F> static hipError_t device_call(F action) {
    return call([&] {
        if(state_pointer) {
            if(state_pointer->terminal || state_pointer->data.poisoned())return hipErrorUnknown;
            if(state_pointer->partial)return hipErrorInvalidDeviceFunction;
        }
        return action();
    });
}
extern "C" hipError_t hipInit(unsigned flags) {
    return device_call([&] { return flags ? hipErrorInvalidValue : hipSuccess; });
}
extern "C" hipError_t hipGetDeviceCount(int *count) {
    const auto result=device_call([&] { return count ? hipSuccess : hipErrorInvalidValue; });
    if(result==hipSuccess)*count=1;
    return result;
}
extern "C" hipError_t hipGetDevice(int *device) {
    const auto result=device_call([&] { return device ? hipSuccess : hipErrorInvalidValue; });
    if(result==hipSuccess)*device=0;
    return result;
}
extern "C" hipError_t hipSetDevice(int device) {
    return device_call([&] { return device==0 ? hipSuccess : hipErrorInvalidDevice; });
}
extern "C" hipError_t hipDeviceGetAttribute(int *value, hipDeviceAttribute_t attribute, int device) {
    const auto result=device_call([&] {
        if(!value)return hipErrorInvalidValue;
        if(device!=0)return hipErrorInvalidDevice;
        return attribute==hipDeviceAttributeWarpSize ? hipSuccess : hipErrorInvalidValue;
    });
    // Publish only after successful unlock; wave32 is the admitted execution ABI.
    if(result==hipSuccess)*value=32;
    return result;
}
extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
    if (pointer) *pointer=nullptr;
    return call([&] { return hipMalloc_locked(pointer,bytes); });
}
static hipError_t allocation_query(void *pointer, void **base, size_t *bytes, hipError_t missing) {
    void *found=nullptr; size_t length=0;
    const auto result=call([&] {
        const auto status=checked([&] {
            using S=ps5hip::NativeDataStatus;
            if(!base || !bytes)throw ps5hip::NativeDataError(S::InvalidValue,"null allocation query output");
            if(!pointer)throw ps5hip::NativeDataError(S::InvalidPointer,"null allocation query pointer");
            const auto hold=state().data.resolve(pointer,1);
            found=reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(pointer)-hold.offset);
            length=hold.offset+hold.remaining;
        });
        return record(status==hipErrorInvalidDevicePointer ? missing : status);
    });
    // An unlock failure must not publish apparently successful query results.
    if(result==hipSuccess) { *base=found; *bytes=length; }
    return result;
}
extern "C" hipError_t hipMemPtrGetInfo(void *pointer, size_t *bytes) {
    void *base=nullptr;
    return allocation_query(pointer,&base,bytes,hipErrorInvalidValue);
}
extern "C" hipError_t hipMemGetAddressRange(hipDeviceptr_t *base, size_t *bytes, hipDeviceptr_t pointer) {
    return allocation_query(pointer,base,bytes,hipErrorNotFound);
}
extern "C" hipError_t hipFree(void *pointer) { return call([&] { return hipFree_locked(pointer); }); }
extern "C" hipError_t hipMemset(void *pointer, int value, size_t bytes) {
    return call([&] {
        const auto result=checked([&] { if(bytes)state().data.fill(pointer,value,bytes); });
        return record(result==hipErrorInvalidDevicePointer ? hipErrorInvalidValue : result);
    });
}
extern "C" hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind) {
    return call([&] { return hipMemcpy_locked(destination,source,bytes,kind); });
}
extern "C" hipError_t hipDeviceSynchronize() { return call([] { return hipDeviceSynchronize_locked(); }); }
namespace {
hipError_t last_error(bool reset) noexcept {
    if (!take_gate()) return hipErrorUnknown;
    hipError_t result = hipErrorDeinitialized;
    // An abandoned launch stack is a fatal lifecycle failure, not a clearable
    // thread-local error. Even a virgin peer must observe the failure.
    if (gate.lifetime == Lifetime::Open && gate.abandoned) result = hipErrorUnknown;
    else if (gate.lifetime == Lifetime::Open) {
        void *value = key_value();
        result = hipSuccess;
        if (is_marker(value)) {
            result = hipErrorOutOfMemory;
            if (reset && !set_key(nullptr)) result = hipErrorUnknown;
        } else if (value) {
            auto *context = static_cast<ThreadContext *>(value);
            result = context->error;
            if (reset) context->error = hipSuccess;
        }
    }
    return release_gate() ? result : hipErrorUnknown;
}
template<class T, class F> T observe(T failure, F action) noexcept {
    if (!take_gate()) return failure;
    T result = failure;
    try { result = action(); } catch (...) {}
    return release_gate() ? result : failure;
}
}
extern "C" hipError_t hipPeekAtLastError() { return last_error(false); }
extern "C" hipError_t hipGetLastError() { return last_error(true); }
extern "C" const char *hipGetErrorString(hipError_t e) {
    switch(e){case hipSuccess:return "success";case hipErrorInvalidValue:return "invalid vector value";
    case hipErrorInvalidDevicePointer:return "invalid device pointer";case hipErrorInvalidDeviceFunction:return "invalid device function";
    case hipErrorNotReady:return "data GPU-owned";case hipErrorNotSupported:return "unmeasured operation";
    case hipErrorDeinitialized:return "runtime closed";
    case hipErrorInvalidDevice:return "invalid device ordinal";
    case hipErrorNotFound:return "allocation not found";
    case hipErrorOutOfMemory:return "host allocation failed";default:return "native/control failure";}
}
extern "C" void ps5hip_runtime_finish() {
    if (!take_gate()) return;
    if (gate.lifetime == Lifetime::Open) {
        gate.lifetime = Lifetime::Closing;
        bool clean = gate.owner_set && pthread_equal(gate.owner,pthread_self()) && !gate.abandoned && !gate.pending;
        // Worker context cleanup must have completed before main closes the runtime.
        void *value = key_value();
        const size_t own = value && !is_marker(value) ? 1 : 0;
        clean = clean && gate.contexts == own;
        try {
            clean = clean && state_pointer && !state_pointer->terminal && !state_pointer->data.poisoned() &&
                state_pointer->data.ranges().empty() && (!state_pointer->queue || !state_pointer->queue->poisoned());
        } catch (...) { clean = false; }
        if (clean) clean = clear_current_context();
        if (clean && state_pointer) {
            state_pointer->main_finished = true;
            state_pointer->registration_complete=state_pointer->registered>0 &&
                state_pointer->registration.active_modules()==state_pointer->registered &&
                state_pointer->registration.modules_ready() && functions_ready(state_pointer->registration.active_functions());
        }
        else gate.abandoned = true;
    }
    (void)release_gate();
}
extern "C" bool ps5hip_runtime_terminal() {
    return observe(true, [] { return gate.lifetime == Lifetime::Destroyed ? !final_observation.checked :
        gate.abandoned || (state_pointer && state_pointer->terminal); });
}
extern "C" const void *ps5hip_runtime_token() {
    return observe<const void *>(nullptr, [] { return gate.lifetime == Lifetime::Open && state_pointer && state_pointer->registered==1 ? state_pointer->host_token : nullptr; });
}
extern "C" void **ps5hip_runtime_handle() {
    return observe<void **>(nullptr, [] { return gate.lifetime == Lifetime::Open && state_pointer ? reinterpret_cast<void **>(state_pointer->handle) : nullptr; });
}
extern "C" bool ps5hip_runtime_gpu_unresolved() {
    return observe(true, [] { return gate.lifetime == Lifetime::Destroyed ? !final_observation.checked :
        state_pointer && state_pointer->queue && state_pointer->queue->poisoned(); });
}
extern "C" const void *ps5hip_runtime_token_for(const char *name) {
    return observe<const void *>(nullptr, [&]() -> const void * {
        if (!name || gate.lifetime != Lifetime::Open || !state_pointer) return nullptr;
        const auto &s = *state_pointer;
        if(s.partial || s.terminal || s.removed) return nullptr;
        const void *found=nullptr;
        for(const auto &item:s.tokens) if(item.name==name) {
            if(found) return nullptr;
            (void)s.registration.lookup(item.host); found=item.host;
        }
        return found;
    });
}
extern "C" bool ps5hip_runtime_registration_ready() noexcept {
    return observe(false, [] {
        if(gate.lifetime != Lifetime::Open || !state_pointer) return false;
        const auto &s = *state_pointer;
        if(!s.registered || !s.host_token || s.partial || gate.abandoned || gate.registration_failed ||
           s.registration.active_modules()!=s.registered || !s.registration.modules_ready() || !functions_ready(s.registration.active_functions()) || s.terminal || s.data.poisoned() ||
           !s.data.ranges().empty() || s.backend || s.queue || s.init_attempted) return false;
        if(ps5hip::hip_runtime_contract().vector_compatibility)
            std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0");
        return true;
    });
}
extern "C" bool ps5hip_runtime_ready() noexcept {
    return observe(false, [] {
        if(gate.lifetime != Lifetime::Closing || !state_pointer) return false;
        const auto &s = *state_pointer;
        return s.registered>0 && !s.partial && !s.removed && completion_expected(s.completed) && s.main_finished &&
            !s.terminal && !s.data.poisoned() && s.data.ranges().empty() &&
            !gate.pending && !gate.contexts && !gate.abandoned && !gate.registration_failed &&
            (s.queue ? !s.queue->poisoned() : (!fixed_diagnostic() && !s.init_attempted)) &&
            s.registration.active_modules()==s.registered && s.registration.modules_ready() && functions_ready(s.registration.active_functions());
    });
}
extern "C" bool ps5hip_runtime_finalized() noexcept {
    return observe(false, [] { return final_observation.checked && final_observation.destroyed; });
}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation() noexcept {
    return observe(ps5hip::HipRuntimeObservation{}, [] {
        if(gate.lifetime == Lifetime::Destroyed) return final_observation.snapshot;
        if(!state_pointer) return ps5hip::HipRuntimeObservation{};
        const auto &s = *state_pointer;
        return ps5hip::HipRuntimeObservation{s.registration.active_modules(),s.registration.active_functions(),s.completed,s.init_attempted};
    });
}
