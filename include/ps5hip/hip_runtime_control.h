#pragma once
#include "ps5hip/native_launch.h"
#include "ps5hip/submission.h"

namespace ps5hip {
// Fixed diagnostics may constrain counts; reusable operation never depends on an oracle.
// Neither mode advertises unrestricted runtime or device capabilities.
enum class HipRuntimeMode { FixedDiagnostic, ReusableSingleModule };
struct HipRuntimeContract {
    unsigned expected_functions, expected_completions;
    bool vector_compatibility;
    HipRuntimeMode mode = HipRuntimeMode::FixedDiagnostic;
};
const HipRuntimeContract &hip_runtime_contract() noexcept;
NativeLaunchRequest hip_runtime_request(const Kernel &, std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads, size_t shared);
void hip_runtime_validate_arguments(const Kernel &, const NativeLaunchRequest &,
    const std::vector<LaunchArgument> &);
}
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend();
#endif
extern "C" {
void ps5hip_runtime_finish();
bool ps5hip_runtime_terminal();
bool ps5hip_runtime_gpu_unresolved();
// Model observations are not HIP exports. Returned tokens are actual bindings.
const void *ps5hip_runtime_token();
void **ps5hip_runtime_handle();
const void *ps5hip_runtime_token_for(const char *name);
bool ps5hip_runtime_registration_ready() noexcept;
bool ps5hip_runtime_ready() noexcept;
// Reads only process-lifetime POD, safe after the runtime State is destroyed.
bool ps5hip_runtime_finalized() noexcept;
}
