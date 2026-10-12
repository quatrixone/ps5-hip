// Ordinary client compiled against the pinned full upstream header package.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#ifdef PS5HIP_MEMORY_MOCK_CLIENT
extern "C" void ps5hip_memory_mock_failure(int);
extern "C" unsigned ps5hip_memory_mock_live();
#endif
namespace {
void check(bool yes) { if (!yes) { std::fprintf(stderr, "HIP memory client assertion failed\n"); std::abort(); } }
}
int main() {
    check(hipMalloc(nullptr, 1) == hipErrorInvalidValue);
    check(hipPeekAtLastError() == hipErrorInvalidValue && hipPeekAtLastError() == hipErrorInvalidValue);
    check(std::strstr(hipGetErrorString(hipPeekAtLastError()), "invalid") != nullptr);
    check(hipGetLastError() == hipErrorInvalidValue && hipPeekAtLastError() == hipSuccess);
    void *empty = reinterpret_cast<void *>(1);
    check(hipMalloc(&empty, 0) == hipSuccess && empty == nullptr);
    check(hipFree(nullptr) == hipSuccess);
    check(hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToDevice) == hipSuccess);
    check(hipMemcpy(nullptr, nullptr, 0, hipMemcpyDeviceToHost) == hipSuccess);
    constexpr size_t count = 73;
    float input[count], output[count]{};
    for (size_t i = 0; i < count; ++i) input[i] = float(i) * 0.25f;
    float *buffer = nullptr;
    check(hipMalloc(&buffer, sizeof(input)) == hipSuccess && buffer != nullptr); // upstream typed overload
    check(hipMemcpy(buffer, input, sizeof(input), hipMemcpyHostToDevice) == hipSuccess);
    std::memset(input, 0, sizeof(input));
    check(hipMemcpy(output, buffer + 1, sizeof(output) - sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i + 1 < count; ++i) check(output[i] == float(i + 1) * 0.25f);
    check(hipMemcpy(output, buffer + count - 1, 2 * sizeof(float), hipMemcpyDeviceToHost) == hipErrorInvalidValue);
    check(hipMemcpy(buffer, buffer + 1, 1, hipMemcpyHostToDevice) == hipErrorInvalidValue);
    check(hipMemcpy(output, buffer, 1, hipMemcpyDeviceToDevice) == hipErrorNotSupported);
    check(hipLaunchKernel(nullptr, dim3(1), dim3(64), nullptr) == hipErrorNotSupported);
    check(hipGetLastError() == hipErrorNotSupported && hipGetLastError() == hipSuccess);
    check(hipFree(buffer + 1) == hipErrorInvalidDevicePointer);
#ifdef PS5HIP_MEMORY_MOCK_CLIENT
    // Native setup failures must not escape a C ABI or poison an existing buffer
    // when the acquired backing has been completely unwound.
    for (int failure : {1, 2}) {
        ps5hip_memory_mock_failure(failure);
        void *failed = reinterpret_cast<void *>(1);
        check(hipMalloc(&failed, 16) == hipErrorUnknown && failed == nullptr);
        ps5hip_memory_mock_failure(0);
        check(hipMemcpy(output, buffer, sizeof(output), hipMemcpyDeviceToHost) == hipSuccess);
    }
#endif
    check(hipFree(buffer) == hipSuccess);
    check(hipFree(buffer) == hipErrorInvalidDevicePointer);
    check(hipMemcpy(output, buffer, 1, hipMemcpyDeviceToHost) == hipErrorInvalidDevicePointer);
#ifdef PS5HIP_MEMORY_MOCK_CLIENT
    check(ps5hip_memory_mock_live() == 0);
    // Retained-close errors poison the ABI registry; never return fake success.
    check(hipMalloc(&buffer, sizeof(input)) == hipSuccess);
    ps5hip_memory_mock_failure(3);
    check(hipFree(buffer) == hipErrorUnknown && ps5hip_memory_mock_live() == 1);
    check(hipMemcpy(output, buffer, 1, hipMemcpyDeviceToHost) == hipErrorUnknown);
    empty = reinterpret_cast<void *>(1);
    check(hipMalloc(&empty, 0) == hipErrorUnknown && empty == nullptr);
    check(hipFree(nullptr) == hipErrorUnknown);
    check(hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToDevice) == hipErrorUnknown);
    check(hipMemcpy(nullptr, nullptr, 0, hipMemcpyDeviceToHost) == hipErrorUnknown);
    ps5hip_memory_mock_failure(0); // shutdown may clean non-GPU-owned backing
#endif
    std::puts("HIP_MEMORY_CLIENT_PASS actual_native_memory=true gpu_calls=0 launch=unsupported hardware_validated=false");
}
