// Distinct serialized experimental ABI control; no GPU initialization/submission.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/native_data.h"
#include <cstdio>
#include <new>

namespace {
ps5hip::NativeDataRegistry &data() { static ps5hip::NativeDataRegistry value; return value; }
hipError_t last_error = hipSuccess; // This control has one serialized host caller.
hipError_t record(hipError_t result) { last_error = result; return result; }
template<class F> hipError_t checked(F operation) {
    try {
        if (data().poisoned()) throw ps5hip::NativeDataError(ps5hip::NativeDataStatus::Poisoned, "native data registry is poisoned");
        operation(); return record(hipSuccess);
    }
    catch (const ps5hip::NativeDataError &error) {
        std::fprintf(stderr, "hip memory control: %s\n", error.what());
        using S = ps5hip::NativeDataStatus;
        switch (error.status()) {
        case S::InvalidValue: return record(hipErrorInvalidValue);
        case S::InvalidPointer: return record(hipErrorInvalidDevicePointer);
        case S::InUse: return record(hipErrorNotReady);
        case S::Poisoned: case S::CleanupFailed: case S::NativeFailure: return record(hipErrorUnknown);
        }
    }
    catch (const std::bad_alloc &) { return record(hipErrorOutOfMemory); }
    catch (const std::exception &error) { std::fprintf(stderr, "hip memory control: %s\n", error.what()); }
    catch (...) { std::fprintf(stderr, "hip memory control: unknown C++ exception\n"); }
    return record(hipErrorUnknown);
}
}
extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
    if (!pointer) return record(hipErrorInvalidValue);
    // Caller certifies writable output storage, as with ordinary C APIs.
    *pointer = nullptr;
    return checked([&] { if (bytes) *pointer = data().allocate(bytes); });
}
extern "C" hipError_t hipFree(void *pointer) {
    return checked([&] { if (pointer) data().free(pointer); });
}
extern "C" hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind) {
    if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost) return record(hipErrorNotSupported);
    return checked([&] {
        if (!bytes) return;
        if (kind == hipMemcpyHostToDevice) data().copy_to_device(destination, source, bytes);
        else data().copy_from_device(destination, source, bytes);
    });
}
extern "C" hipError_t hipGetLastError() { const auto result = last_error; last_error = hipSuccess; return result; }
extern "C" hipError_t hipPeekAtLastError() { return last_error; }
extern "C" const char *hipGetErrorString(hipError_t error) {
    switch (error) {
    case hipSuccess: return "success";
    case hipErrorInvalidValue: return "invalid value";
    case hipErrorInvalidDevicePointer: return "invalid device pointer";
    case hipErrorOutOfMemory: return "out of memory";
    case hipErrorNotReady: return "allocation is GPU-owned";
    case hipErrorNotSupported: return "operation is not supported by the memory control";
    case hipErrorUnknown: return "native memory or unknown control failure (see diagnostic)";
    default: return "unrecognized error code";
    }
}
extern "C" hipError_t hipLaunchKernel(const void *, dim3, dim3, void **, size_t, hipStream_t) {
    return record(hipErrorNotSupported);
}
