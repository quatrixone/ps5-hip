#pragma once
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>
// Blocking, row-major FP32 NN subset. Strides are floats; pointers are tracked
// device allocations. Caller keeps buffers live and exclusive through return.
extern "C" hipError_t ps5hipSgemmNN(uint32_t m, uint32_t n, uint32_t k,
    float alpha, const float* a, size_t lda, const float* b, size_t ldb,
    float beta, float* c, size_t ldc);
