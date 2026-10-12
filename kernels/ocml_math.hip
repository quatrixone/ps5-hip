#include <hip/hip_runtime.h>
extern "C" __global__ void fp32_math(float *out, const float *input, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    out[6*i+0] = fmaf(x, 1.25f, -0.5f);
    out[6*i+1] = sqrtf(fabsf(x) + 0.25f);
    out[6*i+2] = sinf(x);
    out[6*i+3] = expf(x);
    out[6*i+4] = logf(fabsf(x) + 0.25f);
    out[6*i+5] = tanhf(x);
  }
}
