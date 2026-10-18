#pragma once
// Scalar lane translation of ggml CPU AVX2/FMA exp/SiLU and softmax reduction.
// Pinned upstream: ggml-cpu/vec.h, vec.cpp, ops.cpp at
// b76b13379d3b7d67ab0bc424516bb6fd2522bd8e. The upstream exponential
// is adapted from the Arm optimized routine; retain its exact F32 stages.
// MIT License -- Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Define as __attribute__((device,always_inline)) in a device translation unit.
// Use -fno-fast-math -ffp-contract=off; only explicit builtin FMAs may contract.
#ifndef PS5HIP_CPU_MATH_DEVICE
#define PS5HIP_CPU_MATH_DEVICE
#endif
namespace ps5hip_cpu_math {
static_assert(sizeof(unsigned)==4 && sizeof(float)==4, "F32/U32 required");
static inline PS5HIP_CPU_MATH_DEVICE unsigned bits(float value) {
    return __builtin_bit_cast(unsigned, value);
}
static inline PS5HIP_CPU_MATH_DEVICE float from_bits(unsigned value) {
    return __builtin_bit_cast(float, value);
}
static inline PS5HIP_CPU_MATH_DEVICE float exp_f32(float x) {
    const float z=__builtin_fmaf(x,0x1.715476p+0f,0x1.8p23f);
    const float n=z-0x1.8p23f;
    const unsigned e=bits(z)<<23;
    const float k=from_bits(e+0x3f800000u);
    const unsigned g=n<=0.0f?0x82000000u:0u;
    const float s1=from_bits(g+0x7f000000u);
    // Includes masked -Inf -> +0. Avoid evaluating unused invalid polynomial
    // intermediates for infinities; selected output matches the AVX2 bit masks.
    if (__builtin_fabsf(n)>192.0f) return s1*s1;
    const float b=__builtin_fmaf(-n,0x1.7f7d1cp-20f,
                                __builtin_fmaf(-n,0x1.62e4p-1f,x));
    const float u=b*b;
    const float j=__builtin_fmaf(
        __builtin_fmaf(__builtin_fmaf(0x1.0e4020p-7f,b,0x1.573e2ep-5f),u,
                       __builtin_fmaf(0x1.555e66p-3f,b,0x1.fffdb6p-2f)),
        u,0x1.ffffecp-1f*b);
    if (__builtin_fabsf(n)<=126.0f) return __builtin_fmaf(j,k,k);
    const float s2=from_bits(e-g);
    return __builtin_fmaf(s2,j,s2)*s1;
}
static inline PS5HIP_CPU_MATH_DEVICE float silu_f32(float x) {
    const float negative=0.0f-x;
    const float denominator=1.0f+exp_f32(negative);
    return x/denominator;
}
// n must be a positive multiple of8. The qualified attention widths are256/512.
// Inputs are already scaled/masked, with finite maximum and optional -Inf lanes.
// Exact x==out is supported. Other overlapping ranges are caller-rejected.
static inline PS5HIP_CPU_MATH_DEVICE double softmax_exp8(
        const float* x,float* out,unsigned n,float maximum) {
    double sum=0.0;
    for(unsigned i=0;i<n;i+=8) {
        const float a=exp_f32(x[i+0]-maximum),b=exp_f32(x[i+1]-maximum);
        const float c=exp_f32(x[i+2]-maximum),d=exp_f32(x[i+3]-maximum);
        const float e=exp_f32(x[i+4]-maximum),f=exp_f32(x[i+5]-maximum);
        const float g=exp_f32(x[i+6]-maximum),h=exp_f32(x[i+7]-maximum);
        out[i+0]=a;out[i+1]=b;out[i+2]=c;out[i+3]=d;
        out[i+4]=e;out[i+5]=f;out[i+6]=g;out[i+7]=h;
        const float group=((e+a)+(g+c))+((f+b)+(h+d));
        sum+=double(group);
    }
    return sum;
}
static inline PS5HIP_CPU_MATH_DEVICE void softmax_f32(
        const float* x,float* out,unsigned n) {
    float maximum=-__builtin_inff();
    for(unsigned i=0;i<n;++i)maximum=x[i]>maximum?x[i]:maximum;
    const double sum=softmax_exp8(x,out,n,maximum);
    const float inverse=float(1.0/sum);
    for(unsigned i=0;i<n;++i)out[i]=out[i]*inverse;
}
} // namespace ps5hip_cpu_math
