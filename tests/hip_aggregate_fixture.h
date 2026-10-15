#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
struct alignas(16) Aggregate16 { uint64_t a; uint32_t b; uint16_t c; uint8_t d; uint8_t e; };
// Field-for-field shape of locked llama.cpp soft_max_params; no pointers.
struct Softmax128 {
 int64_t nheads; uint32_t n_head_log2;
 int64_t ncols, nrows_x, nrows_y, ne00, ne01, ne02, ne03, nb11, nb12, nb13, ne12, ne13;
 float scale, max_bias, m0, m1;
};
struct PointerAggregate16 { const uint32_t* pointer; uint64_t count; };
static_assert(sizeof(uint3)==12 && alignof(uint3)==4);
static_assert(sizeof(Aggregate16)==16 && alignof(Aggregate16)==16);
static_assert(sizeof(Softmax128)==128 && alignof(Softmax128)==8);
static_assert(offsetof(Softmax128,n_head_log2)==8 && offsetof(Softmax128,ncols)==16);
static_assert(offsetof(Softmax128,scale)==112 && offsetof(Softmax128,m1)==124);
static_assert(sizeof(PointerAggregate16)==16 && offsetof(PointerAggregate16,count)==8);
extern "C" void launch_aggregate_fixture(unsigned char*,uint32_t,uint3,Aggregate16,Softmax128,uint32_t);
extern "C" void launch_pointer_aggregate_fixture(unsigned char*,PointerAggregate16);
