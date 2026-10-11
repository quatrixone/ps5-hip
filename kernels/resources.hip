// Offline resource probes. These are not eligible for the initial native launch.
using u32 = unsigned int;

// A bounded reduction without wave/LDS assumptions: one output per invocation.
extern "C" __attribute__((global)) void reduce_serial(const u32 *input, u32 *output) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x();
    u32 sum = 0;
    for (u32 j = 0; j < 16; ++j) sum += input[i * 16 + j];
    output[i] = sum;
}

// Fixed 64-thread workgroup; all threads reach every barrier.
extern "C" __attribute__((global)) void reduce_lds(const u32 *input, u32 *output) {
    __attribute__((shared)) u32 values[64];
    const u32 lane = __builtin_amdgcn_workitem_id_x();
    const u32 group = __builtin_amdgcn_workgroup_id_x();
    values[lane] = input[group * 64 + lane];
    __builtin_amdgcn_s_barrier();
    for (u32 step = 32; step; step >>= 1) {
        if (lane < step) values[lane] += values[lane + step];
        __builtin_amdgcn_s_barrier();
    }
    if (lane == 0) output[group] = values[0];
}

extern "C" __attribute__((global)) void atomic_count(u32 *output) {
    __atomic_fetch_add(output, 1u, __ATOMIC_RELAXED);
}

// Private array probe, not a claim that register spilling has been induced.
extern "C" __attribute__((global)) void private_array(const u32 *input, u32 *output) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x();
    volatile u32 values[32];
    for (u32 j = 0; j < 32; ++j) values[j] = input[i] + j;
    output[i] = values[input[i] & 31u];
}

// Diagnostic contrast with the generic pointers in private_array, not a public API.
extern "C" __attribute__((global)) void private_explicit(const u32 *input, u32 *output) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x();
    volatile u32 values[32];
    auto *private_values = (volatile u32 __attribute__((address_space(5))) *)values;
    for (u32 j = 0; j < 32; ++j) private_values[j] = input[i] + j;
    output[i] = private_values[input[i] & 31u];
}

// Volatile access order keeps all loads live before stores; cap VGPRs to spill.
extern "C" __attribute__((global, amdgpu_num_vgpr(16)))
void register_spill(const volatile u32 *input, volatile u32 *output) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x();
    u32 values[64];
    #pragma unroll
    for (u32 j = 0; j < 64; ++j) values[j] = input[i * 64 + j];
    #pragma unroll
    for (u32 j = 0; j < 64; ++j) output[i * 64 + j] = values[j];
}
