// Offline LDS probes: 64 threads reach every barrier, including both waves.
using u32 = unsigned int;
extern __attribute__((shared)) u32 dynamic_values[];

extern "C" __attribute__((global)) void lds_static(const u32 *input, u32 *output) {
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

// Caller supplies 64 * stride * sizeof(u32) dynamic bytes; stride is nonzero.
extern "C" __attribute__((global)) void lds_dynamic(const u32 *input, u32 *output, u32 stride) {
    const u32 lane = __builtin_amdgcn_workitem_id_x();
    const u32 group = __builtin_amdgcn_workgroup_id_x();
    dynamic_values[lane * stride] = input[group * 64 + lane];
    __builtin_amdgcn_s_barrier();
    for (u32 step = 32; step; step >>= 1) {
        if (lane < step) dynamic_values[lane * stride] += dynamic_values[(lane + step) * stride];
        __builtin_amdgcn_s_barrier();
    }
    if (lane == 0) output[group] = dynamic_values[0];
}

// Static storage must remain distinct from the compiler-placed dynamic array.
extern "C" __attribute__((global)) void lds_mixed(const u32 *input, u32 *output, u32 stride) {
    __attribute__((shared)) u32 tag[64];
    const u32 lane = __builtin_amdgcn_workitem_id_x();
    const u32 group = __builtin_amdgcn_workgroup_id_x();
    tag[lane] = (group * 64 + lane) ^ 0x53a971bdu;
    dynamic_values[lane * stride] = input[group * 64 + lane];
    __builtin_amdgcn_s_barrier();
    for (u32 step = 32; step; step >>= 1) {
        if (lane < step) dynamic_values[lane * stride] += dynamic_values[(lane + step) * stride];
        __builtin_amdgcn_s_barrier();
    }
    if (lane == 0) output[group] = dynamic_values[0] ^ tag[33];
}
