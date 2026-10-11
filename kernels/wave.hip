// Offline wave32 probes; native wave behavior is not yet qualified.
using u32 = unsigned int;

template<unsigned First, unsigned Count>
__attribute__((device, always_inline)) void wave_record(u32 *output) {
    const u32 i = __builtin_amdgcn_workitem_id_x();
    const u32 lane = i & 31u;
    if (lane >= First && lane < First + Count) {
        const u32 value = i + 1u;
        const u32 broadcast = __builtin_amdgcn_readfirstlane(value);
        const u32 peer = ((lane - First) ^ 1u) + First;
        const u32 shuffled = __builtin_amdgcn_ds_bpermute(peer * 4u, value);
        const u32 ballot = __builtin_amdgcn_ballot_w32((value & 1u) != 0);
        u32 sum = value;
        #pragma unroll
        for (u32 step = 1; step < Count; step *= 2) {
            const u32 partner = ((lane - First) ^ step) + First;
            sum += __builtin_amdgcn_ds_bpermute(partner * 4u, sum);
        }
        output[i * 4] = broadcast;
        output[i * 4 + 1] = shuffled;
        output[i * 4 + 2] = ballot;
        output[i * 4 + 3] = sum;
    }
}

extern "C" __attribute__((global)) void wave_full(u32 *output) {
    wave_record<0, 32>(output);
}
extern "C" __attribute__((global)) void wave_partial(u32 *output) {
    wave_record<4, 16>(output);
}