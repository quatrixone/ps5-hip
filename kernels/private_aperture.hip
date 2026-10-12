using u32 = unsigned int;
using u64 = unsigned long long;
extern "C" __attribute__((global)) void private_aperture(u32 *output) {
    u64 base, limit;
    asm volatile("s_mov_b64 %0, src_private_base" : "=s"(base));
    asm volatile("s_mov_b64 %0, src_private_limit" : "=s"(limit));
    const u32 lane = __builtin_amdgcn_workitem_id_x();
    output[lane * 6 + 0] = u32(base);
    output[lane * 6 + 1] = u32(base >> 32);
    output[lane * 6 + 2] = u32(limit);
    output[lane * 6 + 3] = u32(limit >> 32);
    output[lane * 6 + 4] = lane;
    output[lane * 6 + 5] = lane ^ 0x6b18d3a5u;
}
