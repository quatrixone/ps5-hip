using u32 = unsigned int;
#define KERNEL extern "C" __attribute__((global))
#define INDEX (__builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x())
KERNEL void visibility_producer(const u32 *input, u32 *intermediate) {
    const u32 i = INDEX;
    if (i < 257u) intermediate[i] = input[i] * 17u + i * 31u + 0x13579bdfu;
}
KERNEL void visibility_consumer(const u32 *intermediate, u32 *output) {
    const u32 i = INDEX;
    if (i < 257u) {
        const u32 j = (i + 113u) % 257u;
        output[i] = (intermediate[i] ^ 0xa39d71b5u) + intermediate[j] * 3u;
    }
}
