using u32 = unsigned int;
#define CALL extern "C" __attribute__((device,noinline))
#define KERNEL extern "C" __attribute__((global))
#define INDEX (__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x())
CALL u32 volatile_leaf(volatile u32 *p, u32 k) {
    p[k & 31u] += 3u;
    return p[k & 31u] + p[(k + 7u) & 31u];
}
CALL u32 volatile_middle(volatile u32 *p, u32 k) {
    volatile u32 inner[32];
    for (u32 j = 0; j < 32; ++j) inner[j] = p[j] + j;
    const u32 result = volatile_leaf(inner, k);
    return result + inner[k & 31u] + p[k & 31u];
}
KERNEL void volatile_nested_private(const u32 *input, u32 *output) {
    const u32 i = INDEX, x = input[i];
    volatile u32 outer[32];
    for (u32 j = 0; j < 32; ++j) outer[j] = x + j;
    output[i] = volatile_middle(outer, x);
}
// Every invocation owns 32 writable global words. Both runtime branches are valid.
// choose!=0 selects private; choose==0 deliberately mutates one global word.
KERNEL void volatile_mixed_private(u32 *global_values, u32 *output, u32 choose) {
    const u32 i = INDEX, x = global_values[i * 32u];
    volatile u32 outer[32];
    for (u32 j = 0; j < 32; ++j) outer[j] = global_values[i * 32u + j] + j;
    volatile u32 *selected = choose ? outer : global_values + i * 32u;
    const u32 result = volatile_leaf(selected, x);
    output[i] = result + selected[x & 31u];
}
