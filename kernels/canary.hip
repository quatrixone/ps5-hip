// Device-only HIP bring-up fixtures. Compiler builtins deliberately avoid a
// runtime/header dependency; normal HIP host registration is a later gate.
// One-dimensional workgroups contain 64 invocations for these initial fixtures.
using u32 = unsigned int;

extern "C" __attribute__((global)) void indexed_store(u32 *output, u32 count) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u +
                  __builtin_amdgcn_workitem_id_x();
    if (i < count) output[i] = i ^ 0x5a17c9e3u;
}

extern "C" __attribute__((global)) void vector_add(const float *a, const float *b,
                                                 float *output, u32 count) {
    const u32 i = __builtin_amdgcn_workgroup_id_x() * 64u +
                  __builtin_amdgcn_workitem_id_x();
    if (i < count) output[i] = a[i] + b[i];
}
