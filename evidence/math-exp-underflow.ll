target triple = "amdgcn-amd-amdhsa"
declare float @llvm.exp.f32(float)
define amdgpu_kernel void @exp_probe(ptr addrspace(1) %out, float %x) {
  %value = call float @llvm.exp.f32(float %x)
  store float %value, ptr addrspace(1) %out, align 4
  ret void
}
