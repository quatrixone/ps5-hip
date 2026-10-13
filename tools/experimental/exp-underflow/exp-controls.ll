target triple = "amdgcn-amd-amdhsa"
declare float @llvm.exp.f32(float)
declare <2 x float> @llvm.exp.v2f32(<2 x float>)
declare float @llvm.exp10.f32(float)
define amdgpu_kernel void @exp_scalar(ptr addrspace(1) %out, float %x) {
  %v = call float @llvm.exp.f32(float %x)
  store float %v, ptr addrspace(1) %out
  ret void
}
define amdgpu_kernel void @exp_vector(ptr addrspace(1) %out, <2 x float> %x) {
  %v = call <2 x float> @llvm.exp.v2f32(<2 x float> %x)
  store <2 x float> %v, ptr addrspace(1) %out
  ret void
}
define amdgpu_kernel void @exp_fast(ptr addrspace(1) %out, float %x) {
  %v = call fast float @llvm.exp.f32(float %x)
  store float %v, ptr addrspace(1) %out
  ret void
}
define amdgpu_kernel void @exp10_scalar(ptr addrspace(1) %out, float %x) {
  %v = call float @llvm.exp10.f32(float %x)
  store float %v, ptr addrspace(1) %out
  ret void
}
define amdgpu_kernel void @unrelated_control(ptr addrspace(1) %out, float %x) {
  %v = fmul float %x, 1.25
  store float %v, ptr addrspace(1) %out
  ret void
}
