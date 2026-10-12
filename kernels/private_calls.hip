using u32 = unsigned int;
#define DEV __attribute__((device))
#define INLINE DEV __attribute__((always_inline)) inline
#define CALL DEV __attribute__((noinline))
#define KERNEL extern "C" __attribute__((global))
INLINE u32 local_read(u32 *p, u32 k) { p[k & 31u] += 3; return p[(k + 7) & 31u]; }
INLINE u32 volatile_read(volatile u32 *p, u32 k) { p[k & 31u] += 3; return p[(k + 7) & 31u]; }
CALL u32 call_leaf(u32 *p, u32 k) { p[k & 31u] += 3; return p[k & 31u] + p[(k + 7) & 31u]; }
CALL u32 call_middle(u32 *p, u32 k) {
 u32 inner[32];
 for (u32 j=0;j<32;++j) inner[j]=p[j]+j;
 u32 result = call_leaf(inner,k);
 return result + inner[k&31u] + p[k&31u];
}
CALL u32 volatile_leaf(volatile u32 *p, u32 k) { p[k&31u]+=3; return p[(k+7)&31u]; }
CALL u32 global_leaf(const u32 *p, u32 k) { return p[k]+7; }
KERNEL void indexed_ordinary(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 output[i]=values[input[i]&31u];
}
KERNEL void address_taken_inline(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 output[i]=local_read(values,input[i]);
}
KERNEL void address_taken_volatile_inline(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); volatile u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 output[i]=volatile_read(values,input[i]);
}
KERNEL void nested_private(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 output[i]=call_middle(values,input[i]);
}
KERNEL void called_volatile_private(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); volatile u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 output[i]=volatile_leaf(values,input[i]);
}
KERNEL void ambiguous_volatile(const u32 *input,u32 *output,u32 choose) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); volatile u32 values[32];
 for(u32 j=0;j<32;++j) values[j]=input[i]+j;
 const volatile u32 *p=choose? values : input+i;
 output[i]=p[0];
}
KERNEL void global_call_only(const u32 *input,u32 *output) {
 u32 i=(__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x()); output[i]=global_leaf(input,i);
}
