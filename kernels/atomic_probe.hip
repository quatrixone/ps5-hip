using u32 = unsigned int;
using u64 = unsigned long long;
#define KERNEL extern "C" __attribute__((global))
#define INDEX (__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x())
KERNEL void atomic_ticket32(u32 *counter,u32 *tickets) {
    tickets[INDEX]=__atomic_fetch_add(counter,1u,__ATOMIC_RELAXED);
}
KERNEL void atomic_ticket64(u64 *counter,u64 *tickets) {
    tickets[INDEX]=__atomic_fetch_add(counter,1ull,__ATOMIC_RELAXED);
}
KERNEL void atomic_cas32(u32 *winner,u32 *observed) {
    u32 expected=0;
    __atomic_compare_exchange_n(winner,&expected,INDEX+1,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);
    observed[INDEX]=expected;
}
KERNEL void atomic_cas64(u64 *winner,u64 *observed) {
    u64 expected=0;
    __atomic_compare_exchange_n(winner,&expected,0x13579bdf00000000ull | (u64(INDEX)+1),false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);
    observed[INDEX]=expected;
}

KERNEL void atomic_indirect32(u32 *counter,u32 *tickets,const u32 *offsets) {
    const u32 i=INDEX;
    tickets[i]=__atomic_fetch_add(counter+offsets[i],1u,__ATOMIC_RELAXED);
}
KERNEL void atomic_indirect64(u64 *counter,u64 *tickets,const u32 *offsets) {
    const u32 i=INDEX;
    tickets[i]=__atomic_fetch_add(counter+offsets[i],1ull,__ATOMIC_RELAXED);
}
