#include "hip_aggregate_fixture.h"
template <class T> __device__ void emit_bytes(unsigned char* out,const T& value) {
 const unsigned char* bytes=reinterpret_cast<const unsigned char*>(&value);
 for(unsigned i=0;i<sizeof(T);++i)out[i]=bytes[i];
}
extern "C" __global__ void aggregate_shapes(unsigned char* out,uint32_t before,uint3 v,Aggregate16 a,Softmax128 p,uint32_t after) {
 if(__builtin_amdgcn_workitem_id_x()!=0)return;
 emit_bytes(out,before);emit_bytes(out+4,v);emit_bytes(out+16,a);emit_bytes(out+32,p);emit_bytes(out+160,after);
}
extern "C" __global__ void pointer_aggregate_shape(unsigned char* out,PointerAggregate16 p) {
 if(__builtin_amdgcn_workitem_id_x()!=0)return;
 // Deliberately do not dereference: metadata cannot establish pointee validity.
 emit_bytes(out,p);
}
extern "C" void launch_aggregate_fixture(unsigned char* out,uint32_t before,uint3 v,Aggregate16 a,Softmax128 p,uint32_t after) {
 aggregate_shapes<<<1,64>>>(out,before,v,a,p,after);
}
extern "C" void launch_pointer_aggregate_fixture(unsigned char* out,PointerAggregate16 p) {
 pointer_aggregate_shape<<<1,64>>>(out,p);
}
