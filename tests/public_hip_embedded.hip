#include <hip/hip_runtime.h>
extern "C" __global__ void thread_words(unsigned* output,unsigned count,unsigned salt) {
 const unsigned i=__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x();
 if(i<count)output[i]=i*17u+salt;
}
extern "C" void launch_public_embedded(unsigned* output,unsigned count,unsigned salt) {
 thread_words<<<(count+63)/64,64>>>(output,count,salt);
}
