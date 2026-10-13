#include <hip/hip_runtime.h>
extern "C" __global__ void thread_words(unsigned *output, unsigned count, unsigned salt) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=i*17u+salt;
}
extern "C" void threads_launch(unsigned *output,unsigned count,unsigned salt) {
    thread_words<<<dim3((count+63)/64),dim3(64)>>>(output,count,salt);
}
