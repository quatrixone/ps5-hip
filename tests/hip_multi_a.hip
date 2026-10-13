#include <hip/hip_runtime.h>
extern "C" __global__ __launch_bounds__(64) void multi_a(const unsigned *in,unsigned *out,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=in[i]*3u+7u;
}
extern "C" void launch_a(const unsigned *in,unsigned *out,unsigned n) {multi_a<<<(n+63)/64,64>>>(in,out,n);}
