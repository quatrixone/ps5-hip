#include <hip/hip_runtime.h>
#include "ps5hip/gemm.h"
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" __global__ __launch_bounds__(64)
void ps5hip_sgemm_nn(const float* a,const float* b,float* c,unsigned m,unsigned n,unsigned k,
                     size_t lda,size_t ldb,size_t ldc,float alpha,float beta) {
    __shared__ float as[8][8],bs[8][8];
    unsigned x=threadIdx.x,y=threadIdx.y;
    unsigned row=blockIdx.y*blockDim.y+y,col=blockIdx.x*blockDim.x+x;
    float sum=0;
    for(unsigned tile=0;tile<k/8+(k%8!=0);++tile) {
        unsigned ax=tile*8+x,by=tile*8+y;
        as[y][x]=(row<m&&ax<k)?a[size_t(row)*lda+ax]:0;
        bs[y][x]=(by<k&&col<n)?b[size_t(by)*ldb+col]:0;
        __syncthreads();
        for(unsigned t=0;t<8;++t)sum+=as[y][t]*bs[t][x];
        __syncthreads();
    }
    if(row<m&&col<n) {
        float result=alpha*sum;
        if(beta!=0)result+=beta*c[size_t(row)*ldc+col];
        c[size_t(row)*ldc+col]=result;
    }
}
namespace {
struct Span {uintptr_t begin,end;};
hipError_t span(const float* pointer,size_t rows,size_t cols,size_t stride,Span& out) {
    if(!pointer || uintptr_t(pointer)%alignof(float) || stride<cols || !rows || !cols)
        return hipErrorInvalidValue;
    const size_t limit=SIZE_MAX/sizeof(float);
    if(cols>limit || (rows-1 && stride>(limit-cols)/(rows-1)))return hipErrorInvalidValue;
    const size_t bytes=((rows-1)*stride+cols)*sizeof(float);
    hipDeviceptr_t base=nullptr;size_t owned=0;
    auto error=hipMemGetAddressRange(&base,&owned,const_cast<float*>(pointer));
    if(error!=hipSuccess)return error;
    const auto address=uintptr_t(pointer),start=uintptr_t(base);
    if(address<start || address-start>owned || bytes>owned-(address-start) || address>UINTPTR_MAX-bytes)
        return hipErrorInvalidValue;
    out={address,address+bytes};return hipSuccess;
}
bool overlap(Span a,Span b){return a.begin<b.end&&b.begin<a.end;}
}
extern "C" hipError_t ps5hipSgemmNN(uint32_t m,uint32_t n,uint32_t k,float alpha,
    const float* a,size_t lda,const float* b,size_t ldb,float beta,float* c,size_t ldc) {
    if(!std::isfinite(alpha)||!std::isfinite(beta))return hipErrorInvalidValue;
    if(!m||!n)return hipSuccess;
    // The existing geometry route bounds total threads to uint32_t.
    const uint64_t gx=n/8+(n%8!=0),gy=m/8+(m%8!=0);
    if(gx*gy>UINT32_MAX/64)return hipErrorInvalidValue;
    Span as{},bs{},cs{};
    auto error=span(c,m,n,ldc,cs);if(error!=hipSuccess)return error;
    if(k&&alpha!=0) {
        error=span(a,m,k,lda,as);if(error!=hipSuccess)return error;
        error=span(b,k,n,ldb,bs);if(error!=hipSuccess)return error;
        if(overlap(as,cs)||overlap(bs,cs))return hipErrorInvalidValue;
    }else{
        // Valid tracked placeholders satisfy capture validation; kernel never reads them.
        k=0;a=b=c;lda=ldb=0;alpha=0;
    }
    ps5hip_sgemm_nn<<<dim3(unsigned(gx),unsigned(gy)),dim3(8,8)>>>(a,b,c,m,n,k,lda,ldb,ldc,alpha,beta);
    error=hipGetLastError();if(error!=hipSuccess)return error;
    return hipDeviceSynchronize();
}
