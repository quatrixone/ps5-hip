#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "ps5hip/hip_runtime_control.h"

extern "C" __global__ __launch_bounds__(64)
void tiled_matmul(const float *a, const float *b, float *c, unsigned m, unsigned n, unsigned k) {
    __shared__ float as[8][8], bs[8][8];
    const unsigned x=threadIdx.x, y=threadIdx.y;
    const unsigned row=blockIdx.y*blockDim.y+y, col=blockIdx.x*blockDim.x+x;
    float sum=0.0f;
    const unsigned tiles=k/8+(k%8!=0);
    for(unsigned tile=0;tile<tiles;++tile) {
        const unsigned ax=tile*8+x, by=tile*8+y;
        as[y][x]=(row<m && ax<k)?a[uint64_t(row)*k+ax]:0.0f;
        bs[y][x]=(by<k && col<n)?b[uint64_t(by)*n+col]:0.0f;
        __syncthreads();
        for(unsigned inner=0;inner<8;++inner) sum+=as[y][inner]*bs[inner][x];
        __syncthreads();
    }
    if(row<m && col<n)c[uint64_t(row)*n+col]=sum;
}

#ifdef PS5HIP_RUNTIME_MODEL
extern "C" void ps5hip_matmul_model_configure(const char*);
extern "C" unsigned ps5hip_matmul_model_live(),ps5hip_matmul_model_submits();
extern "C" const void *ps5hip_runtime_token_for(const char*);
extern "C" void **ps5hip_runtime_handle();
extern "C" void **__hipRegisterFatBinary(const void*);
extern "C" const unsigned char __ps5hip_names_begin[],__ps5hip_names_end[];
extern "C" void __hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
#endif
namespace {
constexpr unsigned guard=16,poison=0x7fc12345;
void require(bool yes) {if(!yes)throw std::runtime_error("matmul client invariant");}
size_t product(size_t a,size_t b) {require(!b||a<=std::numeric_limits<size_t>::max()/b);return a*b;}
unsigned bits(float value) {unsigned result;std::memcpy(&result,&value,4);return result;}
float poisoned() {float value;std::memcpy(&value,&poison,4);return value;}
float value_a(unsigned row,unsigned inner,unsigned round) {return float(int((row*3+inner*5+round*11)%7)-3);}
float value_b(unsigned inner,unsigned col,unsigned round) {return float(int((inner*7+col*3+round*11)%5)-2);}
void check(hipError_t error) {require(error==hipSuccess);}
}
int main(int argc,char **argv) {
    const char *mode=argc==2?argv[1]:"normal";
#ifdef PS5HIP_RUNTIME_MODEL
    ps5hip_matmul_model_configure(mode);
    if(!std::strcmp(mode,"failed-module")) {
        require(ps5hip_runtime_registration_ready()&&!__hipRegisterFatBinary(nullptr));
        require(hipGetLastError()==hipErrorInvalidValue && hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
        require(!ps5hip_runtime_token_for("tiled_matmul"));ps5hip_runtime_finish();
        std::puts("MATMUL_FAILED_MODULE latched=true no_gpu=true");return 0;
    }
#else
    (void)mode;
#endif
    void *da=nullptr,*db=nullptr,*dc=nullptr;
    try {
        require(ps5hip_runtime_registration_ready());
        const auto initial=ps5hip_runtime_observation();
        require(initial.modules==1&&initial.functions==1&&!initial.completed&&!initial.backend_attempted);
        struct Shape {unsigned m,n,k;};
        const Shape shapes[]={{8,8,8},{13,21,19},{1,1,17},{17,9,1}};
        unsigned completed=0,values=0;
        for(unsigned shape_index=0;shape_index<4;++shape_index) {
            const auto shape=shapes[shape_index];
            const auto ac=product(shape.m,shape.k),bc=product(shape.k,shape.n),cc=product(shape.m,shape.n);
            require(ac<=SIZE_MAX-2*guard&&bc<=SIZE_MAX-2*guard&&cc<=SIZE_MAX-2*guard);
            const auto ab=product(ac+2*guard,4),bb=product(bc+2*guard,4),cb=product(cc+2*guard,4);
            std::vector<float>a(ac+2*guard,poisoned()),b(bc+2*guard,poisoned()),c(cc+2*guard,poisoned()),read(c.size()),after_a(a.size()),after_b(b.size());
            check(hipMalloc(&da,ab));check(hipMalloc(&db,bb));check(hipMalloc(&dc,cb));
            for(unsigned round=0;round<2;++round) {
                for(unsigned row=0;row<shape.m;++row)for(unsigned inner=0;inner<shape.k;++inner)a[guard+size_t(row)*shape.k+inner]=value_a(row,inner,round);
                for(unsigned inner=0;inner<shape.k;++inner)for(unsigned col=0;col<shape.n;++col)b[guard+size_t(inner)*shape.n+col]=value_b(inner,col,round);
                check(hipMemcpy(da,a.data(),ab,hipMemcpyHostToDevice));check(hipMemcpy(db,b.data(),bb,hipMemcpyHostToDevice));check(hipMemcpy(dc,c.data(),cb,hipMemcpyHostToDevice));
                dim3 block(8,8);size_t shared=0;
#ifdef PS5HIP_RUNTIME_MODEL
                if(!std::strcmp(mode,"shape"))block.y=4;
                if(!std::strcmp(mode,"dynamic"))shared=64;
#endif
                tiled_matmul<<<dim3((shape.n+7)/8,(shape.m+7)/8),block,shared>>>(static_cast<float*>(da)+guard,static_cast<float*>(db)+guard,static_cast<float*>(dc)+guard,shape.m,shape.n,shape.k);
                const auto error=hipGetLastError();
#ifdef PS5HIP_RUNTIME_MODEL
                if(ps5hip_runtime_terminal()) {
                    require(error!=hipSuccess&&hipDeviceSynchronize()!=hipSuccess);
                    require(hipFree(da)!=hipSuccess&&hipMemcpy(read.data(),dc,cb,hipMemcpyDeviceToHost)!=hipSuccess);
                    require(hipGetLastError()!=hipSuccess&&hipDeviceSynchronize()!=hipSuccess);
                    const auto count=ps5hip_matmul_model_submits();
                    require(hipLaunchKernel(ps5hip_runtime_token_for("tiled_matmul"),dim3(1),dim3(8,8),nullptr,0,nullptr)!=hipSuccess);
                    require(ps5hip_matmul_model_submits()==count);
                    std::printf("MATMUL_TERMINAL live=%u unresolved=%u submits=%u\n",ps5hip_matmul_model_live(),unsigned(ps5hip_runtime_gpu_unresolved()),count);
                    std::fflush(nullptr);
#ifdef PS5HIP_MATMUL_TITLE
                    return 30;
#else
                    std::_Exit(30);
#endif
                }
#endif
                check(error);check(hipDeviceSynchronize());
                check(hipMemcpy(after_a.data(),da,ab,hipMemcpyDeviceToHost));check(hipMemcpy(after_b.data(),db,bb,hipMemcpyDeviceToHost));
                require(std::memcmp(after_a.data(),a.data(),ab)==0&&std::memcmp(after_b.data(),b.data(),bb)==0);
                check(hipMemcpy(read.data(),dc,cb,hipMemcpyDeviceToHost));
                for(unsigned row=0;row<shape.m;++row)for(unsigned col=0;col<shape.n;++col) {
                    // Every integer product and partial sum is exact in binary32 for these shapes.
                    int expected=0;
                    for(unsigned inner=0;inner<shape.k;++inner)expected+=int(a[guard+size_t(row)*shape.k+inner])*int(b[guard+size_t(inner)*shape.n+col]);
                    require(bits(read[guard+size_t(row)*shape.n+col])==bits(float(expected)));++values;
                }
                for(unsigned i=0;i<guard;++i)require(bits(read[i])==poison&&bits(read[guard+cc+i])==poison);
                ++completed;
                const auto observed=ps5hip_runtime_observation();
                require(observed.completed==completed&&observed.modules==1&&observed.functions==1&&observed.backend_attempted);
                std::printf("MATMUL_COMPLETE shape=%u round=%u m=%u n=%u k=%u exact=true guards=true inputs=unchanged\n",shape_index,round,shape.m,shape.n,shape.k);
            }
            check(hipFree(dc));dc=nullptr;check(hipFree(db));db=nullptr;check(hipFree(da));da=nullptr;
        }
#ifdef PS5HIP_RUNTIME_MODEL
        if(!std::strcmp(mode,"late-partial")) {
            const void *token=ps5hip_runtime_token_for("tiled_matmul");require(token);const char *linked="tiled_matmul";
            const auto begin=reinterpret_cast<uintptr_t>(__ps5hip_names_begin),end=reinterpret_cast<uintptr_t>(__ps5hip_names_end),address=reinterpret_cast<uintptr_t>(linked);
            require(address>=begin&&address<=end&&sizeof("tiled_matmul")<=end-address);
            __hipRegisterFunction(ps5hip_runtime_handle(),token,const_cast<char*>(linked),linked,UINT32_MAX,nullptr,nullptr,nullptr,nullptr,nullptr);
            require(hipGetLastError()==hipErrorInvalidDeviceFunction&&hipDeviceSynchronize()==hipErrorInvalidDeviceFunction&&!ps5hip_runtime_token_for(linked));
            ps5hip_runtime_finish();require(!ps5hip_runtime_ready());std::puts("MATMUL_LATE_PARTIAL completed=8 finalized=false");return 0;
        }
#endif
        ps5hip_runtime_finish();require(ps5hip_runtime_ready());
        std::printf("MATMUL_CLIENT_PASS cases=4 rounds=2 dispatches=%u values=%u exact_fp32_integer_oracle=true\n",ps5hip_runtime_observation().completed,values);return 0;
    }catch(...) {
        bool clean=true;
        if(dc){clean=hipFree(dc)==hipSuccess&&clean;dc=nullptr;}
        if(db){clean=hipFree(db)==hipSuccess&&clean;db=nullptr;}
        if(da){clean=hipFree(da)==hipSuccess&&clean;da=nullptr;}
        std::printf("MATMUL_CLIENT_FAILURE clean=%u terminal=%u\n",unsigned(clean),unsigned(ps5hip_runtime_terminal()));
        ps5hip_runtime_finish();return 20;
    }
}
