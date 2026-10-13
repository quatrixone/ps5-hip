#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

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

namespace {
constexpr unsigned guard=16, poison=0x7fc12345;
void require(bool yes) {if(!yes)throw std::runtime_error("matmul client invariant");}
size_t product(size_t a,size_t b) {
    require(!b || a<=std::numeric_limits<size_t>::max()/b);return a*b;
}
unsigned bits(float value) {unsigned result;std::memcpy(&result,&value,4);return result;}
float poisoned() {float value;std::memcpy(&value,&poison,4);return value;}
float value_a(unsigned row,unsigned inner,unsigned round) {return float(int((row*3+inner*5+round*11)%7)-3);}
float value_b(unsigned inner,unsigned col,unsigned round) {return float(int((inner*7+col*3+round*11)%5)-2);}
void check(hipError_t error) {require(error==hipSuccess);}
}
int main() {
    // Compiler fixture only. The measurement recipe never links or executes HIP.
    struct Shape {unsigned m,n,k;};
    const Shape shapes[]={{8,8,8},{13,21,19},{1,1,17},{17,9,1}};
    try {
        for(const auto shape:shapes) {
            const auto ac=product(shape.m,shape.k),bc=product(shape.k,shape.n),cc=product(shape.m,shape.n);
            const auto ab=product(ac+2*guard,sizeof(float)),bb=product(bc+2*guard,sizeof(float)),cb=product(cc+2*guard,sizeof(float));
            std::vector<float>a(ac+2*guard,poisoned()),b(bc+2*guard,poisoned()),c(cc+2*guard,poisoned()),read(c.size());
            void *da=nullptr,*db=nullptr,*dc=nullptr;
            check(hipMalloc(&da,ab));check(hipMalloc(&db,bb));check(hipMalloc(&dc,cb));
            for(unsigned round=0;round<2;++round) {
                for(unsigned row=0;row<shape.m;++row)for(unsigned inner=0;inner<shape.k;++inner)a[guard+size_t(row)*shape.k+inner]=value_a(row,inner,round);
                for(unsigned inner=0;inner<shape.k;++inner)for(unsigned col=0;col<shape.n;++col)b[guard+size_t(inner)*shape.n+col]=value_b(inner,col,round);
                check(hipMemcpy(da,a.data(),ab,hipMemcpyHostToDevice));check(hipMemcpy(db,b.data(),bb,hipMemcpyHostToDevice));
                check(hipMemcpy(dc,c.data(),cb,hipMemcpyHostToDevice));
                tiled_matmul<<<dim3(shape.n/8+(shape.n%8!=0),shape.m/8+(shape.m%8!=0)),dim3(8,8)>>>(
                    static_cast<float*>(da)+guard,static_cast<float*>(db)+guard,static_cast<float*>(dc)+guard,shape.m,shape.n,shape.k);
                check(hipGetLastError());check(hipDeviceSynchronize());
                std::vector<float>after_a(a.size()),after_b(b.size());
                check(hipMemcpy(after_a.data(),da,ab,hipMemcpyDeviceToHost));check(hipMemcpy(after_b.data(),db,bb,hipMemcpyDeviceToHost));
                require(std::memcmp(after_a.data(),a.data(),ab)==0 && std::memcmp(after_b.data(),b.data(),bb)==0);
                check(hipMemcpy(read.data(),dc,cb,hipMemcpyDeviceToHost));
                for(unsigned row=0;row<shape.m;++row)for(unsigned col=0;col<shape.n;++col) {
                    float expected=0.0f;
                    for(unsigned inner=0;inner<shape.k;++inner)expected+=a[guard+size_t(row)*shape.k+inner]*b[guard+size_t(inner)*shape.n+col];
                    require(bits(read[guard+size_t(row)*shape.n+col])==bits(expected));
                }
                for(unsigned i=0;i<guard;++i)require(bits(read[i])==poison && bits(read[guard+cc+i])==poison);
            }
            check(hipFree(dc));check(hipFree(db));check(hipFree(da));
        }
        std::puts("MATMUL_CLIENT_PASS cases=4 rounds=2 exact_fp32_integer_oracle=true");return 0;
    } catch(const std::exception &error) {
        std::fprintf(stderr,"MATMUL_CLIENT_REJECT %s\n",error.what());return 1;
    }
}
