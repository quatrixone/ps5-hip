#include "llm_cpu_math.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
extern "C" void ggml_vec_silu_f32(int,float*,const float*);
extern "C" double ggml_vec_soft_max_f32(int,float*,const float*,float);
static unsigned seed=917;
static unsigned random_value(){seed=1664525u*seed+1013904223u;return seed;}
static bool equal(float a,float b){return std::memcmp(&a,&b,4)==0;}
static void check(float a,float b,const char* op,unsigned i) {
    if(!equal(a,b)){std::fprintf(stderr,"%s mismatch index=%u actual=%a expected=%a bits=%08x/%08x\n",op,i,double(a),double(b),ps5hip_cpu_math::bits(a),ps5hip_cpu_math::bits(b));std::abort();}
}
static void silu(const std::vector<float>& x) {
    std::vector<float> cpu(x.size()),actual(x.size());
    ggml_vec_silu_f32(int(x.size()),cpu.data(),x.data());
    for(unsigned i=0;i<x.size();++i){actual[i]=ps5hip_cpu_math::silu_f32(x[i]);check(actual[i],cpu[i],"silu",i);}
    auto inplace=x;for(auto& v:inplace)v=ps5hip_cpu_math::silu_f32(v);
    assert(std::memcmp(inplace.data(),cpu.data(),x.size()*4)==0);
}
static void softmax(const std::vector<float>& x) {
    const float maximum=*std::max_element(x.begin(),x.end());
    std::vector<float> cpu(x.size()),actual(x.size());
    double expected=ggml_vec_soft_max_f32(int(x.size()),cpu.data(),x.data(),maximum);
    double sum=ps5hip_cpu_math::softmax_exp8(x.data(),actual.data(),unsigned(x.size()),maximum);
    assert(std::memcmp(&sum,&expected,sizeof(double))==0);
    for(unsigned i=0;i<x.size();++i)check(actual[i],cpu[i],"exp",i);
    const float inverse=float(1.0/expected);for(auto& v:cpu)v*=inverse;
    ps5hip_cpu_math::softmax_f32(x.data(),actual.data(),unsigned(x.size()));
    for(unsigned i=0;i<x.size();++i)check(actual[i],cpu[i],"softmax",i);
    auto inplace=x;ps5hip_cpu_math::softmax_f32(inplace.data(),inplace.data(),unsigned(x.size()));
    assert(std::memcmp(inplace.data(),cpu.data(),x.size()*4)==0);
}
int main() {
    unsigned silu_cases=0,softmax_cases=0;
    for(unsigned trial=0;trial<32;++trial){std::vector<float>x(5632);for(auto& v:x)v=float(int(random_value()%200001)-100000)*0.0005f;silu(x);++silu_cases;}
    const float infinity=std::numeric_limits<float>::infinity();
    std::vector<float> edges={0.f,-0.f,std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),-std::numeric_limits<float>::min(),std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()};
    for(float edge:{-150.f,-133.5f,-103.97f,-88.38f,-87.33654f,87.33654f,88.38f,103.97f,133.5f,150.f}){
        edges.push_back(std::nextafter(edge,-infinity));edges.push_back(edge);edges.push_back(std::nextafter(edge,infinity));
    }
    for(int n=-192;n<=192;++n){float edge=(float(n)+0.5f)*0x1.62e43p-1f;edges.push_back(std::nextafter(edge,-infinity));edges.push_back(edge);edges.push_back(std::nextafter(edge,infinity));}
    while(edges.size()%8)edges.push_back(0.f);silu(edges);++silu_cases;
    for(unsigned width:{256u,512u})for(unsigned trial=0;trial<64;++trial){
        std::vector<float>x(width);for(unsigned i=0;i<width;++i)x[i]=i<=trial%width?float(int(random_value()%20001)-10000)*.01f:-infinity;
        softmax(x);++softmax_cases;
    }
    for(unsigned width:{256u,512u}){
        std::vector<float>x(width,0.f);softmax(x);++softmax_cases;
        for(unsigned i=0;i<width;++i)x[i]=-float(i)*.5f;softmax(x);++softmax_cases;
        x.assign(width,-infinity);x[width-1]=std::numeric_limits<float>::max();softmax(x);++softmax_cases;
    }
    check(ps5hip_cpu_math::exp_f32(-infinity),0.f,"masked",0);
    std::printf("PASS bitwise pinned CPU SiLU cases=%u softmax cases=%u; host F32/F64 only, device denormals/ISA unqualified\n",silu_cases,softmax_cases);
}
