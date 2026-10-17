#include "llm_layer_numeric_mock.h"
#include <cassert>
#include <cstdio>
#include <cstring>
static float independent(const float* x){float values[1024];for(unsigned i=0;i<1024;++i)values[i]=std::fma(x[i+1024],x[i+1024],x[i]*x[i]);float waves[32];for(unsigned w=0;w<32;++w){for(unsigned step=16;step;step/=2)for(unsigned i=0;i<step;++i)values[w*32+i]+=values[w*32+i+step];waves[w]=values[w*32];}for(unsigned step=16;step;step/=2)for(unsigned i=0;i<step;++i)waves[i]+=waves[i+step];return waves[0];}
int main(){float x[2048];bool differs=false;unsigned seed=17;for(int trial=0;trial<64;++trial){double exact=0;for(auto& v:x){seed=1664525u*seed+1013904223u;v=std::ldexp(float(int(seed%2001)-1000),int(seed%20)-15);exact+=double(v)*v;}float a=layer_numeric::rms_sum_1024(x,2048),b=independent(x);assert(a==b);differs|=a!=float(exact);}assert(differs);bool rejected=false;try{layer_numeric::rms_sum_1024(x,1024);}catch(...){rejected=true;}assert(rejected);puts("PASS fixed1024 F32 RMS tree,FP64 mismatch demonstrated,wrongshape rejected; hardware rsq unmodeled");}
