// Test-only observer of genuine compiler stubs. No GPU runtime or execution claim.
#include "hip_aggregate_fixture.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static dim3 saved_grid,saved_block;
static unsigned calls;
static const void* expected[6];
static const size_t sizes[6]={8,4,12,16,128,4};
extern "C" void** __hipRegisterFatBinary(const void*) { static void* token;return &token; }
extern "C" void __hipUnregisterFatBinary(void**) {}
extern "C" void __hipRegisterFunction(void**,const void*,...) {}
extern "C" hipError_t __hipPushCallConfiguration(dim3 g,dim3 b,size_t,hipStream_t) {saved_grid=g;saved_block=b;return hipSuccess;}
extern "C" hipError_t __hipPopCallConfiguration(dim3* g,dim3* b,size_t* s,hipStream_t* stream) {*g=saved_grid;*b=saved_block;*s=0;*stream=nullptr;return hipSuccess;}
extern "C" hipError_t hipLaunchKernel(const void*,dim3 g,dim3 b,void** args,size_t shared,hipStream_t stream) {
 if(g.x!=1||g.y!=1||g.z!=1||b.x!=64||b.y!=1||b.z!=1||shared||stream)std::abort();
 const unsigned count=calls?2:6;
 for(unsigned i=0;i<count;++i){
  size_t n=calls?(i?16:8):sizes[i];
  // C++ leaves aggregate padding unspecified across by-value copies. Validate
  // every member byte independently and report the four padding bytes separately.
  for(size_t j=0;j<n;++j){if(!calls&&i==4&&j>=12&&j<16)continue;
   if(static_cast<unsigned char*>(args[i])[j]!=static_cast<const unsigned char*>(expected[i])[j])std::abort();}
  std::printf("arg %u %u %zu ",calls,i,n);
  for(size_t j=0;j<n;++j)std::printf("%02x",static_cast<unsigned char*>(args[i])[j]);
  std::puts("");
 }
 ++calls;return hipSuccess;
}
int main(){
 unsigned char data[164]{};unsigned char* out=data;uint32_t before=0x12345678,after=0xfedcba98;
 uint3 v{0x11223344,0x55667788,0x99aabbcc};Aggregate16 a{0x1020304050607080ull,0xa1b2c3d4,0xe5f6,0x07,0x18};
 Softmax128 p;std::memset(&p,0xa5,sizeof(p));p.nheads=17;p.n_head_log2=4;
 p.ncols=512;p.nrows_x=7;p.nrows_y=8;p.ne00=9;p.ne01=10;p.ne02=11;p.ne03=12;p.nb11=13;p.nb12=14;p.nb13=15;p.ne12=16;p.ne13=18;
 p.scale=0.5f;p.max_bias=1.25f;p.m0=-2.5f;p.m1=3.75f;
 expected[0]=&out;expected[1]=&before;expected[2]=&v;expected[3]=&a;expected[4]=&p;expected[5]=&after;
 launch_aggregate_fixture(out,before,v,a,p,after);
 PointerAggregate16 risk{reinterpret_cast<const uint32_t*>(data),37};expected[0]=&out;expected[1]=&risk;
 launch_pointer_aggregate_fixture(out,risk);
 if(calls!=2)std::abort();std::puts("compiler aggregate stub observation PASS; no GPU execution");
}
