// Host-only companion: the kernel definition and compiler stub remain upstream.
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <stdexcept>
__global__ void block_reduce(int*,std::uint64_t*,int*,std::size_t);
namespace {
void require(bool x){if(!x)throw std::runtime_error("reduction companion validation");}
constexpr std::uint64_t patterns[]{0,0xffffffffffffffffull,0xaaaaaaaaaaaaaaaaull,0x5555555555555555ull,0xffffffff00000000ull,1,0x80000000ull,0x8000000080000001ull};
constexpr int recorded[3][16]={{0,1024,512,512,0,32,32,64,0,1024,512,512,0,32,32,64},{272,272,272,272,272,272,272,272,272,272,272,272,272,272,272,272},{131,6,-339,53,563,74,-287,-158,-384,-128,289,173,-206,194,-40,-170}};
}
extern "C" int ps5hip_reduction_companion_main(){
 static_assert(sizeof(int)==4 && sizeof(std::uint64_t)==8,"fixed reduction ABI");
 void *a=nullptr,*b=nullptr,*c=nullptr;
 try {
  int devices=-1,wave=-1;require(hipInit(0)==hipSuccess && hipGetDeviceCount(&devices)==hipSuccess && devices==1);
  require(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,0)==hipSuccess && wave==32);
  require(ps5hip_runtime_registration_ready());auto before=ps5hip_runtime_observation();require(before.modules==1&&before.functions==1&&!before.completed&&!before.backend_attempted);
  std::puts("REDUCTION_DISCOVERY devices=1 wave=32 backend_attempted=false");
  for(unsigned round=0;round<3;++round){
   constexpr unsigned guard=4,n=16384,m=512,blocks=16;
   const std::size_t logical=round==2?16377:n;
   std::vector<int> input(n+2*guard,0x13579bdf),output(blocks+2*guard,0x2468ace),expected=output,after(input.size()),got(output.size());
   std::vector<std::uint64_t> masks(m+2*guard,0x8badf00dcafefeedull),mask_after(masks.size());
   for(unsigned i=0;i<n;++i)input[guard+i]=round==2?int((i*17+(i/1024)*13)%127)-63:1;
   for(unsigned w=0;w<m;++w)masks[guard+w]=patterns[round==0?(w/32)%8:(w*3+w/32+round)%8];
   for(unsigned block=0;block<blocks;++block){int sum=0;for(unsigned i=block*1024;i<(block+1)*1024&&i<logical;++i)if((masks[guard+i/32]>>(i%32))&1)sum+=input[guard+i];require(sum==recorded[round][block]);expected[guard+block]=sum;}
   require(hipMalloc(&a,input.size()*sizeof(int))==hipSuccess && hipMalloc(&b,masks.size()*sizeof(std::uint64_t))==hipSuccess && hipMalloc(&c,output.size()*sizeof(int))==hipSuccess);
   require(hipMemcpy(a,input.data(),input.size()*sizeof(int),hipMemcpyHostToDevice)==hipSuccess && hipMemcpy(b,masks.data(),masks.size()*sizeof(std::uint64_t),hipMemcpyHostToDevice)==hipSuccess && hipMemcpy(c,output.data(),output.size()*sizeof(int),hipMemcpyHostToDevice)==hipSuccess);
   block_reduce<<<dim3(16),dim3(1024),4096>>>(static_cast<int*>(a)+guard,static_cast<std::uint64_t*>(b)+guard,static_cast<int*>(c)+guard,logical);
   require(hipGetLastError()==hipSuccess && hipDeviceSynchronize()==hipSuccess);
   require(hipMemcpy(after.data(),a,after.size()*sizeof(int),hipMemcpyDeviceToHost)==hipSuccess && hipMemcpy(mask_after.data(),b,mask_after.size()*sizeof(std::uint64_t),hipMemcpyDeviceToHost)==hipSuccess && hipMemcpy(got.data(),c,got.size()*sizeof(int),hipMemcpyDeviceToHost)==hipSuccess);
   require(after==input&&mask_after==masks&&got==expected);
   require(hipFree(a)==hipSuccess);a=nullptr;require(hipFree(b)==hipSuccess);b=nullptr;require(hipFree(c)==hipSuccess);c=nullptr;
   std::printf("REDUCTION_ROUND index=%u logical=%zu outputs=16 input_unchanged=true masks_unchanged=true guards=true\n",round,logical);
  }
  ps5hip_runtime_finish();require(ps5hip_runtime_ready());std::puts("REDUCTION_CLIENT_PASS dispatches=3 exact_values=48 guards=72 allocations=9 frees=9");return 0;
 }catch(...){
  if(ps5hip_runtime_terminal()||ps5hip_runtime_gpu_unresolved()){std::puts("REDUCTION_CLIENT_FAILURE terminal=true retained=true");return 30;}
  bool clean=true;for(void* p:{a,b,c})if(p)clean=hipFree(p)==hipSuccess&&clean;
  std::printf("REDUCTION_CLIENT_FAILURE terminal=false clean=%u\n",unsigned(clean));ps5hip_runtime_finish();return 20;
 }
}
