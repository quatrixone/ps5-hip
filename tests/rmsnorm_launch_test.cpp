#include "ps5hip/hip_launch_profile.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
using namespace ps5hip;
static void check(bool ok) { if (!ok) throw std::runtime_error("RMSNorm packet invariant"); }
template<class F> static void reject(F f) { bool caught=false; try { f(); } catch(const ObjectError&) { caught=true; } check(caught); }
int main(int argc,char** argv) { try {
 check(argc==2 || (argc==3 && std::string(argv[2])=="--weighted"));const bool weighted=argc==3; std::ifstream in(argv[1],std::ios::binary); check(bool(in));
 std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in),{}};
 auto object=inspect_code_object(bytes.data(),bytes.size());
 const std::string name=std::string("_ZL12rms_norm_f32ILi1024ELb")+(weighted?"1":"0")+"ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f";
    const uint32_t expected_rsrc1=weighted?0x60af0184u:0x60af00c3u;
 auto found=std::find_if(object.kernels.begin(),object.kernels.end(),[&](const Kernel& k){return k.name==name;});
 check(found!=object.kernels.end()); const auto k=*found;
 check(object.kernels.size()==(weighted?1u:16u) && k.arguments.size()==26 && k.hidden_arguments.size()==14);
 check(k.kernarg_size==488 && k.kernarg_alignment==8 && k.rsrc1==expected_rsrc1 && k.rsrc2==0x38c);
 const unsigned offsets[]={0,8,16,20,24,32,40,48,56,64,72,80,88,96,108,120,132,144,152,160,168,176,188,200,212,224};
 const unsigned sizes[]={8,8,4,4,4,8,8,8,4,8,8,8,8,12,12,12,12,8,8,8,8,12,12,12,12,4};
 std::vector<LaunchArgument> values(26); std::vector<uint8_t> expected(488,0);
 auto put=[&](size_t offset,uint64_t value,unsigned size){for(unsigned b=0;b<size;++b)expected[offset+b]=uint8_t(value>>(8*b));};
 for(unsigned i=0;i<26;++i) {
  const bool pointer=i==0 || i==1 || i==9 || i==17;
  check(k.arguments[i].offset==offsets[i] && k.arguments[i].size==sizes[i] && k.arguments[i].kind==(pointer?"global_buffer":"by_value"));
  if(pointer) { if(i<2 || (weighted && i==9)) values[i]={0x400000000ull+uint64_t(i)*0x100000,17*2048*4,{}}; }
  else if(sizes[i]==12) { values[i].bytes.resize(12); for(unsigned b=0;b<12;++b)expected[offsets[i]+b]=values[i].bytes[b]=uint8_t(i+b); }
  else values[i].value=100+i;
  if(sizes[i]<=8)put(offsets[i],values[i].value,sizes[i]);
 }
 // Packing values are diagnostic patterns, not a numerical kernel invocation.
 const GpuRange module{0x200000000,object.image_size+4096},args{0x300000000,4096};
 unsigned mutations=0;
 for(unsigned rows:{1u,2u,17u}) {
  auto request=select_hip_launch_profile(k,{rows,1,1},{1024,1,1},128);
  check(request.route==NativeLaunchRoute::ReductionLds && request.dynamic_lds_bytes==128);
  auto launch=prepare_reduction_lds_launch(object,name,module,args,rows,values,128);
  check(launch.lds.dynamic_bytes==128 && launch.lds.allocated_bytes==1024 && launch.lds.rsrc2_lds==0x10000);
  std::fill(expected.begin()+232,expected.end(),0);
  put(232,rows,4);put(236,1,4);put(240,1,4);put(244,1024,2);put(246,1,2);put(248,1,2);put(296,1,2);put(352,128,4);
  check(launch.dispatch.kernargs==expected);
  auto words=encode_reduction_lds_dispatch(launch);std::map<unsigned,std::vector<unsigned>> registers;bool dispatched=false;
  for(size_t at=0;at<words.size();) {
   auto h=words[at];size_t n=((h>>16)&0x3fff)+1;check(h>>30==3 && n<words.size()-at);
   if((h&0xff00)==0x7600) { check(n>=2);check(registers.emplace(0xb000+words[at+1]*4,std::vector<unsigned>(words.begin()+at+2,words.begin()+at+n+1)).second); }
   if((h&0xff00)==0x1500) { check(!dispatched && n==4 && words[at+1]==rows && words[at+2]==1 && words[at+3]==1 && words[at+4]==0x8005);dispatched=true; }
   at+=n+1;
  }
  check(dispatched && registers.at(0xb848)==std::vector<unsigned>({expected_rsrc1,0x1038c}));
  check(registers.at(0xb81c)==std::vector<unsigned>({1024,1,1}) && registers.at(0xb854)==std::vector<unsigned>({0x400000}));
  if(weighted && rows==1) {
   for(unsigned vgpr=0;vgpr<64;++vgpr)for(unsigned sgpr=0;sgpr<16;++sgpr) {
    const uint32_t resource=0x60af0000u|(sgpr<<6)|vgpr;
    if((vgpr<=3 && sgpr<=3) || resource==expected_rsrc1)continue;
    auto bad=k;bad.rsrc1=resource;
    reject([&]{select_hip_launch_profile(bad,{1,1,1},{1024,1,1},128);});
    auto packet=launch;packet.dispatch.rsrc1=resource;
    reject([&]{encode_reduction_lds_dispatch(packet);});mutations+=2;
   }
   for(unsigned field=0;field<3;++field) {
    auto bad=k;if(field==0)bad.rsrc2=0x8c;else if(field==1)bad.kernarg_alignment=16;else bad.max_workgroup_size=2048;
    check(!reduction_lds_launch_rejection(bad).empty());
   }
   auto bad=launch;bad.dispatch.rsrc2=0x1008c;reject([&]{encode_reduction_lds_dispatch(bad);});
   bad=launch;bad.dispatch.kernargs.resize(496);reject([&]{encode_reduction_lds_dispatch(bad);});
  }
  for(size_t i=232;i<488;++i) { auto bad=launch;bad.dispatch.kernargs[i]^=1;reject([&]{encode_reduction_lds_dispatch(bad);});++mutations; }
  for(unsigned m=0;m<5;++m) { auto bad=launch;if(m==0)bad.lds.dynamic_bytes=127;else if(m==1)bad.lds.allocated_bytes=512;else if(m==2)bad.lds.rsrc2_lds=0x8000;else if(m==3)bad.lds.static_bytes=128;else bad.dispatch.rsrc2=0x838c;reject([&]{encode_reduction_lds_dispatch(bad);});++mutations; }
  for(unsigned bit:{2u,8u,10u,20u,22u,25u}) {
   auto bad=k;bad.rsrc1^=1u<<bit;
   reject([&]{select_hip_launch_profile(bad,{rows,1,1},{1024,1,1},128);});
   auto changed=object;changed.kernels[size_t(found-object.kernels.begin())]=bad;
   reject([&]{prepare_reduction_lds_launch(changed,name,module,args,rows,values,128);});
   auto packet=launch;packet.dispatch.rsrc1=bad.rsrc1;reject([&]{encode_reduction_lds_dispatch(packet);});mutations+=3;
  }
 }
 for(auto shared:{size_t(0),size_t(65537),size_t(UINT32_MAX)+1})reject([&]{select_hip_launch_profile(k,{1,1,1},{1024,1,1},shared);});
 for(auto threads:{std::array<uint32_t,3>{1025,1,1},{1024,2,1},{512,1,1}})reject([&]{select_hip_launch_profile(k,{1,1,1},threads,128);});
 std::cout<<"RMSNORM_PACKET_PASS rows=1,2,17 arguments=26 implicit=256 dynamic_lds=128 allocated_lds=1024 mutations="<<mutations<<" gpu_execution=false\n";
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }
