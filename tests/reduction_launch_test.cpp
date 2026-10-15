#include "ps5hip/hip_launch_profile.h"
#include "ps5hip/host_launch.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
using namespace ps5hip;
void check(bool x){if(!x)throw std::runtime_error("reduction invariant");}
template<class F>void reject(F f){bool caught=false;try{f();}catch(const ObjectError&){caught=true;}check(caught);}
int main(int argc,char**argv){try{
 check(argc==2);std::ifstream in(argv[1],std::ios::binary);check(bool(in));std::vector<uint8_t>b{std::istreambuf_iterator<char>(in),{}};
 auto object=inspect_code_object(b.data(),b.size());check(object.kernels.size()==1);const auto k=object.kernels[0];
 check(k.kernarg_size==288 && k.hidden_arguments.size()==14 && k.wave_size==32 && k.max_workgroup_size==1024);
 auto request=select_hip_launch_profile(k,{16,1,1},{1024,1,1},4096);check(request.route==NativeLaunchRoute::ReductionLds && request.shape.dimensions==1);
 check(!lds_launch_rejection(k).empty() && !initial_launch_rejection(k).empty());
 unsigned mutations=0;
 for(unsigned m=0;m<12;++m){auto bad=k;
  if(m==0)bad.max_workgroup_size=512;else if(m==1)bad.private_bytes=4;else if(m==2)bad.group_bytes=4;
  else if(m==3)bad.wave_size=64;else if(m==4)bad.rsrc2=0x138c;else if(m==5)bad.preload=1;
  else if(m==6)bad.rsrc3=1;else if(m==7)bad.kernarg_size=280;else if(m==8)bad.sgpr_spills=1;
  else if(m==9)bad.vgpr_spills=1;else if(m==10)bad.required_workgroup={64,1,1};else bad.arguments[3].kind="unsupported";
  reject([&]{select_hip_launch_profile(bad,{16,1,1},{1024,1,1},4096);});++mutations;
 }
 for(size_t i=0;i<k.hidden_arguments.size();++i)for(unsigned m=0;m<4;++m){auto bad=k;
  if(m==0)++bad.hidden_arguments[i].offset;else if(m==1)++bad.hidden_arguments[i].size;
  else if(m==2)bad.hidden_arguments[i].kind="unknown";else bad.hidden_arguments.push_back(bad.hidden_arguments[i]);
  reject([&]{select_hip_launch_profile(bad,{16,1,1},{1024,1,1},4096);});++mutations;
 }
 for(auto size:{0u,65537u,UINT32_MAX}){reject([&]{select_hip_launch_profile(k,{16,1,1},{1024,1,1},size);});++mutations;}
 for(auto groups:{std::array<uint32_t,3>{0,1,1},{UINT32_MAX,1,1},{16,2,1},{16,1,2}}){reject([&]{select_hip_launch_profile(k,groups,{1024,1,1},4096);});++mutations;}
 const GpuRange module{0x200000000,object.image_size+4096},args{0x300000000,4096};
 auto p=prepare_reduction_lds_launch(object,k.name,module,args,16,{{0x400000000,65536},{0x500000000,4096},{0x600000000,64},{16384,0}});
 for(uint32_t bits:{0x3fu,0x3c0u}) {
  auto bad=k;bad.rsrc1|=bits;
  reject([&]{select_hip_launch_profile(bad,{16,1,1},{1024,1,1},4096);});++mutations;
  auto changed=object;changed.kernels[0]=bad;
  reject([&]{prepare_reduction_lds_launch(changed,k.name,module,args,16,{{0x400000000,65536},{0x500000000,4096},{0x600000000,64},{16384,0}});});++mutations;
  auto packet=p;packet.dispatch.rsrc1|=bits;
  reject([&]{encode_reduction_lds_dispatch(packet);});++mutations;
 }
 auto words=encode_reduction_lds_dispatch(p);
 // Independent packet scanner: hard-coded register addresses and expected values.
 std::map<uint32_t,std::vector<uint32_t>> registers;bool dispatched=false;
 for(size_t at=0;at<words.size();){uint32_t header=words[at];check((header>>30)==3);size_t count=((header>>16)&0x3fff)+1;check(at+count<words.size());
  if((header&0xff00)==0x7600){check(count>=2);uint32_t addr=0xb000+words[at+1]*4;check(!registers.count(addr));registers[addr]={words.begin()+at+2,words.begin()+at+count+1};}
  if((header&0xff00)==0x1500){check(!dispatched && count==4);check(words[at+1]==16 && words[at+2]==1 && words[at+3]==1 && words[at+4]==0x8005);dispatched=true;}
  at+=count+1;
 }
 check(dispatched);check(registers.at(0xb81c)==std::vector<uint32_t>({1024,1,1}));check(registers.at(0xb854)==std::vector<uint32_t>({0x400000}));
 check(registers.at(0xb848)==std::vector<uint32_t>({0x60af0080,0x4008c}));
 std::vector<uint8_t> implicit(256,0);auto put=[&](size_t off,uint64_t value,unsigned n){for(unsigned i=0;i<n;++i)implicit[off+i]=uint8_t(value>>(i*8));};
 put(0,16,4);put(4,1,4);put(8,1,4);put(12,1024,2);put(14,1,2);put(16,1,2);put(64,1,2);put(120,4096,4);
 check(std::equal(implicit.begin(),implicit.end(),p.dispatch.kernargs.begin()+32));
 for(size_t i=32;i<288;++i){auto bad=p;bad.dispatch.kernargs[i]^=1;reject([&]{encode_reduction_lds_dispatch(bad);});++mutations;}
 for(unsigned m=0;m<6;++m){auto bad=p;if(m==0)bad.dispatch.threads[0]=64;else if(m==1)bad.dispatch.groups_x=0;else if(m==2)bad.lds.dynamic_bytes=4095;else if(m==3)bad.lds.rsrc2_lds=0;else if(m==4)bad.dispatch.rsrc2=0x4138c;else bad.dispatch.user_sgprs[0]=1;reject([&]{encode_reduction_lds_dispatch(bad);});++mutations;}
 std::cout<<"REDUCTION_PACKET_PASS actual_metadata=true threads=1024 dynamic_lds=4096 resource_limits=4194304 mutations="<<mutations<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
