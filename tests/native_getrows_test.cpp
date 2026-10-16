// Genuine code-object modeled preparation; no GPU submission or numerical claim.
#define PS5HIP_NATIVE_ROUTES_MODEL 1
#define main vector_compatibility_model
#include "native_launch_test.cpp"
#undef main
#include "ps5hip/hip_launch_profile.h"
using namespace ps5hip;
int main(int argc,char**argv){try{
 check(argc==2);std::ifstream f(argv[1],std::ios::binary);check(bool(f));
 BundledCodeObject b;b.bytes={std::istreambuf_iterator<char>(f),{}};b.object=inspect_code_object(b.bytes.data(),b.bytes.size());
 check(b.object.kernels.size()==1);const auto& k=b.object.kernels[0];check(k.rsrc1==0x60af01c1u&&k.kernarg_size==384);
 const std::array<uint32_t,3> grid{1,4,1},block{256,1,1};
 auto profile=select_hip_launch_profile(k,grid,block,0);check(profile.shape.dimensions==2&&profile.route==NativeLaunchRoute::Geometry);
 auto renamed=k;renamed.name="renamed";check(select_hip_launch_profile(renamed,grid,block,0).shape.dimensions==2);
 NativeDataRegistry data;std::array<void*,3> bases{};for(auto& p:bases)p=data.allocate(16384);
 std::vector<LaunchArgument> values(15);for(unsigned i=0;i<3;++i){values[i].value=reinterpret_cast<uintptr_t>(bases[i]);values[i].buffer_bytes=16384;}
 values[3].value=2048;values[4].value=1;values[5].bytes.resize(12);values[5].bytes[8]=1;
 for(unsigned i=6;i<15;++i)values[i].value=i;
 const auto before=allocate_calls;
 {auto n=prepare_native_launch(b,k,values,data,profile);check(n.temporaries.size()==2&&n.resources.size()==5);
  auto bytes=static_cast<const uint8_t*>(n.temporaries[1]->cpu());check(get(bytes+128,4)==1&&get(bytes+132,4)==4&&get(bytes+136,4)==1&&get(bytes+140,2)==256&&get(bytes+192,2)==2);
  check(std::vector<uint32_t>(n.commands.end()-5,n.commands.end())==std::vector<uint32_t>({0xc0031502,1,4,1,0x8005}));
 }
 check(allocate_calls==before+2&&allocations.size()==3);
 for(unsigned i=0;i<15;++i)for(unsigned mode=0;mode<3;++mode){auto wrong=k;if(mode==0)wrong.arguments[i].offset+=4;else if(mode==1)wrong.arguments[i].size+=4;else wrong.arguments[i].kind=i<3?"by_value":"global_buffer";reject([&]{select_hip_launch_profile(wrong,grid,block,0);});}
 for(unsigned i=0;i<13;++i){auto wrong=k;++wrong.hidden_arguments[i].offset;reject([&]{select_hip_launch_profile(wrong,grid,block,0);});}
 for(unsigned mode=0;mode<13;++mode){auto wrong=k;switch(mode){case 0:wrong.rsrc1=0x60af00c1;break;case 1:wrong.kernarg_size=328;break;case 2:wrong.max_workgroup_size=256;break;case 3:wrong.group_bytes=4;break;case 4:wrong.private_bytes=4;break;case 5:wrong.sgpr_spills=1;break;case 6:wrong.vgpr_spills=1;break;case 7:wrong.preload=1;break;case 8:wrong.rsrc3=1;break;case 9:wrong.properties^=0x20;break;case 10:wrong.wave_size=64;break;case 11:wrong.arguments.pop_back();break;case 12:wrong.hidden_arguments.pop_back();break;}reject([&]{select_hip_launch_profile(wrong,grid,block,0);});}
 for(auto wrong:{std::array<uint32_t,3>{2,4,1},{1,1,1},{1,3,1},{1,5,1},{1,4,2}})reject([&]{select_hip_launch_profile(k,wrong,block,0);});
 for(auto wrong:{std::array<uint32_t,3>{128,1,1},{512,1,1},{8,8,1}})reject([&]{select_hip_launch_profile(k,grid,wrong,0);});
 reject([&]{select_hip_launch_profile(k,grid,block,1);});
 auto geometry=prepare_geometry_launch(b.object,k.name,{0x200000000,0x200000},{0x300000000,4096},profile.shape,values);
 for(unsigned bit=0;bit<32;++bit){auto wrong=k;wrong.rsrc1^=1u<<bit;reject([&]{select_hip_launch_profile(wrong,grid,block,0);});wrong=k;wrong.rsrc2^=1u<<bit;reject([&]{select_hip_launch_profile(wrong,grid,block,0);});auto packet=geometry;packet.state.rsrc1^=1u<<bit;reject([&]{encode_geometry_dispatch(packet);});packet=geometry;packet.state.rsrc2^=1u<<bit;reject([&]{encode_geometry_dispatch(packet);});}
 for(unsigned mode=0;mode<4;++mode){auto wrong=geometry;if(mode==0)wrong.state.rsrc1=0x60af00c1;else if(mode==1)wrong.state.kernargs.resize(328);else if(mode==2)wrong.shape.dimensions=1;else wrong.state.kernargs[192]=1;reject([&]{encode_geometry_dispatch(wrong);});}
 for(unsigned stage:{1u,2u}){fail_allocate_at=allocate_calls+stage;reject([&]{prepare_native_launch(b,k,values,data,profile);});fail_allocate_at=0;check(allocations.size()==3&&!data.poisoned());}
 for(auto p:bases){data.free(p);}check(allocations.empty());std::cout<<"GET_ROWS genuine object modeled launch/ABI/negative controls passed; no GPU execution\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
