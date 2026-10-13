#include "ps5hip/hip_launch_profile.h"
#include "ps5hip/hip_runtime_control.h"
#include "ps5hip/host_launch.h"
#include <fstream>
#include <iostream>
#include <iterator>
namespace {
void check(bool yes){if(!yes)throw std::runtime_error("generic profile invariant");}
template<class F>void reject(F f){bool caught=false;try{f();}catch(const ps5hip::ObjectError&){caught=true;}check(caught);}
}
int main(int argc,char **argv) {
 using namespace ps5hip;
 try {
  check(argc>1);unsigned kernels=0;
  for(int file=1;file<argc;++file){std::ifstream input(argv[file],std::ios::binary);check(bool(input));std::vector<uint8_t>b{std::istreambuf_iterator<char>(input),{}};
   const auto obj=inspect_code_object(b.data(),b.size());
   for(const auto &k:obj.kernels){++kernels;const bool geometry=k.rsrc2==0x138c;
    const std::array<uint32_t,3> groups=geometry?std::array<uint32_t,3>{5,3,1}:std::array<uint32_t,3>{7,1,1};
    const std::array<uint32_t,3> threads=geometry?std::array<uint32_t,3>{8,8,1}:std::array<uint32_t,3>{64,1,1};
    auto r=select_hip_launch_profile(k,groups,threads,0);auto renamed=k;renamed.name="arbitrary_other_kernel";check(select_hip_launch_profile(renamed,groups,threads,0).route==r.route);
    for(unsigned mutation=0;mutation<6;++mutation){auto bad=k;
     if(mutation==0)bad.properties=0x429;else if(mutation==1)bad.rsrc2^=0x40;else if(mutation==2)bad.rsrc3=1;
     else if(mutation==3)bad.preload=1;else if(mutation==4)bad.wave_size=64;else{bad.private_bytes=304;bad.properties=0x429;bad.rsrc2=0x91;}
     reject([&]{select_hip_launch_profile(bad,groups,threads,0);});}
    auto bad=k;bad.group_bytes=256;bad.private_bytes=212;reject([&]{select_hip_launch_profile(bad,groups,threads,0);});
    auto shape=groups;shape[0]=0;reject([&]{select_hip_launch_profile(k,shape,threads,0);});shape[0]=UINT32_MAX;reject([&]{select_hip_launch_profile(k,shape,threads,0);});
    reject([&]{select_hip_launch_profile(k,groups,threads,UINT64_MAX);});
    for(size_t i=0;i<k.arguments.size();++i){bad=k;bad.arguments[i].size=16;
     reject([&]{capture_launch_arguments(bad,r.route,{reinterpret_cast<const uint8_t*>(1),k.arguments.size()*8},{},{});});}
    if(!k.hidden_arguments.empty()){bad=k;++bad.hidden_arguments.back().offset;reject([&]{select_hip_launch_profile(bad,groups,threads,0);});}
    if(!geometry && !k.private_bytes && k.hidden_arguments.empty()) {
     auto dynamic=select_hip_launch_profile(k,groups,threads,1280);check(dynamic.route==NativeLaunchRoute::Lds);
     reject([&]{select_hip_launch_profile(k,groups,threads,65537);});
    }
   }
  }
  check(hip_runtime_contract().mode==HipRuntimeMode::ReusableSingleModule);
  std::cout<<"generic profile actual kernels="<<kernels<<" names_independent=true negative_resources_shapes_layouts=pass\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
