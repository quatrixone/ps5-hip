#include "ps5hip/launch.h"
#include "ps5hip/host_launch.h"
#include "ps5hip/hip_launch_profile.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
using namespace ps5hip;
void check(bool yes) { if(!yes) throw std::runtime_error("geometry LDS invariant"); }
template<class F> void reject(F f) { try { f(); } catch(const ObjectError &) { return; } throw std::runtime_error("unsupported geometry LDS accepted"); }
CodeObject read(const char *path) {
    std::ifstream in(path,std::ios::binary);check(bool(in));
    std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in),{}};
    return inspect_code_object(bytes.data(),bytes.size());
}
CodeObject synthetic() {
    Kernel k{};k.name="xy_shared";k.entry_address=256;k.kernarg_size=296;k.kernarg_alignment=8;
    k.wave_size=32;k.max_workgroup_size=64;k.rsrc1=0x60af0083;k.rsrc2=0x98c;k.properties=0x409;k.group_bytes=512;
    k.arguments={{0,8,"global_buffer"},{8,8,"global_buffer"},{16,8,"global_buffer"},{24,4,"by_value"},{28,4,"by_value"},{32,4,"by_value"}};
    for(const auto &f:detail::cov5_geometry)k.hidden_arguments.push_back({40+f.offset,f.size,f.kind});
    return {"gfx1013",16384,{}, {k}};
}
void exercise(const CodeObject &object) {
    check(object.kernels.size()==1);const auto &k=object.kernels[0];
    check(k.rsrc2==0x98c && k.group_bytes==512 && k.private_bytes==0 && k.kernarg_size==296);
    check(geometry_lds_launch_rejection(k).empty());
    for(auto route:{NativeLaunchRoute::Initial,NativeLaunchRoute::Lds,NativeLaunchRoute::BufferScratch,NativeLaunchRoute::Geometry})
        reject([&]{validate_capture_layout(k,route);});
    const GeometryShape shape{{3,2,1},{8,8,1},2};
    const GpuRange code{0x200000000,0x200000},storage{0x300000000,4096};
    const std::vector<LaunchArgument> args{{0x400000000,988},{0x400001000,1596},{0x400002000,1092},{13,0},{21,0},{19,0}};
    auto launch=prepare_geometry_lds_launch(object,k.name,code,storage,shape,args);
    auto words=encode_geometry_lds_dispatch(launch);check(words.size()==56);
    check(launch.lds.allocated_bytes==1024 && launch.lds.rsrc2_lds==0x10000 && launch.geometry.state.rsrc2==0x1098c);
    std::map<uint32_t,uint32_t> regs;size_t at=10;
    while(at<words.size()-5) {const auto n=(words.at(at++)>>16)&0x3fff;const auto address=0xb000+words.at(at++)*4;
        for(unsigned i=0;i<n;++i)check(regs.emplace(address+i*4,words.at(at++)).second);}
    check(regs.size()==23 && regs.at(0xb84c)==0x1098c && regs.at(0xb81c)==8 && regs.at(0xb820)==8 && regs.at(0xb824)==1);
    check(std::vector<uint32_t>(words.end()-5,words.end())==std::vector<uint32_t>({0xc0031502,3,2,1,0x8005}));
    const auto &packed=launch.geometry.state.kernargs;
    for(size_t i=0;i<args.size();++i)for(unsigned b=0;b<k.arguments[i].size;++b)
        check(packed.at(k.arguments[i].offset+b)==uint8_t(args[i].value>>(8*b)));
    check(packed.at(40)==3 && packed.at(44)==2 && packed.at(48)==1 && packed.at(52)==8 && packed.at(54)==8 && packed.at(56)==1 && packed.at(104)==2);
    const auto profile=select_hip_launch_profile(k,shape.groups,shape.threads,0);check(profile.route==NativeLaunchRoute::GeometryLds);
    auto renamed=k;renamed.name="unrelated_name";check(select_hip_launch_profile(renamed,shape.groups,shape.threads,0).route==profile.route);
    reject([&]{select_hip_launch_profile(k,shape.groups,shape.threads,512);});
    reject([&]{select_hip_launch_profile(k,{3,2,2},{8,4,2},0);});
    for(unsigned bit=0;bit<32;++bit) {auto bad=launch;bad.geometry.state.rsrc2^=1u<<bit;reject([&]{encode_geometry_lds_dispatch(bad);});
        auto changed=object;changed.kernels[0].rsrc2^=1u<<bit;reject([&]{prepare_geometry_lds_launch(changed,k.name,code,storage,shape,args);});}
    for(unsigned n=0;n<8;++n) {auto bad=launch;
        if(n==0)bad.lds.static_bytes=256;else if(n==1)bad.lds.dynamic_bytes=1;else if(n==2)bad.lds.allocated_bytes=512;
        else if(n==3)bad.lds.rsrc2_lds=0;else if(n==4)bad.geometry.shape.dimensions=3;
        else if(n==5)bad.geometry.state.kernargs.at(54)=4;else if(n==6)bad.geometry.state.user_sgprs[0]=1;
        else bad.geometry.state.kernargs.resize(36);
        reject([&]{encode_geometry_lds_dispatch(bad);});}
    for(unsigned n=0;n<8;++n) {auto bad=object;auto &q=bad.kernels[0];
        if(n==0)q.group_bytes=256;else if(n==1)q.private_bytes=4;else if(n==2)q.preload=1;
        else if(n==3)q.sgpr_spills=1;else if(n==4)q.vgpr_spills=1;else if(n==5)q.hidden_arguments.clear();
        else if(n==6)++q.hidden_arguments.back().offset;else q.properties=0x429;
        reject([&]{prepare_geometry_lds_launch(bad,k.name,code,storage,shape,args);});}
    for(unsigned axis=0;axis<3;++axis)for(uint32_t value:{0u,UINT32_MAX}) {auto bad=shape;bad.groups[axis]=value;
        reject([&]{prepare_geometry_lds_launch(object,k.name,code,storage,bad,args);});}
    reject([&]{prepare_geometry_lds_launch(object,k.name,code,{code.address,4096},shape,args);});
    reject([&]{encode_geometry_dispatch(launch.geometry);});
    reject([&]{encode_lds_dispatch({launch.geometry.state,launch.lds});});
}
int main(int argc,char **argv) {
    try {check(argc==1 || argc==3);exercise(argc==1?synthetic():read(argv[1]));
        if(argc==3) {auto fixed=read(argv[2]);check(fixed.kernels.size()==1 && fixed.kernels[0].kernarg_size==36);
            check(!geometry_lds_launch_rejection(fixed.kernels[0]).empty());}
        std::cout<<"geometry LDS packet, implicit arguments, profile and rejection checks passed; no GPU execution\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
