#include "ps5hip/launch.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
using namespace ps5hip;
void check(bool ok) { if (!ok) throw std::runtime_error("geometry assertion failed"); }
template<class F> void reject(F action) {
    try { action(); } catch (const ObjectError &) { return; }
    throw std::runtime_error("invalid geometry accepted");
}
void put(std::vector<uint8_t> &b, size_t at, uint64_t v, unsigned width) {
    for (unsigned n=0;n<width;++n) b.at(at+n)=uint8_t(v>>(n*8));
}
CodeObject synthetic() {
    Kernel k{};k.name="geometry_2d";k.entry_address=256;k.kernarg_size=296;k.kernarg_alignment=8;
    k.wave_size=32;k.max_workgroup_size=64;k.rsrc1=0x60af0042;k.rsrc2=0x138c;k.properties=0x409;
    k.arguments={{0,8,"global_buffer"},{8,8,"global_buffer"},{16,8,"global_buffer"},
                 {24,4,"by_value"},{28,4,"by_value"},{32,4,"by_value"},{36,4,"by_value"}};
    k.hidden_arguments={{40,4,"hidden_block_count_x"},{44,4,"hidden_block_count_y"},{48,4,"hidden_block_count_z"},
        {52,2,"hidden_group_size_x"},{54,2,"hidden_group_size_y"},{56,2,"hidden_group_size_z"},
        {58,2,"hidden_remainder_x"},{60,2,"hidden_remainder_y"},{62,2,"hidden_remainder_z"},
        {80,8,"hidden_global_offset_x"},{88,8,"hidden_global_offset_y"},{96,8,"hidden_global_offset_z"},
        {104,2,"hidden_grid_dims"}};
    auto z=k;z.name="geometry_3d";z.entry_address=512;
    return {"synthetic",16384,{}, {k,z}};
}
void packets(const std::vector<uint32_t> &w,const GeometryLaunch &l) {
    check(w.size()==56);
    check(std::vector<uint32_t>(w.begin(),w.begin()+10)==std::vector<uint32_t>({0xc0065800,0,0xffffffff,0xff,0,0,10,0x4381,0xc0004200,0}));
    size_t at=10;std::map<uint32_t,uint32_t> regs;
    while(at<w.size() && ((w[at]>>8)&255)==0x76) {
        uint32_t header=w.at(at++);check((header>>30)==3 && (header&3)==0);
        unsigned n=(header>>16)&0x3fff;uint32_t address=0xb000+w.at(at++)*4;
        for(unsigned i=0;i<n;++i)check(regs.emplace(address+i*4,w.at(at++)).second);
    }
    check(regs.size()==23);
    check(regs.at(0xb81c)==l.shape.threads[0] && regs.at(0xb820)==l.shape.threads[1] && regs.at(0xb824)==l.shape.threads[2]);
    check(regs.at(0xb84c)==0x138c && regs.at(0xb848)==l.state.rsrc1 && regs.at(0xb8a0)==0);
    check(regs.at(0xb854)==0 && regs.at(0xb858)==UINT32_MAX && regs.at(0xb85c)==UINT32_MAX);
    check(regs.at(0xb830)==uint32_t(l.state.entry_address>>8) && regs.at(0xb834)==uint32_t(l.state.entry_address>>40));
    for(unsigned i=0;i<6;++i)check(regs.at(0xb900+i*4)==l.state.user_sgprs[i]);
    check(at+5==w.size() && w[at]==0xc0031502 && w[at+4]==0x8005);
    for(unsigned i=0;i<3;++i)check(w[at+1+i]==l.shape.groups[i]);
}
void verify(const CodeObject &object) {
    check(object.kernels.size()==2);
    const GpuRange module{0x200040000,16384}, params{0x4000000000,16384};
    for(const auto &k:object.kernels) {
        check(k.name=="geometry_2d" || k.name=="geometry_3d");const bool d3=k.name=="geometry_3d";
        const GeometryShape shape=d3?GeometryShape{{2,3,4},{8,4,2},3}:GeometryShape{{3,2,1},{8,8,1},2};
        const uint32_t x=d3?16:24,y=d3?12:16,z=d3?8:1,total=x*y*z;
        const std::vector<LaunchArgument> args{{0x4000008000,total*4},{0x4000010000,total*64},
            {0x4000040000,total*4},{x,0},{y,0},{z,0},{total*16,0}};
        check(k.rsrc2==0x138c && k.properties==0x409 && k.hidden_arguments.size()==13 && k.arguments.size()==7 && k.kernarg_size==296);
        check(!initial_launch_rejection(k).empty() && geometry_launch_rejection(k).empty());
        auto prepare=[&](const CodeObject &o,const GeometryShape &g,const std::vector<LaunchArgument> &a) {
            return prepare_geometry_launch(o,k.name,module,params,g,a);
        };
        const auto l=prepare(object,shape,args);std::vector<uint8_t> expected(296,0);
        for(unsigned i=0;i<3;++i)put(expected,8*i,args[i].value,8);
        for(unsigned i=3;i<7;++i)put(expected,24+4*(i-3),args[i].value,4);
        for(unsigned i=0;i<3;++i){put(expected,40+4*i,shape.groups[i],4);put(expected,52+2*i,shape.threads[i],2);}
        put(expected,104,shape.dimensions,2);
        check(l.state.kernargs==expected); // all296 bytes, including untouched implicit padding
        check(l.state.user_sgpr_count==6 && l.state.user_sgprs==std::array<uint32_t,6>{0,0,0,0,0,0x40});
        const auto words=encode_geometry_dispatch(l);packets(words,l);
        reject([&]{prepare_initial_launch(object,k.name,module,params,shape.groups[0],args);});
        reject([&]{encode_initial_dispatch(l.state);});
        // The old stream is identical except exact XYZ resource/thread/group words.
        InitialLaunch old=l.state;old.rsrc2=0x8c;old.threads={64,1,1};
        auto one=encode_initial_dispatch(old),modified=words;
        for(size_t at=10;at<modified.size()-5;) {
            auto n=(modified.at(at++)>>16)&0x3fff;auto address=0xb000+modified.at(at++)*4;
            for(unsigned j=0;j<n;++j,++at) {
                if(address+j*4==0xb84c)modified[at]=0x8c;
                if(address+j*4==0xb81c)modified[at]=64;
                if(address+j*4==0xb820 || address+j*4==0xb824)modified[at]=1;
            }
        }
        modified[modified.size()-3]=modified[modified.size()-2]=1;check(modified==one);
        for(unsigned bit=0;bit<32;++bit) {
            auto bad=object;for(auto &q:bad.kernels)if(q.name==k.name)q.rsrc2^=1u<<bit;
            reject([&]{prepare(bad,shape,args);});auto b=l;b.state.rsrc2^=1u<<bit;reject([&]{encode_geometry_dispatch(b);});
        }
        for(unsigned axis=0;axis<3;++axis)for(uint32_t v:{0u,UINT32_MAX}) {
            auto bad=shape;bad.groups[axis]=v;reject([&]{prepare(object,bad,args);});
            auto b=l;b.shape=bad;reject([&]{encode_geometry_dispatch(b);});
        }
        for(auto bad:std::vector<GeometryShape>{{{65536,65536,1},{8,8,1},2},{{2,3,4},{64,1,1},3},{{2,3,1},{8,8,1},3},{{2,3,2},{8,8,1},2},{{2,3,4},{8,8,2},3}})
            reject([&]{prepare(object,bad,args);});
        auto boundary=shape;boundary.groups={UINT32_MAX/64,1,1};check(prepare(object,boundary,args).shape.groups==boundary.groups);
        boundary.groups[0]++;reject([&]{prepare(object,boundary,args);});
        reject([&]{prepare(object,shape,{});});auto badargs=args;badargs[3].value=uint64_t(1)<<32;reject([&]{prepare(object,shape,badargs);});
        badargs=args;badargs[0].value=params.address;reject([&]{prepare(object,shape,badargs);});
        for(unsigned mutation=0;mutation<15;++mutation) {
            auto bad=object;for(auto &q:bad.kernels)if(q.name==k.name) {
                switch(mutation) {
                case 0:q.private_bytes=4;break;case 1:q.group_bytes=256;break;case 2:q.wave_size=64;break;
                case 3:q.preload=1;break;case 4:q.rsrc3=1;break;case 5:q.properties|=0x800;break;
                case 6:q.hidden_arguments.clear();break;case 7:q.hidden_arguments[0].offset++;break;
                case 8:q.hidden_arguments[0].kind="unknown";break;case 9:q.hidden_arguments[1]=q.hidden_arguments[0];break;
                case 10:q.required_workgroup={64,1,1};break;case 11:q.max_workgroup_size=32;break;
                case 12:q.arguments[1].offset=0;break;case 13:q.sgpr_spills=1;break;case 14:q.vgpr_spills=1;break;
                }
            }
            reject([&]{prepare(bad,shape,args);});
        }
        auto matching=object;for(auto &q:matching.kernels)if(q.name==k.name)q.required_workgroup=shape.threads;
        check(prepare(matching,shape,args).state.kernargs==expected);
        auto b=l;b.state.threads[1]++;reject([&]{encode_geometry_dispatch(b);});
        b=l;b.state.groups_x++;reject([&]{encode_geometry_dispatch(b);});
        // Legal packet shapes must still agree with every owned implicit byte.
        for (unsigned axis=0;axis<3;++axis) {
            b=l;b.shape.groups[axis]++;b.state.groups_x=b.shape.groups[0];
            reject([&]{encode_geometry_dispatch(b);});
        }
        b=l;b.shape=d3?GeometryShape{{2,3,1},{8,8,1},2}:GeometryShape{{3,2,1},{8,4,2},3};
        b.state.threads=b.shape.threads;reject([&]{encode_geometry_dispatch(b);});
        for (size_t i=40;i<296;++i) {
            b=l;b.state.kernargs[i]^=1;reject([&]{encode_geometry_dispatch(b);});
        }
        for (size_t size:{size_t(0),size_t(255),size_t(65537)}) {
            b=l;b.state.kernargs.resize(size);reject([&]{encode_geometry_dispatch(b);});
        }
        // Independent packet oracle detects swapped dimensions/resources/initiator.
        for(size_t i: {size_t(27),words.size()-2,words.size()-1}) {
            auto changed=words;changed[i]^=1;bool failed=false;
            try{packets(changed,l);}catch(const std::exception&){failed=true;}check(failed);
        }
    }
}
int main(int argc,char **argv) {
    try {
        if(argc==1)verify(synthetic());
        else {check(argc==2);std::ifstream f(argv[1],std::ios::binary);check(bool(f));
            std::vector<uint8_t>b{std::istreambuf_iterator<char>(f),{}};verify(inspect_code_object(b.data(),b.size()));}
        std::cout<<"Geometry full implicit bytes, independent packet fields, initial preservation and rejection checks passed; no GPU execution\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
