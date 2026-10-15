#include "ps5hip/launch.h"
#include "ps5hip/hip_launch_profile.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
using namespace ps5hip;
void check(bool b){if(!b)throw std::runtime_error("quantized packet invariant");}
template<class F>void reject(F f){try{f();}catch(const ObjectError&){return;}throw std::runtime_error("invalid quantized packet accepted");}
void exercise(const char* path,bool q4){
 std::ifstream in(path,std::ios::binary);check(bool(in));std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in),{}};
 auto o=inspect_code_object(bytes.data(),bytes.size());check(o.kernels.size()==1);const auto& k=o.kernels[0];
 check(k.arguments.size()==(q4?19u:9u)&&k.kernarg_size==(q4?192u:328u));
 const GeometryShape shape{{8,1,1},q4?std::array<uint32_t,3>{32,4,1}:std::array<uint32_t,3>{256,1,1},q4?2u:1u};
 const auto profile=select_hip_launch_profile(k,shape.groups,shape.threads,0);
 check(profile.route==(q4?NativeLaunchRoute::GeometryLds:NativeLaunchRoute::Geometry)&&profile.shape.dimensions==shape.dimensions);
 std::vector<LaunchArgument> args;std::vector<uint8_t> expected(k.kernarg_size);
 for(size_t i=0;i<k.arguments.size();++i){const auto& a=k.arguments[i];LaunchArgument v{};
  if(a.kind=="global_buffer"){v.value=0x400000000ull+i*4096;v.buffer_bytes=4096;if(q4&&i==2)v={};}
  else if(a.size>8){v.bytes.resize(a.size);for(unsigned b=0;b<a.size;++b)v.bytes[b]=uint8_t(i+b);}
  else v.value=37+i;
  for(unsigned b=0;b<a.size;++b)expected[a.offset+b]=v.bytes.empty()?uint8_t(v.value>>(8*b)):v.bytes[b];args.push_back(v);
 }
 if(!q4){const uint64_t values[]={8,1,1,256,1,1,0,0,0,0,0,0,1};for(size_t i=0;i<detail::cov5_geometry.size();++i){auto f=detail::cov5_geometry[i];for(unsigned b=0;b<f.size;++b)expected[72+f.offset+b]=uint8_t(values[i]>>(8*b));}}
 auto prepare=[&](const CodeObject& object,const GeometryShape& s){if(q4)return prepare_geometry_lds_launch(object,k.name,{0x200000000,0x200000},{0x300000000,4096},s,args).geometry;return prepare_geometry_launch(object,k.name,{0x200000000,0x200000},{0x300000000,4096},s,args);};
 auto launch=prepare(o,shape);check(launch.state.kernargs==expected);
 auto encode=[&](const GeometryLaunch& l){return q4?encode_geometry_lds_dispatch({l,plan_lds(384,0)}):encode_geometry_dispatch(l);};
 auto words=encode(launch);std::map<uint32_t,uint32_t> regs;size_t at=10;
 while(at<words.size()-5){auto n=(words.at(at++)>>16)&0x3fff;auto addr=0xb000+words.at(at++)*4;for(unsigned i=0;i<n;++i)check(regs.emplace(addr+4*i,words.at(at++)).second);}
 check(regs.at(0xb84c)==(q4?0x10b8cu:0x38cu)&&regs.at(0xb854)==0x400000);
 check(regs.at(0xb81c)==shape.threads[0]&&regs.at(0xb820)==shape.threads[1]&&regs.at(0xb824)==1);
 check(std::vector<uint32_t>(words.end()-5,words.end())==std::vector<uint32_t>({0xc0031502,8,1,1,0x8005}));
 for(unsigned bit=0;bit<32;++bit){auto bad=launch;bad.state.rsrc2^=1u<<bit;reject([&]{encode(bad);});auto changed=o;changed.kernels[0].rsrc2^=1u<<bit;reject([&]{prepare(changed,shape);});}
 for(unsigned n=0;n<6;++n){auto s=shape;if(n==0)s.groups[0]=0;else if(n==1)s.groups[0]=UINT32_MAX;else if(n==2)s.dimensions=3;else if(n==3)s.threads={128,1,1};else if(n==4)s.groups[2]=2;else s.threads={8,8,1};reject([&]{prepare(o,s);});}
 for(unsigned n=0;n<5;++n){auto bad=launch;if(n==0)bad.state.rsrc1^=1;else if(n==1)bad.state.kernargs.resize(191);else if(n==2)bad.state.threads={64,1,1};else if(n==3)bad.state.rsrc3=1;else {if(q4)bad.state.kernargs.resize(448);else bad.state.kernargs[136]=2;}reject([&]{encode(bad);});}
 reject([&]{select_hip_launch_profile(k,shape.groups,shape.threads,1);});
 if(q4){auto wrong=plan_lds(512,0);reject([&]{encode_geometry_lds_dispatch({launch,wrong});});}
 // Aggregate bytes are deliberately packing patterns, not numerical/fused arguments.
}
int main(int argc,char**argv){try{check(argc==3);exercise(argv[1],false);exercise(argv[2],true);std::cout<<"actual Q8/Q4 packing/resources/negative controls passed; no GPU execution\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
