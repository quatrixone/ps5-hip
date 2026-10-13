#include "ps5hip/launch.h"
#include <iostream>
#include <stdexcept>
using namespace ps5hip;
void check(bool value) { if (!value) throw std::runtime_error("planner integration assertion"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ObjectError &) { return; }
    throw std::runtime_error("cross-contract launch accepted");
}
int main() {
    try {
        CodeObject object{}; object.image_size=4096;
        Kernel k{}; k.name="test"; k.entry_address=256; k.kernarg_size=8; k.kernarg_alignment=8;
        k.max_workgroup_size=64; k.wave_size=32; k.properties=0x409; k.rsrc1=0x60af0040; k.rsrc2=0x8c;
        k.arguments={{0,8,"global_buffer"}}; object.kernels={k};
        const GpuRange module{0x200040000,4096}, kernarg{0x4000000000,4096};
        const std::vector<LaunchArgument> args{{0x4003000000,4096}};
        const auto initial=prepare_initial_launch(object,"test",module,kernarg,2,args);
        // Independent literal baseline: initial six-SGPR packet bytes before integration.
        const std::vector<uint32_t> expected{
            0xc0065800,0,0xffffffff,0xff,0,0,10,0x4381,0xc0004200,0,
            0xc0027600,0x216,0xffffffff,0xffffffff,
            0xc0027600,0x219,0,0,
            0xc0047600,0x224,0,0,0,0,
            0xc0027600,0x20c,0x2000401,0,
            0xc0027600,0x212,0x60af0040,0x8c,
            0xc0017600,0x228,0,
            0xc0017600,0x215,0,
            0xc0037600,0x207,64,1,1,
            0xc0067600,0x240,0,0,0,0,0,0x40,
            0xc0031502,2,1,1,0x8005};
        check(encode_initial_dispatch(initial)==expected);
        const auto scratch=plan_buffer_scratch(212,{0x4000100000,9175040});
        const auto calls=plan_buffer_scratch(304,{0x4100100000,13107200});
        for (unsigned route=0; route<4; ++route) {
            object.kernels[0]=k; auto &selected=object.kernels[0];
            if(route==1){selected.private_bytes=212;selected.rsrc2=0x8d;}
            if(route==2)selected.group_bytes=256;
            if(route==3){selected.private_bytes=304;selected.rsrc2=0x91;selected.properties=0x429;}
            auto prepare_initial=[&]{return prepare_initial_launch(object,"test",module,kernarg,2,args);};
            auto prepare_scratch=[&]{return prepare_buffer_scratch_launch(object,"test",module,kernarg,2,args,scratch);};
            auto prepare_lds=[&]{return prepare_lds_launch(object,"test",module,kernarg,2,args,0);};
            auto prepare_calls=[&]{return prepare_flat_call_launch(object,"test",module,kernarg,2,args,calls);};
            if(route==0)check(encode_initial_dispatch(prepare_initial())==expected);else reject(prepare_initial);
            if(route==1)check(encode_buffer_scratch_dispatch(prepare_scratch()).size()==59);else reject(prepare_scratch);
            if(route==2)check(encode_lds_dispatch(prepare_lds()).size()==56);else reject(prepare_lds);
            if(route==3)check(encode_flat_call_dispatch(prepare_calls()).size()==61);else reject(prepare_calls);
            if(route==1 || route==3){selected.group_bytes=256;reject(prepare_scratch);reject(prepare_calls);reject(prepare_lds);}
        }
        // No persistent register state in the planner: initial output stays exact
        // after preparing every other route. This does not simulate a GPU reset.
        check(encode_initial_dispatch(initial)==expected);
        auto bad=initial;bad.user_sgpr_count=8;bad.rsrc2=0x91;reject([&]{encode_initial_dispatch(bad);});
        bad=initial;bad.rsrc2|=0x10000;reject([&]{encode_initial_dispatch(bad);});
        std::cout<<"planner entry separation and initial packet baseline passed\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
