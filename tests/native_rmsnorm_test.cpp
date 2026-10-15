// Actual upstream norm image with real host mappings; no queue or GPU submission.
#define PS5HIP_NATIVE_ROUTES_MODEL 1
#define main vector_compatibility_model
#include "native_launch_test.cpp"
#undef main
#include "ps5hip/hip_launch_profile.h"
using namespace ps5hip;
int main(int argc,char** argv) { try {
    check(argc==2);std::ifstream input(argv[1],std::ios::binary);check(bool(input));
    BundledCodeObject bundle;bundle.bytes={std::istreambuf_iterator<char>(input),{}};
    bundle.object=inspect_code_object(bundle.bytes.data(),bundle.bytes.size());
    const std::string name="_ZL12rms_norm_f32ILi1024ELb0ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f";
    auto found=std::find_if(bundle.object.kernels.begin(),bundle.object.kernels.end(),[&](const Kernel& k){return k.name==name;});
    check(found!=bundle.object.kernels.end());const auto& kernel=*found;
    const size_t index=size_t(found-bundle.object.kernels.begin());
    check(bundle.object.kernels.size()==16 && kernel.arguments.size()==26 && kernel.kernarg_size==488);
    const unsigned offsets[]={0,8,16,20,24,32,40,48,56,64,72,80,88,96,108,120,132,144,152,160,168,176,188,200,212,224};
    const unsigned sizes[]={8,8,4,4,4,8,8,8,4,8,8,8,8,12,12,12,12,8,8,8,8,12,12,12,12,4};
    {
        NativeDataRegistry data;std::array<void*,3> bases{};
        for(auto& base:bases)base=data.allocate(17*2048*4);
        std::vector<std::vector<uint8_t>> host(26);std::vector<const void*> addresses;
        std::vector<ReadableSpan> spans;std::vector<uint8_t> expected(488,0);
        auto put=[&](size_t offset,uint64_t value,unsigned size){for(unsigned b=0;b<size;++b)expected[offset+b]=uint8_t(value>>(8*b));};
        for(unsigned i=0;i<26;++i) {
            const bool pointer=i==0 || i==1 || i==9 || i==17;
            check(kernel.arguments[i].offset==offsets[i] && kernel.arguments[i].size==sizes[i] &&
                kernel.arguments[i].kind==(pointer?"global_buffer":"by_value"));
            host[i].resize(sizes[i]);
            // Diagnostic packing patterns; no claim these are numerical RMSNorm inputs.
            if(pointer)put(offsets[i],i<2?uint64_t(bases[i]):0,8);
            else if(sizes[i]<=8)put(offsets[i],100+i,sizes[i]);
            else for(unsigned b=0;b<sizes[i];++b)expected[offsets[i]+b]=uint8_t(i+b);
            std::copy_n(expected.begin()+offsets[i],sizes[i],host[i].begin());
            addresses.push_back(host[i].data());spans.push_back({host[i].data(),host[i].size()});
        }
        const auto values=capture_launch_arguments(kernel,NativeLaunchRoute::ReductionLds,
            {reinterpret_cast<const uint8_t*>(addresses.data()),addresses.size()*sizeof(void*)},spans,data.ranges());
        check(values.size()==26 && values[9].value==0 && values[9].buffer_bytes==0 && values[17].value==0 && values[17].buffer_bytes==0);
        // Captured aggregate data must not alias caller storage.
        for(auto& bytes:host)std::fill(bytes.begin(),bytes.end(),0xee);
        for(unsigned rows:{1u,2u,17u}) {
            auto request=select_hip_launch_profile(kernel,{rows,1,1},{1024,1,1},128);
            std::weak_ptr<NativeMemory> code,arguments;
            const auto released=releases;
            {
                auto launch=prepare_native_launch(bundle,kernel,values,data,request);
                check(launch.temporaries.size()==2 && launch.resources.size()==4 && allocations.size()==5);
                code=launch.temporaries[0];arguments=launch.temporaries[1];
                const auto unused=data.resolve(bases[2],17*2048*4).memory;
                check(std::find(launch.resources.begin(),launch.resources.end(),unused)==launch.resources.end());
                for(unsigned i=0;i<2;++i) {
                    const auto used=data.resolve(bases[i],17*2048*4).memory;
                    check(std::count(launch.resources.begin(),launch.resources.end(),used)==1);
                }
                std::fill(expected.begin()+232,expected.end(),0);
                put(232,rows,4);put(236,1,4);put(240,1,4);put(244,1024,2);put(246,1,2);put(248,1,2);put(296,1,2);put(352,128,4);
                const auto* packed=static_cast<const uint8_t*>(launch.temporaries[1]->cpu());
                check(std::equal(expected.begin(),expected.end(),packed));
                bool resources=false,dispatch=false;
                for(size_t at=0;at<launch.commands.size();) {
                    const auto& words=launch.commands;const auto h=words[at];const size_t n=((h>>16)&0x3fff)+1;
                    check(h>>30==3 && n<words.size()-at);
                    if((h&0xff00)==0x7600 && words[at+1]==(0xb848-0xb000)/4) {
                        check(!resources && n==3 && words[at+2]==0x60af00c3 && words[at+3]==0x1038c);resources=true;
                    }
                    if((h&0xff00)==0x1500) {check(!dispatch && n==4 && words[at+1]==rows && words[at+2]==1 && words[at+3]==1);dispatch=true;}
                    at+=n+1;
                }
                check(resources && dispatch);
                for(const auto& resource:launch.resources)check(!resource->gpu_in_use());
            }
            check(code.expired() && arguments.expired() && releases==released+2 && allocations.size()==3 && !data.poisoned());
            const auto before=allocate_calls;
            for(unsigned mutation=0;mutation<7;++mutation) {
                auto bad=request;
                if(mutation==0)bad.dynamic_lds_bytes=0;else if(mutation==1)bad.dynamic_lds_bytes=65537;
                else if(mutation==2)bad.shape.groups[0]=0;else if(mutation==3)bad.shape.groups[1]=2;
                else if(mutation==4)bad.shape.threads={512,1,1};else if(mutation==5)bad.shape.dimensions=2;
                else bad.shape.groups[0]=UINT32_MAX;
                reject([&]{prepare_native_launch(bundle,kernel,values,data,bad);});
            }
            for(unsigned mutation=0;mutation<4;++mutation) {
                auto altered=bundle;auto& k=altered.object.kernels[index];
                if(mutation==0)k.rsrc1|=1u<<20;else if(mutation==1)k.rsrc2|=1u<<11;
                else if(mutation==2)k.group_bytes=128;else k.private_bytes=4;
                reject([&]{prepare_native_launch(altered,k,values,data,request);});
            }
            auto bad=values;bad[9].buffer_bytes=1;reject([&]{prepare_native_launch(bundle,kernel,bad,data,request);});
            bad=values;bad[13].bytes.pop_back();reject([&]{prepare_native_launch(bundle,kernel,bad,data,request);});
            check(allocate_calls==before);
        }
        for(auto base:bases)data.free(base);
    }
    check(allocations.empty());
    std::cout<<"NATIVE_RMSNORM_PREPARE_PASS rows=1,2,17 arguments=26 kernargs=488 hidden_lds_offset=352 dynamic_lds=128 rsrc2=0x1038c null_unretained=2 ownership_released=true gpu_execution=false\n";
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;} }
