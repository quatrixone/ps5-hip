// Genuine selected code objects and owned host mappings; no GPU execution.
#define PS5HIP_NATIVE_ROUTES_MODEL 1
#define main vector_compatibility_model
#include "native_launch_test.cpp"
#undef main
#include "ps5hip/hip_launch_profile.h"
using namespace ps5hip;

static void qualify(const char* path, bool q4) {
    std::ifstream input(path,std::ios::binary);check(bool(input));
    BundledCodeObject bundle;bundle.bytes={std::istreambuf_iterator<char>(input),{}};
    bundle.object=inspect_code_object(bundle.bytes.data(),bundle.bytes.size());
    check(bundle.object.kernels.size()==1);const auto& kernel=bundle.object.kernels.front();
    const std::vector<unsigned> offsets=q4?std::vector<unsigned>{0,8,16,24,104,112,116,128,132,136,140,152,156,160,164,176,180,184,188}:std::vector<unsigned>{0,8,16,24,32,40,48,56,60};
    const std::vector<unsigned> sizes=q4?std::vector<unsigned>{8,8,8,80,8,4,12,4,4,4,12,4,4,4,12,4,4,4,4}:std::vector<unsigned>{8,8,8,8,8,8,8,4,12};
    check(kernel.arguments.size()==sizes.size() && kernel.kernarg_size==(q4?192u:328u));
    check(kernel.hidden_arguments.size()==(q4?0u:13u));
    const auto route=q4?NativeLaunchRoute::GeometryLds:NativeLaunchRoute::Geometry;
    const std::array<uint32_t,3> threads=q4?std::array<uint32_t,3>{32,4,1}:std::array<uint32_t,3>{256,1,1};
    NativeDataRegistry data;std::array<void*,4> bases{};
    for(auto& base:bases)base=data.allocate(4096);
    std::vector<std::vector<uint8_t>> host(sizes.size());
    std::vector<const void*> addresses;std::vector<ReadableSpan> spans;
    std::vector<uint8_t> expected(kernel.kernarg_size,0);
    auto put=[&](size_t offset,uint64_t value,unsigned size){for(unsigned b=0;b<size;++b)expected.at(offset+b)=uint8_t(value>>(8*b));};
    for(size_t i=0;i<sizes.size();++i) {
        const bool pointer=i<2 || (q4 && (i==2 || i==4));
        check(kernel.arguments[i].offset==offsets[i] && kernel.arguments[i].size==sizes[i] && kernel.arguments[i].kind==(pointer?"global_buffer":"by_value"));
        // Only packing patterns are exercised; the nonfused aggregate stays null.
        if(pointer)put(offsets[i],q4&&i==2?0:reinterpret_cast<uintptr_t>(bases[i==4?2:i])+16,8);
        else if(!(q4&&i==3))for(unsigned b=0;b<sizes[i];++b)expected[offsets[i]+b]=uint8_t(i+b+1);
        host[i].assign(expected.begin()+offsets[i],expected.begin()+offsets[i]+sizes[i]);
        addresses.push_back(host[i].data());spans.push_back({host[i].data(),host[i].size()});
    }
    auto values=capture_launch_arguments(kernel,route,{reinterpret_cast<const uint8_t*>(addresses.data()),addresses.size()*sizeof(void*)},spans,data.ranges());
    for(auto& bytes:host)std::fill(bytes.begin(),bytes.end(),0xee);
    for(unsigned groups:{1u,8u,22u}) {
        auto request=select_hip_launch_profile(kernel,{groups,1,1},threads,0);
        check(request.route==route && request.shape.dimensions==(q4?2u:1u));
        if(!q4) {
            std::fill(expected.begin()+72,expected.end(),0);
            put(72,groups,4);put(76,1,4);put(80,1,4);
            put(84,256,2);put(86,1,2);put(88,1,2);put(136,1,2);
        }
        const auto released=releases;
        {
            auto launch=prepare_native_launch(bundle,kernel,values,data,request);
            check(launch.temporaries.size()==2 && launch.resources.size()==(q4?5u:4u) && allocations.size()==6);
            check(std::equal(expected.begin(),expected.end(),static_cast<const uint8_t*>(launch.temporaries[1]->cpu())));
            const auto image=materialize_code_object(bundle.bytes.data(),bundle.bytes.size());
            check(std::memcmp(image.data(),launch.temporaries[0]->cpu(),image.size())==0);
            for(unsigned i=0;i<4;++i) {
                const auto memory=data.resolve(bases[i],4096).memory;
                check(std::count(launch.resources.begin(),launch.resources.end(),memory)==(i<(q4?3u:2u)?1:0));
            }
            bool limits=false,shape=false,resources=false,dispatch=false;
            const auto& words=launch.commands;
            for(size_t at=0;at<words.size();) {
                const auto h=words[at];const size_t n=((h>>16)&0x3fff)+1;
                check(h>>30==3 && n<words.size()-at);
                if((h&0xff00)==0x7600) {
                    const auto address=words[at+1];
                    if(address==(0xb854-0xb000)/4){check(!limits&&n==2&&words[at+2]==(1u<<22));limits=true;}
                    if(address==(0xb81c-0xb000)/4){check(!shape&&n==4&&std::equal(threads.begin(),threads.end(),words.begin()+at+2));shape=true;}
                    if(address==(0xb848-0xb000)/4){check(!resources&&n==3&&words[at+2]==(q4?0x60af00c3u:0x60af00c1u)&&words[at+3]==(q4?0x10b8cu:0x38cu));resources=true;}
                }
                if((h&0xff00)==0x1500){check(!dispatch&&n==4&&words[at+1]==groups&&words[at+2]==1&&words[at+3]==1);dispatch=true;}
                at+=n+1;
            }
            check(limits&&shape&&resources&&dispatch);
        }
        check(releases==released+2 && allocations.size()==4 && !data.poisoned());
        const auto before=allocate_calls;
        for(unsigned mutation=0;mutation<5;++mutation) {
            auto bad=request;
            if(mutation==0)bad.shape.threads={128,1,1};
            else if(mutation==1)bad.shape.groups[0]=0;
            else if(mutation==2)bad.shape.groups[0]=UINT32_MAX;
            else if(mutation==3)bad.shape.dimensions=3;
            else bad.dynamic_lds_bytes=1;
            reject([&]{prepare_native_launch(bundle,kernel,values,data,bad);});
        }
        auto bad=values;bad[q4?6:8].bytes.pop_back();
        reject([&]{prepare_native_launch(bundle,kernel,bad,data,request);});
        bad=values;bad[0].buffer_bytes--;
        reject([&]{prepare_native_launch(bundle,kernel,bad,data,request);});
        auto altered=bundle;altered.object.kernels[0].rsrc2^=0x100;
        reject([&]{prepare_native_launch(altered,altered.object.kernels[0],values,data,request);});
        check(allocate_calls==before);
        for(unsigned stage:{1u,2u}) {
            fail_allocate_at=allocate_calls+stage;
            reject([&]{prepare_native_launch(bundle,kernel,values,data,request);});
            fail_allocate_at=0;
            check(allocations.size()==4&&!data.poisoned());
        }
    }
    for(auto base:bases)data.free(base);
    check(allocations.empty());
}
int main(int argc,char** argv) {try {
    check(argc==3);qualify(argv[1],false);qualify(argv[2],true);
    std::cout<<"NATIVE_Q4_PREPARE_PASS objects=2 groups=1,8,22 capture=owned null_unretained=true packet=checked allocation_failures=clean gpu_execution=false\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
