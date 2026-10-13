// Real object contract/capture boundaries; no native allocations or backend calls.
#include "ps5hip/hip_runtime_control.h"
#include "ps5hip/host_launch.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
void check(bool value){if(!value)throw std::runtime_error("routes contract test");}
template<class F> void reject(F action){bool failed=false;try{action();}catch(const ps5hip::ObjectError&){failed=true;}check(failed);}
}
int main(int argc,char **argv) {
    using namespace ps5hip;
    try {
        check(argc==2);std::ifstream in(argv[1],std::ios::binary);check(bool(in));
        const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in),{}};
        const auto object=inspect_code_object(bytes.data(),bytes.size());check(object.target=="gfx1013"&&object.kernels.size()==4);
        const auto &contract=hip_runtime_contract();check(contract.expected_functions==4&&contract.expected_completions==8&&!contract.vector_compatibility);
        for(const auto &kernel:object.kernels) {
            const bool geometry=kernel.name=="routes_geometry";
            const std::array<uint32_t,3> groups=geometry?std::array<uint32_t,3>{3,2,1}:std::array<uint32_t,3>{2,1,1};
            const std::array<uint32_t,3> threads=geometry?std::array<uint32_t,3>{8,8,1}:std::array<uint32_t,3>{64,1,1};
            const auto request=hip_runtime_request(kernel,groups,threads,0);
            validate_capture_layout(kernel,request.route);
            std::vector<uint64_t> slots(kernel.arguments.size(),0);
            std::vector<ReadableSpan> storage;std::vector<uintptr_t> pointers;
            const std::vector<GpuRange> ranges{{0x40000000,65536},{0x50000000,65536},{0x60000000,65536}};
            unsigned pointer=0,scalar=0;
            const uint32_t scalars[]{geometry?24u:128u,geometry?16u:0x11u,1,6144};
            for(size_t i=0;i<slots.size();++i) {
                const auto &a=kernel.arguments[i];
                slots[i]=a.kind=="global_buffer"?ranges.at(pointer++).address+64:scalars[scalar++];
                pointers.push_back(reinterpret_cast<uintptr_t>(&slots[i]));
                storage.push_back({reinterpret_cast<const uint8_t*>(&slots[i]),a.size});
            }
            const ReadableSpan addresses{reinterpret_cast<const uint8_t*>(pointers.data()),pointers.size()*sizeof(uintptr_t)};
            auto values=capture_launch_arguments(kernel,request.route,addresses,storage,ranges);
            hip_runtime_validate_arguments(kernel,request,values);
            const auto snapshot=values;
            for(auto &slot:slots)slot=UINT64_MAX;
            for(size_t i=0;i<values.size();++i)check(values[i].value==snapshot[i].value&&values[i].buffer_bytes==snapshot[i].buffer_bytes);
            hip_runtime_validate_arguments(kernel,request,values);
            auto altered=request;altered.route=NativeLaunchRoute(99);reject([&]{hip_runtime_validate_arguments(kernel,altered,values);});
            altered=request;++altered.shape.dimensions;reject([&]{hip_runtime_validate_arguments(kernel,altered,values);});
            for(unsigned axis=0;axis<3;++axis) {
                auto bad=groups;++bad[axis];reject([&]{hip_runtime_request(kernel,bad,threads,0);});
                bad=threads;++bad[axis];reject([&]{hip_runtime_request(kernel,groups,bad,0);});
            }
            reject([&]{hip_runtime_request(kernel,groups,threads,1);});
            for(size_t i=0;i<values.size();++i) {
                auto bad=values;
                if(kernel.arguments[i].kind=="global_buffer")--bad[i].buffer_bytes;
                else bad[i].value^=0x40000000;
                reject([&]{hip_runtime_validate_arguments(kernel,request,bad);});
            }
            auto alias=values;alias[1]=alias[0];reject([&]{hip_runtime_validate_arguments(kernel,request,alias);});
            for(size_t index=0;index<kernel.arguments.size();++index)for(unsigned mutation=0;mutation<3;++mutation) {
                auto bad=kernel;
                if(mutation==0)bad.arguments[index].size=16;
                else if(mutation==1)++bad.arguments[index].offset;
                else bad.arguments[index].kind="by_reference";
                reject([&]{hip_runtime_request(bad,groups,threads,0);});
                // Invalid layouts must reject before touching even argument-array bytes.
                reject([&]{capture_launch_arguments(bad,request.route,{reinterpret_cast<const uint8_t*>(1),addresses.size},storage,ranges);});
            }
            if(geometry) {
                auto bad=kernel;bad.hidden_arguments.front().offset++;
                reject([&]{capture_launch_arguments(bad,request.route,{reinterpret_cast<const uint8_t*>(1),addresses.size},storage,ranges);});
            }
            if(kernel.name=="routes_integer") {
                auto good=values;good[3].value=0x22;hip_runtime_validate_arguments(kernel,request,good);
            }
        }
        std::cout<<"routes contract/capture: four real kernels, copied scalars, invalid layouts before reads, shapes/extents/aliases rejected; no native execution\n";
    } catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
