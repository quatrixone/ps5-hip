// Fixed four-kernel client policy; shared runtime owns every HIP ABI/lifetime path.
#include "ps5hip/hip_runtime_control.h"
#include <algorithm>

namespace ps5hip {
namespace {
void require(bool value,const char *message){if(!value)throw ObjectError(message);}
unsigned route_index(const Kernel &kernel) {
    const char *names[]{"routes_integer","routes_lds","routes_spill","routes_geometry"};
    for(unsigned i=0;i<4;++i)if(kernel.name==names[i])return i;
    throw ObjectError("unmeasured routes kernel");
}
void layout(const Kernel &kernel,unsigned route) {
    const unsigned pointers=route==3?3:2;
    const unsigned scalars=route==0?2:route==3?4:0;
    require(kernel.arguments.size()==pointers+scalars,"routes argument count");
    uint32_t offset=0;
    for(unsigned i=0;i<pointers+scalars;++i) {
        const auto &a=kernel.arguments[i];const uint32_t bytes=i<pointers?8:4;
        require(a.offset==offset && a.size==bytes && a.kind==(i<pointers?"global_buffer":"by_value"),
                "routes argument layout");
        offset+=bytes;
    }
    require(kernel.kernarg_alignment==8 && kernel.wave_size==32 && kernel.properties==0x409 &&
            kernel.preload==0 && kernel.rsrc3==0 && kernel.max_workgroup_size>=64,
            "routes entry ABI");
    require(kernel.kernarg_size==(route==3?296:offset) &&
            kernel.hidden_arguments.size()==(route==3?13u:0u),"routes implicit ABI");
    require(kernel.group_bytes==(route==1?256u:0u),"routes static LDS contract");
    require(kernel.private_bytes==(route==2?212u:0u),"routes private allocation contract");
    require(kernel.rsrc2==std::array<uint32_t,4>{0x8c,0x8c,0x8d,0x138c}[route],"routes register entry");
}
}
const HipRuntimeContract &hip_runtime_contract() noexcept {
    static constexpr HipRuntimeContract contract{4,8,false};return contract;
}
NativeLaunchRequest hip_runtime_request(const Kernel &kernel,std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads,size_t shared) {
    const auto route=route_index(kernel);layout(kernel,route);
    require(!shared,"routes dynamic LDS unsupported");
    require(groups==(route==3?std::array<uint32_t,3>{3,2,1}:std::array<uint32_t,3>{2,1,1}) &&
            threads==(route==3?std::array<uint32_t,3>{8,8,1}:std::array<uint32_t,3>{64,1,1}),
            "routes launch shape");
    const NativeLaunchRoute selected[]{NativeLaunchRoute::Initial,NativeLaunchRoute::Lds,
        NativeLaunchRoute::BufferScratch,NativeLaunchRoute::Geometry};
    return {selected[route],{groups,threads,route==3?2u:1u},0};
}
void hip_runtime_validate_arguments(const Kernel &kernel,const NativeLaunchRequest &request,
    const std::vector<LaunchArgument> &values) {
    const auto expected=hip_runtime_request(kernel,request.shape.groups,request.shape.threads,request.dynamic_lds_bytes);
    require(request.route==expected.route && request.shape.dimensions==expected.shape.dimensions,"routes request mismatch");
    const auto route=route_index(kernel);const unsigned pointers=route==3?3:2;
    require(values.size()==kernel.arguments.size(),"routes copied argument count");
    for(unsigned i=0;i<pointers;++i) {
        require(values[i].value>=64 && values[i].value%64==0 && values[i].buffer_bytes==65536-64,
                "routes persistent allocation extent");
        for(unsigned j=0;j<i;++j) {
            const auto low=std::min(values[i].value,values[j].value),high=std::max(values[i].value,values[j].value);
            require(high-low>=65536,"routes persistent allocations overlap");
        }
    }
    for(size_t i=pointers;i<values.size();++i)
        require(!values[i].buffer_bytes && values[i].value<=UINT32_MAX,"routes scalar storage");
    if(route==0)require(values[2].value==128 && (values[3].value==0x11 || values[3].value==0x22),"routes integer scalars");
    if(route==3)require(values[3].value==24 && values[4].value==16 && values[5].value==1 && values[6].value==6144,
                        "routes geometry pitches");
}
} // namespace ps5hip
