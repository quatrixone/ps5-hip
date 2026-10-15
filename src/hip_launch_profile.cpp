#include "ps5hip/hip_launch_profile.h"
#include "ps5hip/hip_runtime_control.h"
#include "ps5hip/host_launch.h"
#include <limits>

namespace ps5hip {
namespace {
void require(bool value,const char *message) { if(!value) throw ObjectError(message); }
}
NativeLaunchRequest select_hip_launch_profile(const Kernel &kernel,std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads,size_t shared) {
    require(shared<=std::numeric_limits<uint32_t>::max(),"dynamic LDS size overflow");
    NativeLaunchRequest request;
    request.shape.groups=groups;request.shape.threads=threads;
    request.dynamic_lds_bytes=uint32_t(shared);
    const bool reduction=threads==std::array<uint32_t,3>{1024,1,1};
    const bool geometry=!reduction && threads!=std::array<uint32_t,3>{64,1,1};
    // Never drop one resource to force a kernel through a different entry ABI.
    require(!(kernel.private_bytes && (kernel.group_bytes || shared || geometry)),"combined scratch resources unsupported");
    if(reduction) {
        require(shared==4096 && !kernel.private_bytes && !kernel.group_bytes,"unsupported reduction LDS request");
        request.route=NativeLaunchRoute::ReductionLds;
    }
    else if(kernel.private_bytes) request.route=NativeLaunchRoute::BufferScratch;
    else if(geometry && kernel.group_bytes) {
        require(!shared && threads==std::array<uint32_t,3>{8,8,1},"unsupported combined geometry/LDS shape or dynamic size");
        request.route=NativeLaunchRoute::GeometryLds;
    }
    else if(geometry && shared) throw ObjectError("dynamic geometry/LDS unsupported");
    else if(kernel.group_bytes || shared) request.route=NativeLaunchRoute::Lds;
    else if(geometry) request.route=NativeLaunchRoute::Geometry;
    else request.route=NativeLaunchRoute::Initial;
    validate_capture_layout(kernel,request.route);
    require(kernel.max_workgroup_size>=(reduction ? 1024u : 64u) && (kernel.required_workgroup==std::array<uint32_t,3>{} || kernel.required_workgroup==threads),"required workgroup mismatch");
    if(geometry) {
        require((threads==std::array<uint32_t,3>{8,8,1} && groups[2]==1) || threads==std::array<uint32_t,3>{8,4,2},"unsupported geometry block");
        request.shape.dimensions=threads[2]==1?2u:3u;
        uint64_t product=1;
        for(unsigned axis=0;axis<3;++axis) {
            require(groups[axis] && groups[axis]<=UINT32_MAX/threads[axis] && product<=(UINT32_MAX/64)/groups[axis],"geometry extent overflow");
            product*=groups[axis];
        }
    } else {
        require(groups[0] && groups[0]<=UINT32_MAX/(reduction ? 1024u : 64u) && groups[1]==1 && groups[2]==1,"unsupported 1D extent");
        request.shape.dimensions=1;
    }
    if(request.route==NativeLaunchRoute::Lds)
        require(plan_lds(kernel.group_bytes,request.dynamic_lds_bytes).allocated_bytes!=0,"empty LDS allocation");
    if(request.route==NativeLaunchRoute::BufferScratch) (void)buffer_scratch_ring_bytes(kernel.private_bytes);
    return request;
}
const HipRuntimeContract &hip_runtime_contract() noexcept {
    static constexpr HipRuntimeContract contract{0,0,false,HipRuntimeMode::ReusableModules};return contract;
}
NativeLaunchRequest hip_runtime_request(const Kernel &kernel,std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads,size_t shared) {
    return select_hip_launch_profile(kernel,groups,threads,shared);
}
void hip_runtime_validate_arguments(const Kernel &kernel,const NativeLaunchRequest &request,
    const std::vector<LaunchArgument> &values) {
    const auto expected=select_hip_launch_profile(kernel,request.shape.groups,request.shape.threads,request.dynamic_lds_bytes);
    require(expected.route==request.route && expected.shape.dimensions==request.shape.dimensions,"launch profile changed");
    require(values.size()==kernel.arguments.size(),"captured argument count");
    for(size_t i=0;i<values.size();++i) {
        validate_launch_argument(kernel.arguments[i], values[i]);
        if(kernel.arguments[i].kind=="global_buffer") require(values[i].buffer_bytes!=0,"empty live pointer extent");
    }
    // The registry/capture and native preparer validate live extents. Scalar
    // meanings and the kernel's indexing bounds remain the program's contract.
}
}
