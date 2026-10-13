#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "ps5hip/hip_runtime_control.h"
#include "hip_multi_query_client.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

extern "C" void launch_a(const unsigned*,unsigned*,unsigned);
extern "C" void launch_b(const unsigned*,unsigned*,unsigned);
namespace ps5hip_multi_query {
namespace {
constexpr unsigned words=87,guard=7,count=73;
constexpr size_t bytes=words*sizeof(unsigned);
static_assert(sizeof(unsigned)==4 && bytes==348);
std::atomic_flag invoked=ATOMIC_FLAG_INIT;
Report last;
bool require(Report &r,bool good,unsigned error) noexcept {
    if(!good&&!r.error)r.error=error;
    return good;
}
bool hip(Report &r,hipError_t e) noexcept {
    if(e!=hipSuccess){if(!r.hip_error)r.hip_error=unsigned(e);return require(r,false,1);}
    return true;
}
bool observation(Report &r,unsigned completed) noexcept {
    const auto o=ps5hip_runtime_observation();
    return require(r,o.modules==2&&o.functions==2&&
        o.completed==completed&&o.backend_attempted==(completed!=0),2);
}
bool query(Report &r,unsigned *allocation) noexcept {
    for(unsigned offset:{0u,guard}){
        size_t size=713;void *base=nullptr;
        if(!hip(r,hipMemPtrGetInfo(allocation+offset,&size))||!require(r,size==bytes,3))return false;
        ++r.queries;size=713;
        if(!hip(r,hipMemGetAddressRange(&base,&size,allocation+offset))||
            !require(r,base==allocation&&size==bytes,4))return false;
        ++r.queries;
    }
    return true;
}
bool reject(Report &r,void *pointer) noexcept {
    size_t size=713;void *base=reinterpret_cast<void*>(uintptr_t(713));
    const auto info=hipMemPtrGetInfo(pointer,&size);
    if(!require(r,info==hipErrorInvalidValue&&size==713&&hipGetLastError()==info&&
        hipPeekAtLastError()==hipSuccess,5))return false;
    ++r.rejections;
    const auto range=hipMemGetAddressRange(&base,&size,pointer);
    if(!require(r,range==hipErrorNotFound&&size==713&&base==reinterpret_cast<void*>(uintptr_t(713))&&
        hipGetLastError()==range&&hipPeekAtLastError()==hipSuccess,6))return false;
    ++r.rejections;return true;
}
}
const Report &report() noexcept {return last;}
Report run() noexcept {
    Report r;
    if(invoked.test_and_set()){r.error=7;last=r;return r;}
    unsigned *input=nullptr,*output=nullptr;
    std::array<unsigned,words> source{},readback{},poison{};
    for(unsigned i=0;i<words;++i)source[i]=i*37u+9u;
    poison.fill(0xdeadbeefu);
    do {
        if(!require(r,ps5hip_runtime_registration_ready(),2)||!observation(r,0))break;
        std::puts("MULTI_QUERY REGISTERED modules=2 functions=2 completed=0 backend=false");
        if(!hip(r,hipMalloc(reinterpret_cast<void**>(&input),bytes)))break;
        ++r.allocated;
        if(!hip(r,hipMalloc(reinterpret_cast<void**>(&output),bytes)))break;
        ++r.allocated;
        if(!query(r,input)||!query(r,output)||!observation(r,0))break;
        if(!reject(r,input+words)||!reject(r,output+words))break;
        if(!hip(r,hipMemcpy(output,poison.data(),bytes,hipMemcpyHostToDevice)))break;
        for(unsigned step=0;step<3&&!r.error;++step){
            if(!hip(r,hipMemcpy(input,source.data(),bytes,hipMemcpyHostToDevice)))break;
            (step==1?launch_b:launch_a)(input+guard,output+guard,count);
            if(!hip(r,hipGetLastError())||!observation(r,step+1))break;
            ++r.dispatches;
            if(!hip(r,hipMemcpy(readback.data(),output,bytes,hipMemcpyDeviceToHost)))break;
            for(unsigned i=0;i<words;++i){
                const bool active=i>=guard&&i<guard+count;
                const unsigned expected=active?source[i]*3u+(step==1?11u:7u):poison[i];
                if(!require(r,readback[i]==expected,8))break;
                ++r.comparisons;if(active)++r.active_values;else ++r.guards;
            }
            if(r.error||!hip(r,hipMemcpy(readback.data(),input,bytes,hipMemcpyDeviceToHost)))break;
            for(unsigned i=0;i<words;++i){
                if(!require(r,readback[i]==source[i],9))break;
                ++r.comparisons;if(i<guard||i>=guard+count)++r.guards;
            }
            if(r.error||!query(r,input)||!query(r,output))break;
            std::printf("MULTI_QUERY STEP index=%u module=%c count=73 exact=true source=true guards=true\n",step,step==1?'B':'A');
            for(unsigned i=guard;i<guard+count;++i)source[i]+=19;
        }
    }while(false);
    const auto input_address=reinterpret_cast<uintptr_t>(input),output_address=reinterpret_cast<uintptr_t>(output);
    // A terminal transport or uncertain GPU ownership retains every live allocation.
    if(!ps5hip_runtime_terminal()&&!ps5hip_runtime_gpu_unresolved()){
        if(input&&hip(r,hipFree(input))){input=nullptr;++r.freed;}
        if(output&&hip(r,hipFree(output))){output=nullptr;++r.freed;}
    }
    r.cleanup_complete=!input&&!output;
    if(!r.error&&r.cleanup_complete){
        (void)(reject(r,reinterpret_cast<void*>(input_address))&&reject(r,reinterpret_cast<void*>(output_address)));
    }
    r.passed=!r.error&&r.cleanup_complete&&r.dispatches==3&&r.active_values==219&&r.comparisons==522&&
        r.guards==84&&r.queries==32&&r.rejections==8&&r.allocated==2&&r.freed==2;
    if(!r.passed&&!r.error)r.error=10;
    last=r;return r;
}
}
int ps5hip_multi_query_client_main(int,char**) {
    const auto r=ps5hip_multi_query::run();
    if(r.passed){
        std::puts("MULTI_QUERY QUERIES success=32 rejected=8 logical_bytes=348");
        std::puts("MULTI_QUERY CLIENT_PASS dispatches=3 exact_values=219 comparisons=522 guards=84 allocated=2 freed=2");
        return 0;
    }
    std::printf("MULTI_QUERY CLIENT_FAIL error=%u hip_error=%u dispatches=%u allocated=%u freed=%u cleanup=%u\n",
        r.error,r.hip_error,r.dispatches,r.allocated,r.freed,unsigned(r.cleanup_complete));
    return 1;
}
