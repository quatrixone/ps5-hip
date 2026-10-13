#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef PS5HIP_GENERIC_TITLE
#include "ps5hip/hip_runtime_control.h"
#endif

extern "C" __global__ __launch_bounds__(64)
void affine_words(const unsigned *input,unsigned *output,unsigned n,uint64_t salt) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned i=__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x();
    if(i<n) output[i]=(input[i]+unsigned(salt))^unsigned(salt>>32);
#endif
}
extern "C" __global__ __launch_bounds__(64)
void block_sum(const unsigned *input,unsigned *output,unsigned n) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned values[64];
    const unsigned lane=__builtin_amdgcn_workitem_id_x(),group=__builtin_amdgcn_workgroup_id_x();
    const unsigned i=group*64+lane;values[lane]=i<n?input[i]:0;
    __builtin_amdgcn_s_barrier();
    for(unsigned stride=32;stride;stride>>=1) {
        if(lane<stride)values[lane]+=values[lane+stride];
        __builtin_amdgcn_s_barrier();
    }
    if(!lane)output[group]=values[0];
#endif
}
#ifdef PS5HIP_GENERIC_EXTRA_KERNEL
extern "C" __global__ __launch_bounds__(64)
void deliberately_unused(unsigned *output,unsigned value) {
#if defined(__HIP_DEVICE_COMPILE__)
    if(!__builtin_amdgcn_workitem_id_x())*output=value;
#endif
}
#endif
#ifndef PS5HIP_GENERIC_TITLE
extern "C" void ps5hip_runtime_finish();
extern "C" bool ps5hip_runtime_ready(),ps5hip_runtime_registration_ready(),ps5hip_runtime_terminal(),ps5hip_runtime_gpu_unresolved();
#endif
#ifdef PS5HIP_RUNTIME_MODEL
extern "C" void ps5hip_generic_model_configure(const char*);
extern "C" unsigned ps5hip_generic_model_live(),ps5hip_generic_model_submits();
extern "C" const void *ps5hip_runtime_token_for(const char*);
extern "C" void **ps5hip_runtime_handle();
extern "C" void **__hipRegisterFatBinary(const void*);
extern "C" const unsigned char __ps5hip_names_begin[],__ps5hip_names_end[];
extern "C" void __hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
#endif
namespace {
void require(bool yes) {if(!yes)throw std::runtime_error("generic client invariant");}
constexpr unsigned poison=0xa5a5a5a5;
unsigned source(unsigned i,unsigned iteration) {return (i*37u+iteration*1231u)^0x951dac37u;}
}
#ifdef PS5HIP_MEMORY_FILL_TITLE
#include "hip_memory_fill_client.h"
#else
int main(int argc,char **argv) {
    const char *mode=argc==2?argv[1]:"seven";
#ifdef PS5HIP_RUNTIME_MODEL
    ps5hip_generic_model_configure(mode);
    if(!std::strcmp(mode,"partial") || !std::strcmp(mode,"second-module")) {
        require(hipGetLastError()!=hipSuccess);
        require(hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
        require(!ps5hip_runtime_registration_ready());
        require(!ps5hip_runtime_token_for("affine_words"));
        ps5hip_runtime_finish();std::puts("GENERIC_PARTIAL no_gpu=true");return 0;
    }
    if(!std::strcmp(mode,"failed-module")) {
        require(ps5hip_runtime_registration_ready());
        require(!__hipRegisterFatBinary(nullptr));
        require(hipGetLastError()==hipErrorInvalidValue);
        require(hipDeviceSynchronize()==hipErrorInvalidDeviceFunction && !ps5hip_runtime_token_for("affine_words"));
        ps5hip_runtime_finish();std::puts("GENERIC_FAILED_MODULE latched=true no_gpu=true");return 0;
    }
#endif
    require(ps5hip_runtime_registration_ready());
    void *a=nullptr,*b=nullptr;
    try {
#ifdef PS5HIP_GENERIC_TITLE
        const unsigned counts[]{0,1,3,7};
        for(unsigned sequence=0;sequence<4;++sequence) {
            const unsigned count=counts[sequence];
            const auto before=ps5hip_runtime_observation();
            require(before.modules==1 && before.functions==2);
            if(sequence==0)require(before.completed==0 && !before.backend_attempted);
#else
        const unsigned sequence=0;
        const unsigned count=!std::strcmp(mode,"zero")?0:!std::strcmp(mode,"one")?1:!std::strcmp(mode,"three")?3:7;
#endif
        for(unsigned iteration=0;iteration<count;++iteration) {
            const unsigned seed=iteration+sequence*17;
            const unsigned sizes[]{1,63,64,65,127,193,257};const unsigned n=sizes[iteration];
            const unsigned guard=3+iteration,groups=(n+63)/64;
            std::vector<unsigned> input(n+2*guard,poison),output(n+2*guard,poison),observed(output.size()),input_after(input.size());
            for(unsigned i=0;i<n;++i)input[guard+i]=source(i,seed);
            const size_t bytes=input.size()*sizeof(unsigned);
            require(hipMalloc(&a,bytes)==hipSuccess && hipMalloc(&b,bytes)==hipSuccess);
            require(hipMemcpy(a,input.data(),bytes,hipMemcpyHostToDevice)==hipSuccess && hipMemcpy(b,output.data(),bytes,hipMemcpyHostToDevice)==hipSuccess);
            auto *in=static_cast<unsigned*>(a)+guard,*out=static_cast<unsigned*>(b)+guard;
            const uint64_t salt=0x13579bdf00000000ull+17+seed;
            if(iteration%2==0)affine_words<<<groups,64>>>(in,out,n,salt);
            else block_sum<<<groups,64>>>(in,out,n);
            const auto error=hipGetLastError();
#ifdef PS5HIP_RUNTIME_MODEL
            if(ps5hip_runtime_terminal()) {
                require(error!=hipSuccess && hipDeviceSynchronize()!=hipSuccess);
                require(hipFree(a)!=hipSuccess && hipMemcpy(observed.data(),b,bytes,hipMemcpyDeviceToHost)!=hipSuccess);
                require(hipGetLastError()!=hipSuccess && hipDeviceSynchronize()!=hipSuccess);
                std::printf("GENERIC_TERMINAL live=%u unresolved=%u submits=%u\n",ps5hip_generic_model_live(),unsigned(ps5hip_runtime_gpu_unresolved()),ps5hip_generic_model_submits());
                std::fflush(nullptr);
#ifdef PS5HIP_GENERIC_TITLE
                return 30;
#else
                std::_Exit(30);
#endif
            }
#endif
            require(error==hipSuccess && hipDeviceSynchronize()==hipSuccess);
            if(iteration%2==0)for(unsigned i=0;i<n;++i)output[guard+i]=(input[guard+i]+unsigned(salt))^unsigned(salt>>32);
            else for(unsigned g=0;g<groups;++g) {
                unsigned sum=0;for(unsigned lane=0;lane<64 && g*64+lane<n;++lane)sum+=input[guard+g*64+lane];output[guard+g]=sum;
            }
            require(hipMemcpy(observed.data(),b,bytes,hipMemcpyDeviceToHost)==hipSuccess && hipMemcpy(input_after.data(),a,bytes,hipMemcpyDeviceToHost)==hipSuccess);
            require(observed==output && input_after==input);
            require(hipFree(a)==hipSuccess);a=nullptr;require(hipFree(b)==hipSuccess);b=nullptr;
        }
#ifdef PS5HIP_GENERIC_TITLE
            const auto after=ps5hip_runtime_observation();
            require(after.completed-before.completed==count);
            std::printf("GENERIC_SEQUENCE index=%u requested=%u observed=%u completed=%u prebackend=%u\n",sequence,count,after.completed-before.completed,after.completed,unsigned(!before.backend_attempted));
        }
#endif
#ifdef PS5HIP_RUNTIME_MODEL
        if(!std::strcmp(mode,"late-partial")) {
            const void *old=ps5hip_runtime_token_for("affine_words");require(old);
            const char *linked="affine_words";
            const auto begin=reinterpret_cast<uintptr_t>(__ps5hip_names_begin),end=reinterpret_cast<uintptr_t>(__ps5hip_names_end),address=reinterpret_cast<uintptr_t>(linked);
            require(address>=begin && address<=end && sizeof("affine_words")<=end-address);
            __hipRegisterFunction(ps5hip_runtime_handle(),old,const_cast<char*>(linked),linked,UINT32_MAX,nullptr,nullptr,nullptr,nullptr,nullptr);
            require(hipGetLastError()==hipErrorInvalidDeviceFunction && hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
            require(!ps5hip_runtime_token_for(linked));
            ps5hip_runtime_finish();require(!ps5hip_runtime_ready());
#ifdef PS5HIP_GENERIC_TITLE
            std::puts("GENERIC_LATE_PARTIAL completed=11 finalized=false");
#else
            std::puts("GENERIC_LATE_PARTIAL completed=7 finalized=false");
#endif
            return 0;
        }
#endif
        ps5hip_runtime_finish();require(ps5hip_runtime_ready());
#ifdef PS5HIP_GENERIC_TITLE
        const auto observed=ps5hip_runtime_observation();require(observed.completed==11);
        std::printf("GENERIC_CLIENT_PASS launches=%u sequences=0,1,3,7 variable_sizes=true full_guards=true inputs_unchanged=true\n",observed.completed);
#else
        std::printf("GENERIC_CLIENT_PASS launches=%u variable_sizes=true full_guards=true unused_function_allowed=true\n",count);
#endif
        return 0;
    } catch(...) {
        bool clean=true;
        if(a){clean=hipFree(a)==hipSuccess&&clean;a=nullptr;}
        if(b){clean=hipFree(b)==hipSuccess&&clean;b=nullptr;}
        std::printf("GENERIC_CLIENT_FAILURE clean=%u terminal=%u\n",unsigned(clean),unsigned(ps5hip_runtime_terminal()));
        ps5hip_runtime_finish();return 20;
    }
}

#endif
