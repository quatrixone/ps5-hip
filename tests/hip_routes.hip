#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <memory>
#ifndef PS5HIP_RUNTIME_MODEL
#include <unistd.h>
#endif

extern "C" __global__ __launch_bounds__(64)
void routes_integer(const unsigned *input,unsigned *output,unsigned n,unsigned salt) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned i=__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x();
    if(i<n) output[i]=input[i]^salt;
#endif
}
extern "C" __global__ __launch_bounds__(64)
void routes_lds(const unsigned *input,unsigned *output) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned values[64];
    const unsigned lane=__builtin_amdgcn_workitem_id_x(),group=__builtin_amdgcn_workgroup_id_x();
    values[lane]=input[group*64+lane];
    __builtin_amdgcn_s_barrier();
    for(unsigned step=32;step;step>>=1) {
        if(lane<step) values[lane]+=values[lane+step];
        __builtin_amdgcn_s_barrier();
    }
    if(!lane) output[group]=values[0];
#endif
}
extern "C" __global__ __attribute__((amdgpu_num_vgpr(16)))
void routes_spill(const volatile unsigned *input,volatile unsigned *output) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned i=__builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x();
    // These arguments are registry-owned global allocations, not private pointers.
    const auto *global_input=(const volatile unsigned __attribute__((address_space(1)))*)input;
    auto *global_output=(volatile unsigned __attribute__((address_space(1)))*)output;
    unsigned values[64];
    #pragma unroll
    for(unsigned j=0;j<64;++j) values[j]=global_input[i*64+j];
    #pragma unroll
    for(unsigned j=0;j<64;++j) global_output[i*64+j]=values[j];
#endif
}
extern "C" __global__ __launch_bounds__(64)
void routes_geometry(const unsigned *input,unsigned *records,unsigned *tail,
                     unsigned pitch_x,unsigned pitch_y,unsigned pitch_z,unsigned record_words) {
    const unsigned bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
    const unsigned tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;
    if(pitch_x!=24 || pitch_y!=16 || pitch_z!=1 || record_words!=6144 ||
       bx>=3 || by>=2 || bz || tx>=8 || ty>=8 || tz) return;
    const unsigned x=8*bx+tx,y=8*by+ty,i=y*pitch_x+x;
    if(i>=384 || i>=record_words/16) return;
    unsigned *p=records+16*i;
    p[0]=bx;p[1]=by;p[2]=bz;p[3]=tx;p[4]=ty;p[5]=tz;
    p[6]=blockDim.x;p[7]=blockDim.y;p[8]=blockDim.z;
    p[9]=gridDim.x;p[10]=gridDim.y;p[11]=gridDim.z;
    p[12]=x;p[13]=y;p[14]=0;p[15]=input[i];
    if(x<21 && y<13) tail[i]=input[i]^(x*17u+y*257u+0x53a971bdu);
}

extern "C" void ps5hip_runtime_finish();
extern "C" bool ps5hip_runtime_terminal(),ps5hip_runtime_gpu_unresolved();
#ifdef PS5HIP_RUNTIME_MODEL
extern "C" const void *ps5hip_runtime_token_for(const char*);
extern "C" void **ps5hip_runtime_handle();
extern "C" const unsigned char __ps5hip_names_begin[],__ps5hip_names_end[];
extern "C" void __hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
extern "C" void ps5hip_routes_model_configure(const char*);
extern "C" unsigned ps5hip_routes_model_backends(),ps5hip_routes_model_live(),ps5hip_routes_model_submits(),ps5hip_routes_model_releases();
#endif
namespace {
constexpr size_t words=16384,guard=16,source_words=8192;
constexpr unsigned poison=0xa5a5a5a5;
using Page=std::array<unsigned,words>;
void check(bool ok) { if(!ok) throw std::runtime_error("routes client assertion"); }
unsigned value(unsigned i,unsigned round) { return (i*17u+round*1009u+31u)^0x531a9c7du; }
bool close_data(void *&a,void *&b,void *&c) {
    bool ok=true;
    for(void **owner:{&a,&b,&c}) {
        if(ps5hip_runtime_terminal()) return false;
        if(*owner) {if(hipFree(*owner)==hipSuccess)*owner=nullptr;else ok=false;}
    }
    return ok;
}
void launch(unsigned route,const unsigned *input,unsigned *output,unsigned *tail,unsigned salt) {
    switch(route) {
    case 0:routes_integer<<<2,64>>>(input,output,128,salt);break;
    case 1:routes_lds<<<2,64>>>(input,output);break;
    case 2:routes_spill<<<2,64>>>(input,output);break;
    case 3:routes_geometry<<<dim3(3,2,1),dim3(8,8,1)>>>(input,output,tail,24,16,1,6144);break;
    default:throw std::runtime_error("unknown route");
    }
}
void oracle(unsigned route,const Page &input,Page &output,Page &tail,unsigned salt) {
    output.fill(poison);tail.fill(poison);
    if(route==0)for(unsigned i=0;i<128;++i)output[guard+i]=input[guard+i]^salt;
    if(route==1)for(unsigned group=0;group<2;++group) {
        unsigned sum=0;for(unsigned lane=0;lane<64;++lane)sum+=input[guard+64*group+lane];
        output[guard+group]=sum;
    }
    if(route==2)for(unsigned i=0;i<source_words;++i)output[guard+i]=input[guard+i];
    if(route==3)for(unsigned i=0;i<384;++i) {
        const unsigned x=i%24,y=i/24;
        const unsigned fields[16]={x/8,y/8,0,x%8,y%8,0,8,8,1,3,2,1,x,y,0,input[guard+i]};
        for(unsigned j=0;j<16;++j)output[guard+16*i+j]=fields[j];
        if(x<21 && y<13)tail[guard+i]=input[guard+i]^(x*17u+y*257u+0x53a971bdu);
    }
}
}
int main(int argc,char **argv) {
    const char *mode=argc==2?argv[1]:"normal";
#ifdef PS5HIP_RUNTIME_MODEL
    ps5hip_routes_model_configure(mode);
#else
    (void)mode;
#endif
    void *pi=nullptr,*po=nullptr,*pt=nullptr;
    try {
#ifdef PS5HIP_RUNTIME_MODEL
        if(std::strcmp(mode,"partial")==0) {
        check(hipPeekAtLastError()==hipErrorInvalidDeviceFunction);
        check(!ps5hip_routes_model_backends()&&!ps5hip_routes_model_live()&&!ps5hip_routes_model_submits());
        const unsigned *in=nullptr;unsigned *out=nullptr,n=128,salt=0x11;void *args[]={&in,&out,&n,&salt};
        check(hipGetLastError()==hipErrorInvalidDeviceFunction);
        void *zero=reinterpret_cast<void*>(1);
        check(hipMalloc(&zero,0)==hipErrorInvalidDeviceFunction&&zero==nullptr);
        check(hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
        check(hipLaunchKernel(reinterpret_cast<const void*>(routes_integer),dim3(2),dim3(64),args)==hipErrorInvalidDeviceFunction);
        check(!ps5hip_runtime_token_for("routes_integer")&&!ps5hip_runtime_token_for("routes_lds")&&!ps5hip_runtime_token_for("routes_spill"));
        ps5hip_runtime_finish();std::puts("ROUTES_CLIENT_PARTIAL_PASS no_gpu=true");return 0;
        }
#endif
        check(hipDeviceSynchronize()==hipSuccess);
#ifdef PS5HIP_RUNTIME_MODEL
        const char *names[]={"routes_integer","routes_lds","routes_spill","routes_geometry"};
        const void *tokens[4]{};
        for(unsigned i=0;i<4;++i) {
            tokens[i]=ps5hip_runtime_token_for(names[i]);check(tokens[i]!=nullptr);
            for(unsigned j=0;j<i;++j)check(tokens[i]!=tokens[j]);
        }
        const unsigned *unused_in=nullptr;unsigned *unused_out=nullptr,n=128,salt=0x11;
        void *unknown_args[]={&unused_in,&unused_out,&n,&salt};
        check(hipLaunchKernel(reinterpret_cast<const void*>(1),dim3(2),dim3(64),unknown_args)==hipErrorInvalidValue);
        check(!ps5hip_routes_model_backends()&&!ps5hip_routes_model_live()&&!ps5hip_routes_model_submits());
#endif
        auto pages=std::make_unique<std::array<Page,5>>();
        auto &input=(*pages)[0],&empty=(*pages)[1],&expected=(*pages)[2],&expected_tail=(*pages)[3],&read=(*pages)[4];
        empty.fill(poison);
        check(hipMalloc(&pi,sizeof(Page))==hipSuccess && hipMalloc(&po,sizeof(Page))==hipSuccess && hipMalloc(&pt,sizeof(Page))==hipSuccess);
        auto *in=static_cast<unsigned*>(pi)+guard,*out=static_cast<unsigned*>(po)+guard,*tail=static_cast<unsigned*>(pt)+guard;
        for(unsigned round=0;round<2;++round) {
            input.fill(poison);for(unsigned i=0;i<source_words;++i)input[guard+i]=value(i,round);
            check(hipMemcpy(pi,input.data(),sizeof(Page),hipMemcpyHostToDevice)==hipSuccess);
            for(unsigned route=0;route<4;++route) {
                check(hipMemcpy(po,empty.data(),sizeof(Page),hipMemcpyHostToDevice)==hipSuccess);
                check(hipMemcpy(pt,empty.data(),sizeof(Page),hipMemcpyHostToDevice)==hipSuccess);
                const unsigned salt=round?0x22:0x11;launch(route,in,out,tail,salt);
#ifdef PS5HIP_RUNTIME_MODEL
                if(std::strcmp(mode,"preflush")==0 && !round && !route && hipPeekAtLastError()!=hipSuccess) {
                    check(!ps5hip_runtime_terminal()&&ps5hip_routes_model_live()==3&&!ps5hip_routes_model_submits());
                    launch(route,in,out,tail,salt); // One explicitly model-only clean pre-submit retry.
                }
#endif
                check(hipGetLastError()==hipSuccess && hipDeviceSynchronize()==hipSuccess);
                oracle(route,input,expected,expected_tail,salt);
                check(hipMemcpy(read.data(),pi,sizeof(Page),hipMemcpyDeviceToHost)==hipSuccess && read==input);
                check(hipMemcpy(read.data(),po,sizeof(Page),hipMemcpyDeviceToHost)==hipSuccess && read==expected);
                check(hipMemcpy(read.data(),pt,sizeof(Page),hipMemcpyDeviceToHost)==hipSuccess && read==expected_tail);
                std::printf("ROUTES_COMPLETE round=%u route=%u blocking=true full_guards=true inputs=unchanged\n",round+1,route);
            }
        }
        check(close_data(pi,po,pt));
#ifdef PS5HIP_RUNTIME_MODEL
        if(std::strcmp(mode,"late-partial")==0) {
            const auto submits=ps5hip_routes_model_submits(),releases=ps5hip_routes_model_releases();
            check(submits==8&&!ps5hip_routes_model_live());
            const char *name="routes_integer";
            const auto address=reinterpret_cast<uintptr_t>(name),begin=reinterpret_cast<uintptr_t>(__ps5hip_names_begin),end=reinterpret_cast<uintptr_t>(__ps5hip_names_end);
            check(address>=begin && address<end && sizeof("routes_integer")<=end-address);
            __hipRegisterFunction(ps5hip_runtime_handle(),tokens[0],const_cast<char*>(name),name,~0u,nullptr,nullptr,nullptr,nullptr,nullptr);
            check(hipPeekAtLastError()==hipErrorInvalidDeviceFunction);
            check(hipGetLastError()==hipErrorInvalidDeviceFunction);
            void *zero=reinterpret_cast<void*>(1);
            check(hipMalloc(&zero,0)==hipErrorInvalidDeviceFunction&&zero==nullptr);
            check(hipLaunchKernel(tokens[0],dim3(2),dim3(64),unknown_args)==hipErrorInvalidDeviceFunction);
            check(!ps5hip_runtime_token_for(name)&&hipDeviceSynchronize()==hipErrorInvalidDeviceFunction);
            check(ps5hip_routes_model_submits()==submits&&ps5hip_routes_model_releases()==releases);
            ps5hip_runtime_finish();std::puts("ROUTES_CLIENT_LATE_PARTIAL_PASS completed=8 stale_tokens=true finalized_required=false");return 0;
        }
#endif
        ps5hip_runtime_finish();
#ifdef PS5HIP_ROUTES_TITLE
        std::puts("ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true");return 0;
#else
        std::puts("ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true hardware_validated=false");return 0;
#endif
    } catch(const std::exception &error) {
        std::fprintf(stderr,"ROUTES_CLIENT_REJECT mode=%s reason=%s terminal=%u\n",mode,error.what(),unsigned(ps5hip_runtime_terminal()));
        if(!ps5hip_runtime_terminal()&&close_data(pi,po,pt)) {ps5hip_runtime_finish();return 20;}
#ifdef PS5HIP_RUNTIME_MODEL
        const bool unresolved=ps5hip_runtime_gpu_unresolved();
        const bool completed_cleanup=std::strcmp(mode,"scratch-cleanup")==0;
        const auto live=ps5hip_routes_model_live(),submits=ps5hip_routes_model_submits(),releases=ps5hip_routes_model_releases();
        check(unresolved!=completed_cleanup && live==(completed_cleanup?4u:8u) && submits==3);
        unsigned sink=0;void *zero=reinterpret_cast<void*>(1);
        check(hipMemcpy(&sink,po,sizeof(sink),hipMemcpyDeviceToHost)==hipErrorUnknown);
        check(hipFree(po)==hipErrorUnknown&&hipFree(nullptr)==hipErrorUnknown);
        check(hipMalloc(&zero,0)==hipErrorUnknown&&zero==nullptr&&hipDeviceSynchronize()==hipErrorUnknown);
        launch(2,static_cast<const unsigned*>(pi)+guard,static_cast<unsigned*>(po)+guard,static_cast<unsigned*>(pt)+guard,0x11);
        check(hipGetLastError()==hipErrorUnknown);
        check(ps5hip_routes_model_live()==live&&ps5hip_routes_model_submits()==submits&&ps5hip_routes_model_releases()==releases);
        std::printf("ROUTES_CLIENT_TERMINAL_PASS mode=%s unresolved=%u live=%u submits=%u releases=%u ownership_retained=true\n",mode,unsigned(unresolved),live,submits,releases);
        std::fflush(nullptr);
#ifdef PS5HIP_ROUTES_TITLE
        return 30; // The title wrapper owns quarantine and forbids shell finalization.
#else
        std::_Exit(30);
#endif
#else
        for(;;)::sleep(1);
#endif
    }
}
