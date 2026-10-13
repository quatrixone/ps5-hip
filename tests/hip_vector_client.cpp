// Full pinned HIP header client; launch_vector_add contains the genuine <<<>>> stub.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#ifndef PS5HIP_VECTOR_MODEL
#include <unistd.h>
#endif
void launch_vector_add(const float*,const float*,float*,unsigned);
extern "C" void ps5hip_vector_finish();
extern "C" bool ps5hip_vector_terminal();
#ifdef PS5HIP_VECTOR_MODEL
extern "C" const void *ps5hip_vector_token();
extern "C" bool ps5hip_vector_gpu_unresolved();
extern "C" void ps5hip_vector_model_configure(const char*);
extern "C" unsigned ps5hip_vector_model_backends(),ps5hip_vector_model_live(),ps5hip_vector_model_submits();
#endif
namespace{
void check(bool v){if(!v)throw std::runtime_error("vector client assertion");}
uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
constexpr size_t words=4096,guard=16,count=73;
constexpr uint32_t poison=0x7fcabcde;
using Page=std::array<uint32_t,words>;
}
int main(int argc,char**argv){
    const char*mode=argc==2?argv[1]:"normal";
#ifdef PS5HIP_VECTOR_MODEL
    ps5hip_vector_model_configure(mode);
#else
    (void)mode;
#endif
    try{
#ifdef PS5HIP_VECTOR_PARTIAL
        check(hipPeekAtLastError()==hipErrorInvalidDeviceFunction);
#ifdef PS5HIP_VECTOR_MODEL
        check(!ps5hip_vector_model_backends()&&!ps5hip_vector_model_live());
#endif
        ps5hip_vector_finish();std::puts("VECTOR_CLIENT_PARTIAL_PASS no_gpu=true");return 0;
#endif
        check(hipDeviceSynchronize()==hipSuccess);
        Page a,b,out,read;
        a.fill(poison);b.fill(poison);out.fill(poison);
        for(size_t i=0;i<count;++i){a[guard+i]=bits(float(i)*.25f);b[guard+i]=bits(32.0f-float(i)*.125f);}
        void *pa=nullptr,*pb=nullptr,*po=nullptr;
        check(hipMalloc(&pa,sizeof(a))==hipSuccess&&hipMalloc(&pb,sizeof(b))==hipSuccess&&hipMalloc(&po,sizeof(out))==hipSuccess);
        auto*A=reinterpret_cast<float*>(static_cast<unsigned char*>(pa)+64);
        auto*B=reinterpret_cast<float*>(static_cast<unsigned char*>(pb)+64);
        auto*O=reinterpret_cast<float*>(static_cast<unsigned char*>(po)+64);
        check(hipMemcpy(pa,a.data(),sizeof(a),hipMemcpyHostToDevice)==hipSuccess);
        check(hipMemcpy(pb,b.data(),sizeof(b),hipMemcpyHostToDevice)==hipSuccess);
#ifdef PS5HIP_VECTOR_MODEL
        check(!ps5hip_vector_model_backends());
        unsigned n=73;void*args[]={&A,&B,&O,&n};const auto token=ps5hip_vector_token();
        check(hipLaunchKernel(nullptr,dim3(2),dim3(64),args)==hipErrorInvalidValue);
        check(hipLaunchKernel(token,dim3(3),dim3(64),args)==hipErrorNotSupported);
        check(hipLaunchKernel(token,dim3(2),dim3(32),args)==hipErrorNotSupported);
        check(hipLaunchKernel(token,dim3(2),dim3(64),args,1)==hipErrorNotSupported);
        check(hipLaunchKernel(token,dim3(2),dim3(64),args,0,reinterpret_cast<hipStream_t>(1))==hipErrorNotSupported);
        n=74;check(hipLaunchKernel(token,dim3(2),dim3(64),args)==hipErrorInvalidValue);n=73;
        const auto saved=A;A=nullptr;check(hipLaunchKernel(token,dim3(2),dim3(64),args)==hipErrorInvalidValue);A=saved;
        void*dead=nullptr;check(hipMalloc(&dead,16)==hipSuccess&&hipFree(dead)==hipSuccess);
        A=static_cast<float*>(dead);check(hipLaunchKernel(token,dim3(2),dim3(64),args)==hipErrorInvalidValue);A=saved;
        A=reinterpret_cast<float*>(uintptr_t(0x4000000000ull));
        check(hipLaunchKernel(token,dim3(2),dim3(64),args)==hipErrorInvalidValue);A=saved;
        A=reinterpret_cast<float*>(static_cast<unsigned char*>(pa)+sizeof(a)-4);
        check(hipLaunchKernel(token,dim3(2),dim3(64),args)==hipErrorInvalidValue);A=saved;
        check(!ps5hip_vector_model_backends()&&!ps5hip_vector_model_submits());
#endif
        for(unsigned round=0;round<2;++round){
            check(hipMemcpy(po,out.data(),sizeof(out),hipMemcpyHostToDevice)==hipSuccess);
            launch_vector_add(A,B,O,73);
            const auto result=hipPeekAtLastError();
#ifdef PS5HIP_VECTOR_MODEL
            if(std::strcmp(mode,"preflush")==0&&round==0){
                check(result!=hipSuccess&&!ps5hip_vector_terminal()&&ps5hip_vector_model_live()==3);
                launch_vector_add(A,B,O,73);
            }else if(result!=hipSuccess){
                check(result==hipErrorUnknown);
                if(ps5hip_vector_terminal()){
                    check(hipDeviceSynchronize()==hipErrorUnknown);
                    check(hipMemcpy(read.data(),po,sizeof(read),hipMemcpyDeviceToHost)==hipErrorUnknown);
                    check(hipFree(po)==hipErrorUnknown&&hipFree(nullptr)==hipErrorUnknown);
                    void*empty=reinterpret_cast<void*>(1);check(hipMalloc(&empty,0)==hipErrorUnknown&&empty==nullptr);
                    const auto submits=ps5hip_vector_model_submits();launch_vector_add(A,B,O,73);
                    check(hipPeekAtLastError()==hipErrorUnknown&&ps5hip_vector_model_submits()==submits);
                    const bool unresolved=ps5hip_vector_gpu_unresolved();
                    check(ps5hip_vector_model_live()==(unresolved?7u:3u));
                    ps5hip_vector_finish();
                    std::puts(unresolved?"VECTOR_CLIENT_QUARANTINE_PASS retained=7 no_successful_unregister=true model_only=true":
                        "VECTOR_CLIENT_CLEANUP_FAILURE_PASS gpu_unresolved=false no_successful_unregister=true model_only=true");return 0;
                }
                const bool init=std::strcmp(mode,"init")==0;
                check(hipDeviceSynchronize()==hipSuccess);
                check(ps5hip_vector_model_live()==3);
                if(init){
                    check(ps5hip_vector_model_backends()==1);
                    launch_vector_add(A,B,O,73);check(hipPeekAtLastError()==hipErrorUnknown&&ps5hip_vector_model_backends()==1);
                }else check(std::strcmp(mode,"code-allocate")==0||std::strcmp(mode,"kernarg-map")==0||
                           std::strcmp(mode,"command-map")==0||std::strcmp(mode,"marker-map")==0);
                check(hipMemcpy(read.data(),pa,sizeof(read),hipMemcpyDeviceToHost)==hipSuccess&&read==a);
                check(hipFree(pa)==hipSuccess&&hipFree(pb)==hipSuccess&&hipFree(po)==hipSuccess);
                ps5hip_vector_finish();std::puts(init?"VECTOR_CLIENT_INIT_FAILURE_PASS no_retry=true":
                    "VECTOR_CLIENT_SETUP_FAILURE_PASS clean_unwind=true");return 0;
            }
#else
            if(result!=hipSuccess){
                std::fprintf(stderr,"VECTOR_CLIENT_FAILURE error=%d terminal=%u\n",int(result),unsigned(ps5hip_vector_terminal()));
                if(ps5hip_vector_terminal()){std::fflush(nullptr);for(;;)::sleep(1);}
                throw std::runtime_error("vector launch failed");
            }
#endif
            check(hipPeekAtLastError()==hipSuccess&&hipDeviceSynchronize()==hipSuccess);
            check(hipMemcpy(read.data(),po,sizeof(read),hipMemcpyDeviceToHost)==hipSuccess);
            for(size_t i=0;i<words;++i)check(read[i]==(i>=guard&&i<guard+count?bits(32.0f+float(i-guard)*.125f):poison));
            check(hipMemcpy(read.data(),pa,sizeof(read),hipMemcpyDeviceToHost)==hipSuccess&&read==a);
            check(hipMemcpy(read.data(),pb,sizeof(read),hipMemcpyDeviceToHost)==hipSuccess&&read==b);
        }
        check(hipFree(pa)==hipSuccess&&hipFree(pb)==hipSuccess&&hipFree(po)==hipSuccess);
        check(hipFree(po)==hipErrorInvalidDevicePointer);
        check(hipGetLastError()==hipErrorInvalidDevicePointer&&hipGetLastError()==hipSuccess);
#ifdef PS5HIP_VECTOR_MODEL
        check(ps5hip_vector_model_backends()==1&&ps5hip_vector_model_submits()==2&&!ps5hip_vector_model_live());
#endif
        ps5hip_vector_finish();std::puts("VECTOR_CLIENT_PASS exact_values=146 immutable_inputs=true full_guards=true blocking=true hardware_validated=false");return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"VECTOR_CLIENT_REJECT %s\n",e.what());ps5hip_vector_finish();return 20;}
}
