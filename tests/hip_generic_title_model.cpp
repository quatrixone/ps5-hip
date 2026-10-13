// Actual compiler lifecycle and title wrapper; host services only, no GPU ISA.
#include "elevation.hpp"
#include "ps5hip/hip_runtime_control.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <unistd.h>
extern "C" void ps5hip_install_shell_exit_probe();
extern "C" void ps5hip_install_vector_title_finalizer();
extern "C" void catchReturnFromMain(int);
extern "C" unsigned ps5hip_generic_model_live(),ps5hip_generic_model_submits();
extern "C" void ps5hip_generic_model_configure(const char*);
extern "C" void __real___hipUnregisterFatBinary(void**);
extern "C" void __hipRegisterFunction(void**,const void*,char*,const char*,unsigned,uint3*,uint3*,dim3*,dim3*,int*);
extern "C" const unsigned char __ps5hip_names_begin[],__ps5hip_names_end[];
int ps5hip_generic_title_main(int,char**);
namespace {
char scenario[80]{},config[4096]{};
unsigned shell_requests=0,fini_calls=0,sync_calls=0;
[[noreturn]] void fail(){::_exit(90);}
bool is(const char *value){return std::strcmp(scenario,value)==0;}
bool uncertain(){return is("submit")||is("kick")||is("timeout")||is("throw")||is("unknown");}
__attribute__((constructor(101))) void install_before_compiler() {
    ps5hip_install_shell_exit_probe();
    ps5hip_install_vector_title_finalizer();
}
}
extern "C" const char *ps5hip_title_model_config(){return config;}
namespace elevation {
Status request(Capability capability,const char *helper) noexcept {
    if(capability!=Capability::filesystem || std::strcmp(helper,"/app0/lapy.elf") ||
       ps5hip_generic_model_live() || ps5hip_runtime_observation().backend_attempted)fail();
    return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" void __wrap___hipUnregisterFatBinary(void **handle) {
    if(is("skip-unregister"))return;
    if(is("late-partial-destructor")) {
        const char *name="affine_words";
        const auto token=ps5hip_runtime_token_for(name);
        const auto address=reinterpret_cast<uintptr_t>(name),begin=reinterpret_cast<uintptr_t>(__ps5hip_names_begin),end=reinterpret_cast<uintptr_t>(__ps5hip_names_end);
        if(address<begin || address>=end || sizeof("affine_words")>end-address)fail();
        if(!token || ps5hip_generic_model_submits()!=11 || ps5hip_generic_model_live())fail();
        __hipRegisterFunction(handle,token,const_cast<char*>(name),name,~0u,nullptr,nullptr,nullptr,nullptr,nullptr);
        if(hipGetLastError()!=hipErrorInvalidDeviceFunction || hipDeviceSynchronize()!=hipErrorInvalidDeviceFunction)fail();
    }
    __real___hipUnregisterFatBinary(is("reject-unregister")?reinterpret_cast<void**>(1):handle);
}
extern "C" int ps5hip_title_model_fsync(int fd) {
    ++sync_calls;
    if((is("startup-sync")&&sync_calls==1)||(is("final-sync")&&ps5hip_runtime_finalized())){errno=EIO;return -1;}
    return ::fsync(fd);
}
extern "C" void ps5hip_title_model_fini() {
    if(!ps5hip_runtime_finalized()||ps5hip_generic_model_live()||++fini_calls!=1)fail();
}
extern "C" int ps5hip_title_model_shell_exit() {
    if(!ps5hip_runtime_finalized()||ps5hip_generic_model_live()||fini_calls!=1||++shell_requests!=1)fail();
    return is("shell-fail")?-7:0;
}
extern "C" unsigned ps5hip_title_model_sleep(unsigned) {
    const bool success=is("normal")||is("shell-fail");
    if(shell_requests!=(success?1u:0u))fail();
    const unsigned expected_live=uncertain()?6u:is("cleanup")?3u:0u;
    if(ps5hip_generic_model_live()!=expected_live)fail();
    std::printf("HIP_TITLE_MODEL STOP shell_requests=%u live=%u model_only=true\n",shell_requests,expected_live);
    std::fflush(nullptr);::_exit(0);
}
int main(int argc,char **argv) {
    if(argc!=3||std::strlen(argv[1])>=sizeof(scenario)||std::strlen(argv[2])>=sizeof(config))return 91;
    std::strcpy(scenario,argv[1]);std::strcpy(config,argv[2]);
    const bool client_mode=uncertain()||is("cleanup")||is("init")||
        is("missing")||is("corrupt")||is("input")||is("guard")||
        is("wrong-route-output")||is("preflush")||is("late-partial");
    const char *mode=client_mode?scenario:"normal";
    ps5hip_generic_model_configure(mode);
    char *client_argv[]={argv[0],const_cast<char*>(mode)};
    const int status=ps5hip_generic_title_main(2,client_argv);
    if(is("late-live")){void *extra=nullptr;if(hipMalloc(&extra,16)!=hipSuccess)fail();}
    catchReturnFromMain(status);
    std::exit(status);
}
