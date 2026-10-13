// Actual title/client/compiler constructors, with host-only platform services.
#include "elevation.hpp"
#include "ps5hip/hip_runtime_control.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
extern "C" void ps5hip_install_shell_exit_probe();
extern "C" void ps5hip_install_vector_title_finalizer();
extern "C" void catchReturnFromMain(int);
extern "C" bool ps5hip_vector_title_finalized() noexcept;
extern "C" unsigned ps5hip_vector_model_live(),ps5hip_vector_model_backends(),ps5hip_vector_model_submits();
extern "C" void ps5hip_vector_model_configure(const char*);
extern "C" void __real___hipUnregisterFatBinary(void**),__real_ps5hip_runtime_finish();
int ps5hip_vector_title_main(int,char**);
namespace {
bool finish_injected=false;
char scenario[80]{},config[4096]{};
unsigned shell_requests=0,fini_calls=0,sync_calls=0;
[[noreturn]] void fail(){::_exit(90);}
bool is(const char *value){return std::strcmp(scenario,value)==0;}
__attribute__((constructor(101))) void install_before_compiler() {
    ps5hip_install_shell_exit_probe();
    ps5hip_install_vector_title_finalizer();
}
}
extern "C" const char *ps5hip_title_model_config(){return config;}
namespace elevation {
Status request(Capability capability,const char *helper) noexcept {
    if(capability!=Capability::filesystem || std::strcmp(helper,"/app0/lapy.elf") ||
       ps5hip_vector_model_live() || ps5hip_vector_model_backends()) fail();
    return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" void __wrap_ps5hip_runtime_finish() {
    if (is("late-live") && !finish_injected) {
        finish_injected=true;
        void *extra=nullptr;
        if (hipMalloc(&extra,16)!=hipSuccess || !extra || ps5hip_vector_model_live()!=1) fail();
        std::puts("HIP_TITLE_MODEL PRE_FINISH live=1 rejection_expected=true");
    }
    if (!is("no-finish")) __real_ps5hip_runtime_finish();
}
extern "C" void __wrap___hipUnregisterFatBinary(void **handle) {
    if(is("skip-unregister")) return;
    __real___hipUnregisterFatBinary(is("reject-unregister")?reinterpret_cast<void**>(1):handle);
}
extern "C" int ps5hip_title_model_fsync(int fd) {
    ++sync_calls;
    if((is("startup-sync") && sync_calls==1) || (is("final-sync") && ps5hip_vector_title_finalized())) {
        errno=EIO;return -1;
    }
    return ::fsync(fd);
}
extern "C" void ps5hip_title_model_fini() {
    if(!ps5hip_vector_title_finalized() || ps5hip_vector_model_live() || ++fini_calls!=1)fail();
}
extern "C" int ps5hip_title_model_shell_exit() {
    if(!ps5hip_vector_title_finalized() || ps5hip_vector_model_live() || fini_calls!=1 || ++shell_requests!=1)fail();
    return is("shell-fail")?-7:0;
}
extern "C" unsigned ps5hip_title_model_sleep(unsigned) {
    const bool success=is("normal") || is("post-finish") || is("shell-fail");
    if(shell_requests!=(success?1u:0u))fail();
    unsigned expected_live=is("late-live")?1u:0u;
    if(is("submit") || is("kick") || is("timeout") || is("unknown-submit"))expected_live=7;
    if(is("data-close"))expected_live=3;
    if(ps5hip_vector_model_live()!=expected_live)fail();
    std::printf("HIP_TITLE_MODEL STOP shell_requests=%u live=%u model_only=true\n",shell_requests,expected_live);
    std::fflush(nullptr);::_exit(0);
}
int main(int argc,char **argv) {
    if(argc!=3 || std::strlen(argv[1])>=sizeof(scenario) || std::strlen(argv[2])>=sizeof(config))return 91;
    std::strcpy(scenario,argv[1]);std::strcpy(config,argv[2]);
    const bool backend_mode=is("submit") || is("kick") || is("timeout") || is("unknown-submit") ||
        is("missing") || is("corrupt") || is("input") || is("guard") || is("data-allocate") || is("data-close") || is("init");
    ps5hip_vector_model_configure(backend_mode?scenario:"normal");
    const int status=ps5hip_vector_title_main(1,argv);

    if(is("post-finish")) {
        const auto live=ps5hip_vector_model_live(), submits=ps5hip_vector_model_submits();
        void *extra=reinterpret_cast<void*>(1);
        if(hipMalloc(&extra,16)!=hipErrorDeinitialized || extra || ps5hip_vector_model_live()!=live || ps5hip_vector_model_submits()!=submits || !ps5hip_runtime_ready())fail();
        std::puts("HIP_TITLE_MODEL POST_FINISH allocation_rejected=true ownership_unchanged=true");
    }
    catchReturnFromMain(status);
    std::exit(status);
}
