// Real translated compiler/client lifecycle; services and GPU transport are models.
#include "elevation.hpp"
#include "ps5hip/hip_runtime_control.h"
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
extern "C" void ps5hip_install_shell_exit_probe(),ps5hip_install_vector_title_finalizer(),catchReturnFromMain(int);
extern "C" unsigned ps5hip_hipify_model_live(),ps5hip_hipify_model_submits(),ps5hip_hipify_model_backends();
extern "C" void ps5hip_hipify_model_configure(const char*),ps5hip_hipify_model_duplicate(),ps5hip_hipify_model_no_retry();
extern "C" void __real___hipUnregisterFatBinary(void**),__real_ps5hip_runtime_finish();
int ps5hip_hipify_title_main(int,char**);
namespace {
char scenario[80]{},config[4096]{};unsigned shell_requests=0,fini_calls=0,sync_calls=0;
[[noreturn]] void fail(){::_exit(90);}
bool is(const char*s){return std::strcmp(scenario,s)==0;}
bool uncertain(){return is("submit")||is("kick")||is("timeout")||is("throw")||is("unknown")||is("postflush");}
__attribute__((constructor(101))) void install_before_compiler(){ps5hip_install_shell_exit_probe();ps5hip_install_vector_title_finalizer();}
}
extern "C" const char *ps5hip_title_model_config(){return config;}
namespace elevation {
Status request(Capability cap,const char*helper) noexcept {
 if(cap!=Capability::filesystem||std::strcmp(helper,"/app0/lapy.elf")||ps5hip_hipify_model_live()||ps5hip_runtime_observation().backend_attempted)fail();
 return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" void __wrap_ps5hip_runtime_finish(){if(!is("no-finish"))__real_ps5hip_runtime_finish();}
extern "C" void __wrap___hipUnregisterFatBinary(void**handle){
 if(is("skip-unregister"))return;
 if(is("late-partial-destructor")){if(ps5hip_hipify_model_submits()!=6||ps5hip_hipify_model_live())fail();ps5hip_hipify_model_duplicate();}
 __real___hipUnregisterFatBinary(is("reject-unregister")?reinterpret_cast<void**>(1):handle);
}
extern "C" int ps5hip_title_model_fsync(int fd){
 ++sync_calls;if((is("startup-sync")&&sync_calls==1)||(is("final-sync")&&ps5hip_runtime_finalized())){errno=EIO;return -1;}return ::fsync(fd);
}
extern "C" void ps5hip_title_model_fini(){if(!ps5hip_runtime_finalized()||ps5hip_hipify_model_live()||++fini_calls!=1)fail();}
extern "C" int ps5hip_title_model_shell_exit(){
 if(!ps5hip_runtime_finalized()||ps5hip_hipify_model_live()||fini_calls!=1||++shell_requests!=1)fail();return is("shell-fail")?-7:0;
}
extern "C" unsigned ps5hip_title_model_sleep(unsigned){
 const bool success=is("normal")||is("shell-fail");if(shell_requests!=unsigned(success))fail();
 const unsigned live=uncertain()?7u:is("cleanup")?4u:0u;if(ps5hip_hipify_model_live()!=live)fail();
 if(uncertain()||is("cleanup")){
  if(ps5hip_runtime_gpu_unresolved()!=uncertain())fail();ps5hip_hipify_model_no_retry();
 }
 std::printf("HIP_TITLE_MODEL STOP shell_requests=%u live=%u model_only=true\n",shell_requests,live);std::fflush(nullptr);::_exit(0);
}
int main(int argc,char**argv){
 if(argc!=3||std::strlen(argv[1])>=sizeof(scenario)||std::strlen(argv[2])>=sizeof(config))return 91;
 std::strcpy(scenario,argv[1]);std::strcpy(config,argv[2]);
 const bool fault=uncertain()||is("cleanup")||is("init")||is("preflush")||is("data-allocate")||is("missing")||is("corrupt")||is("input")||is("guard")||is("stale-second");
 ps5hip_hipify_model_configure(fault?scenario:"normal");
 if(!ps5hip_runtime_registration_ready()||ps5hip_hipify_model_backends()||ps5hip_hipify_model_live())fail();
 if(is("partial"))ps5hip_hipify_model_duplicate();
 const int status=ps5hip_hipify_title_main(0,nullptr);
 if(is("late-live")){void*extra=nullptr;if(hipMalloc(&extra,16)!=hipSuccess)fail();}
 catchReturnFromMain(status);std::exit(status);
}
