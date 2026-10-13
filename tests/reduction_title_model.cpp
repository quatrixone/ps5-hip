// Reuse the reviewed actual-runtime transport; no second kernel or transport body.
#define __wrap_main reduction_unused_model_main
#include "reduction_companion_model.cpp"
#undef __wrap_main
#include "elevation.hpp"
#include <cerrno>
#include <unistd.h>
extern "C" void ps5hip_install_shell_exit_probe(),ps5hip_install_vector_title_finalizer(),catchReturnFromMain(int);
extern "C" int ps5hip_reduction_title_main(int,char**);
extern "C" void __real___hipUnregisterFatBinary(void**);
namespace {
std::string scenario,configuration;unsigned shell_requests=0,fini_calls=0,sync_calls=0;
bool is(const char*s){return scenario==s;}
bool uncertain(){return is("submit")||is("code-image")||is("entry");}
[[noreturn]]void fail_title(){::_exit(90);}
__attribute__((constructor(101)))void install_title(){ps5hip_install_shell_exit_probe();ps5hip_install_vector_title_finalizer();}
}
extern "C" const char*ps5hip_title_model_config(){return configuration.c_str();}
namespace elevation {
Status request(Capability capability,const char*helper)noexcept{
 if(capability!=Capability::filesystem||std::strcmp(helper,"/app0/lapy.elf")||!allocations.empty()||ps5hip_runtime_observation().backend_attempted)fail_title();
 return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" void __wrap___hipUnregisterFatBinary(void**handle){if(!is("skip-unregister"))__real___hipUnregisterFatBinary(handle);}
extern "C" int ps5hip_title_model_fsync(int fd){++sync_calls;if((is("startup-sync")&&sync_calls==1)||(is("final-sync")&&ps5hip_runtime_finalized())){errno=EIO;return-1;}return::fsync(fd);}
extern "C" void ps5hip_title_model_fini(){if(!ps5hip_runtime_finalized()||!allocations.empty()||++fini_calls!=1)fail_title();}
extern "C" int ps5hip_title_model_shell_exit(){if(!ps5hip_runtime_finalized()||!allocations.empty()||fini_calls!=1||++shell_requests!=1)fail_title();return is("shell-fail")?-7:0;}
extern "C" unsigned ps5hip_title_model_sleep(unsigned){
 const bool success=is("normal")||is("shell-fail");if(shell_requests!=unsigned(success)||allocations.size()!=(uncertain()?7u:0u))fail_title();
 if(uncertain()&&(releases||submits!=1||!ps5hip_runtime_gpu_unresolved()))fail_title();
 std::printf("REDUCTION_TITLE_MODEL STOP shell_requests=%u live=%zu releases=%u submits=%u model_only=true\n",shell_requests,allocations.size(),releases,submits);std::fflush(nullptr);::_exit(0);
}
extern "C" int __wrap_main(int argc,char**argv){
 if(argc!=5)return 91;scenario=argv[1];configuration=argv[2];
 const bool client_fault=is("corrupt")||is("missing")||is("stale")||is("highbits")||is("tail")||is("input-guard")||is("mask-guard")||is("output-guard")||uncertain();
 if(!client_fault&&!is("normal")&&!is("shell-fail")&&!is("skip-unregister")&&!is("elevation")&&!is("config")&&!is("log-exists")&&!is("startup-sync")&&!is("final-sync"))return 91;
 mode=client_fault?scenario:"normal";
 std::ifstream codefile(argv[3],std::ios::binary),fixturefile(argv[4],std::ios::binary);check(bool(codefile)&&bool(fixturefile));
 const std::vector<uint8_t>code{std::istreambuf_iterator<char>(codefile),{}};fixture={std::istreambuf_iterator<char>(fixturefile),{}};auto object=ps5hip::inspect_code_object(code.data(),code.size());check(object.kernels.size()==1&&object.kernels[0].name=="_Z12block_reducePiPmS_m");entry_offset=object.kernels[0].entry_address;image=ps5hip::materialize_code_object(code.data(),code.size());
 const int result=ps5hip_reduction_title_main(0,nullptr);
 if(is("config")||is("elevation")||is("log-exists")){if(result!=2||submits||!allocations.empty())fail_title();std::puts("REDUCTION_TITLE_MODEL PRECLIENT rejected=true");std::fflush(nullptr);::_exit(0);}
 catchReturnFromMain(result);std::exit(result);
}
