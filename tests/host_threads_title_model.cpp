#include "elevation.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
extern "C" void ps5hip_install_shell_exit_probe(),ps5hip_install_threads_title_finalizer();
extern "C" void catchReturnFromMain(int),ps5hip_threads_test_configure(const char*);
extern "C" bool ps5hip_threads_title_complete() noexcept;
int ps5hip_threads_title_main(int,char**);
namespace {
char scenario[80]{}, config[4096]{};
unsigned shells=0,finis=0,syncs=0;
bool is(const char *s){return std::strcmp(scenario,s)==0;}
[[noreturn]] void fail(){::_exit(90);}
__attribute__((constructor(101))) void install(){
    ps5hip_install_shell_exit_probe();ps5hip_install_threads_title_finalizer();
}
}
extern "C" const char *ps5hip_threads_model_config(){return config;}
namespace elevation {
Status request(Capability cap,const char *helper) noexcept {
    if(cap!=Capability::filesystem || std::strcmp(helper,"/app0/lapy.elf"))fail();
    return is("elevation")?Status::unavailable:Status::ok;
}
}
extern "C" int ps5hip_threads_model_fsync(int fd) {
    ++syncs;
    // Startup, result, three shell checkpoints, then finalizer persistence.
    if((is("startup-sync")&&syncs==1)||(is("result-sync")&&syncs==2)||(is("final-sync")&&syncs==6)){errno=EIO;return -1;}
    return ::fsync(fd);
}
extern "C" void ps5hip_threads_model_fini(){if(!ps5hip_threads_title_complete()||++finis!=1)fail();}
extern "C" int ps5hip_threads_model_shell_exit(){
    if(!ps5hip_threads_title_complete()||finis!=1||++shells!=1)fail();return is("shell-fail")?-7:0;
}
extern "C" unsigned ps5hip_threads_model_sleep(unsigned) {
    bool success=is("normal")||is("shell-fail");
    if(shells!=unsigned(success)||finis!=unsigned(success))fail();
    std::printf("THREAD_TITLE_MODEL STOP shell_requests=%u fini=%u actual_pthreads=true\n",shells,finis);
    std::fflush(nullptr);::_exit(0);
}
int main(int argc,char**argv){
    if(argc!=3 || std::strlen(argv[1])>=sizeof(scenario)||std::strlen(argv[2])>=sizeof(config))return 91;
    std::strcpy(scenario,argv[1]);std::strcpy(config,argv[2]);
    const bool fault=is("create-second")||is("join-error")||is("mutex-close")||is("wait-timeout")||is("missing-destructor")||is("main-unlock")||is("worker-unlock");
    ps5hip_threads_test_configure(fault?scenario:"normal");
    const int status=ps5hip_threads_title_main(1,argv);
    catchReturnFromMain(status);std::exit(status);
}
