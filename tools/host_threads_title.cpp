// CPU-only title; the shell/fini implementation is the existing measured helper.
#include "host_threads.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include <cstdlib>
#include <fstream>

namespace {
bool installed=false, passed=false;
ps5hip_threads::Report final_report{};
bool persist() noexcept { return !std::ferror(stdout) && !std::ferror(stderr) && ps5hip_canary::sync_log(); }
[[noreturn]] void quarantine(const char *reason) noexcept {
    std::fprintf(stderr,"HOST_THREADS QUARANTINED reason=%s no_shell_request=true\n",reason);
    (void)ps5hip_canary::sync_log(); for(;;)::sleep(1);
}
void finalized() {
    if(!passed || !final_report.passed || !final_report.cleanup_complete)quarantine("finalization");
    std::printf("HOST_THREADS FINAL joined=%u destructors=%u,%u synchronization=closed cpu_only=true\n",
        final_report.joined,final_report.destructors[0],final_report.destructors[1]);
    if(!persist())quarantine("final-log");
}
}
extern "C" bool ps5hip_threads_title_complete() noexcept {return passed;}
extern "C" void ps5hip_install_threads_title_finalizer() {installed=std::atexit(finalized)==0;}
#ifdef PS5HIP_THREADS_MODEL
extern "C" const char *ps5hip_threads_model_config();
#endif
int main(int,char**) {
    try {
#ifdef PS5HIP_THREADS_MODEL
        const char *path=ps5hip_threads_model_config();
#else
        const char *path="/app0/results-dir.txt";
#endif
        std::ifstream configuration(path,std::ios::binary);
        char bytes[242]{};configuration.read(bytes,sizeof(bytes));const auto n=configuration.gcount();
        if(!configuration.eof() || n<2 || n>240)return 2;
        const std::string directory(bytes,static_cast<std::size_t>(n));
        if(!ps5hip_canary::valid_results_directory(directory))return 2;
        if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok)return 2;
        if(!ps5hip_canary::redirect_log((directory+"/ps5-hip-host_threads.log").c_str()))return 2;
        std::puts("PS5HIP START target=cpu kernel=host_threads");
        if(!installed)quarantine("installer");
        ps5hip_canary::check_runtime();
        if(!persist())quarantine("startup-log");
        final_report=ps5hip_threads::run();
        const auto &r=final_report;
        std::printf("HOST_THREADS RESULT passed=%u clean=%u error=%d created=%u joined=%u busy=%u counter=%u turns=%u destructors=%u,%u tls=%u,%u,%u stack_bytes=%zu\n",
            unsigned(r.passed),unsigned(r.cleanup_complete),r.error,r.created,r.joined,r.busy,r.counter,r.turns,
            r.destructors[0],r.destructors[1],r.main_tls,r.worker_tls[0],r.worker_tls[1],ps5hip_threads::worker_stack_bytes);
        if(!r.passed || !r.cleanup_complete)quarantine("thread-proof");
        passed=true;
        std::puts("HOST_THREADS PASS workers=2 updates=256 ordered_turns=256 key_destructors=2 gpu_calls=0 finalization=pending");
        if(!persist())quarantine("result-log");
        return 0;
    } catch(const std::exception &e){quarantine(e.what());}
    catch(...){quarantine("unknown-exception");}
}
