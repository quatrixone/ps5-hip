// One fixed HIP title; startup/exit follow the existing finite shell lifecycle.
#include "native_runtime.h"
#include "elevation.hpp"
#include <cstdlib>
#include <fstream>

#ifdef PS5HIP_ROUTES_TITLE
#include "ps5hip/hip_runtime_control.h"
#define ps5hip_vector_title_registration ps5hip_runtime_registration_ready
#define ps5hip_vector_title_ready ps5hip_runtime_ready
#define ps5hip_vector_title_finalized ps5hip_runtime_finalized
#define ps5hip_vector_terminal ps5hip_runtime_terminal
#define ps5hip_vector_client_main ps5hip_routes_client_main
#endif
extern "C" bool ps5hip_vector_title_registration() noexcept;
extern "C" bool ps5hip_vector_title_ready() noexcept;
extern "C" bool ps5hip_vector_title_finalized() noexcept;
extern "C" bool ps5hip_vector_terminal();
int ps5hip_vector_client_main(int,char**);
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
extern "C" const char *ps5hip_title_model_config();
#endif
namespace {
bool finalizer_installed=false;
[[noreturn]] void quarantine(const char *reason) noexcept {
    std::fprintf(stderr,"HIP_TITLE QUARANTINED reason=%s terminal=%u no_shell_request=true\n",reason,unsigned(ps5hip_vector_terminal()));
    (void)ps5hip_canary::sync_log();
    for(;;) ::sleep(1);
}
bool persist() noexcept {
    return !std::ferror(stdout) && !std::ferror(stderr) && ps5hip_canary::sync_log();
}
void finalized() {
    // State has been destroyed: only inspect its trivial external observation.
    if(!ps5hip_vector_title_finalized()) {
        std::fputs("HIP_TITLE QUARANTINED reason=finalization no_shell_request=true\n",stderr);
        (void)ps5hip_canary::sync_log();
        for(;;) ::sleep(1);
    }
#ifdef PS5HIP_ROUTES_TITLE
    std::puts("HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=8");
#else
    std::puts("HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed exact_values=146");
#endif
    if(!persist()) {
        std::fputs("HIP_TITLE QUARANTINED reason=final-log no_shell_request=true\n",stderr);
        (void)ps5hip_canary::sync_log();
        for(;;) ::sleep(1);
    }
}
}
extern "C" void ps5hip_install_vector_title_finalizer() {
    finalizer_installed=std::atexit(finalized)==0;
}
int main(int argc,char **argv) {
    try {
        // Read sandbox configuration before the helper changes the process root.
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
        const char *config_path=ps5hip_title_model_config();
#else
        const char *config_path="/app0/results-dir.txt";
#endif
        std::ifstream configuration(config_path,std::ios::binary);
        char bytes[242]{};configuration.read(bytes,sizeof(bytes));
        const auto length=configuration.gcount();
        if(!configuration.eof() || length<2 || length>240) return 2;
        const std::string directory(bytes,static_cast<size_t>(length));
        if(!ps5hip_canary::valid_results_directory(directory)) return 2;
        if(elevation::request(elevation::Capability::filesystem,"/app0/lapy.elf")!=elevation::Status::ok) return 2;
#ifdef PS5HIP_ROUTES_TITLE
        const auto log=directory+"/ps5-hip-gfx1013-hip_routes.log";
#else
        const auto log=directory+"/ps5-hip-gfx1013-hip_vector.log";
#endif
        if(!ps5hip_canary::redirect_log(log.c_str())) return 2;
#ifdef PS5HIP_ROUTES_TITLE
        std::puts("PS5HIP START target=gfx1013 kernel=hip_routes");
#else
        std::puts("PS5HIP START target=gfx1013 kernel=hip_vector");
#endif
        if(!finalizer_installed || !ps5hip_vector_title_registration()) quarantine("registration");
#ifdef PS5HIP_ROUTES_TITLE
        std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0");
#endif
        ps5hip_canary::check_runtime();
        if(!persist()) quarantine("startup-log");
        const int status=ps5hip_vector_client_main(argc,argv);
        if(status || !ps5hip_vector_title_ready()) quarantine("client-or-ownership");
#ifdef PS5HIP_ROUTES_TITLE
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=8 word_comparisons=393216 finalization=pending");
#else
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=2 exact_values=146 finalization=pending");
#endif
        if(!persist()) quarantine("compute-log");
        return 0;
    } catch(const std::exception &e) { quarantine(e.what()); }
    catch(...) { quarantine("unknown exception"); }
}
