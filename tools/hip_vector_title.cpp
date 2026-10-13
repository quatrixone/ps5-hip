// One fixed HIP title; startup/exit follow the existing finite shell lifecycle.
#include "native_runtime.h"
#include "elevation.hpp"
#include <cstdlib>
#include <fstream>

#if defined(PS5HIP_ROUTES_TITLE) || defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_WORKERS_MAIN_TITLE)
#include "ps5hip/hip_runtime_control.h"
#define ps5hip_vector_title_registration ps5hip_runtime_registration_ready
#define ps5hip_vector_title_ready ps5hip_runtime_ready
#define ps5hip_vector_title_finalized ps5hip_runtime_finalized
#define ps5hip_vector_terminal ps5hip_runtime_terminal
#ifdef PS5HIP_WORKERS_MAIN_TITLE
#include "hip_workers_client.h"
#include "hip_workers_agc.h"
#define ps5hip_vector_client_main ps5hip_workers_client_main
#elif defined(PS5HIP_HIPIFY_TITLE)
#define ps5hip_vector_client_main hipify_client_main
#elif defined(PS5HIP_MATMUL_TITLE)
#define ps5hip_vector_client_main ps5hip_matmul_client_main
#elif defined(PS5HIP_GENERIC_TITLE)
#define ps5hip_vector_client_main ps5hip_generic_client_main
#else
#define ps5hip_vector_client_main ps5hip_routes_client_main
#endif
#endif
extern "C" bool ps5hip_vector_title_registration() noexcept;
extern "C" bool ps5hip_vector_title_ready() noexcept;
extern "C" bool ps5hip_vector_title_finalized() noexcept;
extern "C" bool ps5hip_vector_terminal();
#ifdef PS5HIP_HIPIFY_TITLE
int ps5hip_vector_client_main();
#else
int ps5hip_vector_client_main(int,char**);
#endif
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
extern "C" const char *ps5hip_title_model_config();
#endif
namespace {
bool finalizer_installed=false;
#ifdef PS5HIP_WORKERS_MAIN_TITLE
[[noreturn]] void quarantine_workers() noexcept {
    // An unjoined worker may hold the runtime gate: never query it here.
    std::fputs("HIP_TITLE QUARANTINED reason=worker-lifecycle no_shell_request=true\n",stderr);
    (void)ps5hip_canary::sync_log();for(;;)::sleep(1);
}
#endif
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
#if defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_WORKERS_MAIN_TITLE)
#ifdef PS5HIP_WORKERS_MAIN_TITLE
    constexpr unsigned expected_completed=5;
#elif defined(PS5HIP_HIPIFY_TITLE)
    constexpr unsigned expected_completed=6;
#elif defined(PS5HIP_MATMUL_TITLE)
    constexpr unsigned expected_completed=8;
#else
    constexpr unsigned expected_completed=11;
#endif
    const auto observation=ps5hip_runtime_observation();
    if(observation.modules || observation.functions || observation.completed!=expected_completed) {
        std::fputs("HIP_TITLE QUARANTINED reason=final-observation no_shell_request=true\n",stderr);
        (void)ps5hip_canary::sync_log();for(;;)::sleep(1);
    }
    std::printf("HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=%u\n",observation.completed);
#elif defined(PS5HIP_ROUTES_TITLE)
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
#ifdef PS5HIP_HIPIFY_TITLE
    (void)argc;(void)argv;
#endif
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
#ifdef PS5HIP_WORKERS_MAIN_TITLE
        const auto log=directory+"/ps5-hip-gfx1013-hip_workers_main.log";
#elif defined(PS5HIP_HIPIFY_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hipify.log";
#elif defined(PS5HIP_MATMUL_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_matmul.log";
#elif defined(PS5HIP_GENERIC_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_generic.log";
#elif defined(PS5HIP_ROUTES_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_routes.log";
#else
        const auto log=directory+"/ps5-hip-gfx1013-hip_vector.log";
#endif
        if(!ps5hip_canary::redirect_log(log.c_str())) return 2;
#ifdef PS5HIP_WORKERS_MAIN_TITLE
        std::puts("PS5HIP START target=gfx1013 kernel=hip_workers_main");
#elif defined(PS5HIP_HIPIFY_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hipify");
#elif defined(PS5HIP_MATMUL_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_matmul");
#elif defined(PS5HIP_GENERIC_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_generic");
#elif defined(PS5HIP_ROUTES_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_routes");
#else
        std::puts("PS5HIP START target=gfx1013 kernel=hip_vector");
#endif
        if(!finalizer_installed || !ps5hip_vector_title_registration()) quarantine("registration");
#if defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_WORKERS_MAIN_TITLE)
#if defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_WORKERS_MAIN_TITLE)
        constexpr unsigned expected_functions=1;
#else
        constexpr unsigned expected_functions=2;
#endif
        const auto registered=ps5hip_runtime_observation();
        if(registered.modules!=1 || registered.functions!=expected_functions || registered.completed || registered.backend_attempted)quarantine("registration-observation");
        std::printf("HIP_TITLE REGISTRATION observed_after_redirect=true modules=%zu functions=%zu constructor_gpu_calls=0\n",registered.modules,registered.functions);
#elif defined(PS5HIP_ROUTES_TITLE)
        std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0");
#endif
        ps5hip_canary::check_runtime();
        if(!persist()) quarantine("startup-log");
#ifdef PS5HIP_WORKERS_MAIN_TITLE
        const int status=ps5hip_vector_client_main(argc,argv);
        const auto &workers=ps5hip_workers::report();
        // A failed join may leave a live caller: do not enter runtime/static teardown.
        if(status || !workers.passed || !workers.cleanup_complete || workers.unsafe_teardown || workers.created!=2 || workers.joined!=2)
            quarantine_workers();
        if(workers.dispatches!=5 || workers.active_values!=661 || workers.word_comparisons!=1355 ||
           workers.template_comparisons!=1355 || workers.phase_checks!=17)quarantine("worker-counts");
        if(!ps5hip_workers::validate_agc_trace())quarantine("worker-agc-attribution");
        ps5hip_runtime_finish();
#elif defined(PS5HIP_HIPIFY_TITLE)
        const int status=ps5hip_vector_client_main();
        if(!status)ps5hip_runtime_finish();
#else
        const int status=ps5hip_vector_client_main(argc,argv);
#endif
        if(status || !ps5hip_vector_title_ready()) quarantine("client-or-ownership");
#ifdef PS5HIP_WORKERS_MAIN_TITLE
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=5)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=5 exact_values=661 word_comparisons=1355 template_comparisons=1355 created=2 joined=2 finalization=pending");
#elif defined(PS5HIP_HIPIFY_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=6)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=6 exact_values=662 word_comparisons=2238 finalization=pending");
#elif defined(PS5HIP_MATMUL_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=8)quarantine("completion-observation");
        std::printf("HIP_TITLE COMPUTE complete=true dispatches=%u cases=4 rounds=2 exact_values=982 finalization=pending\n",completed.completed);
#elif defined(PS5HIP_GENERIC_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=11)quarantine("completion-observation");
        std::printf("HIP_TITLE COMPUTE complete=true dispatches=%u sequences=0,1,3,7 finalization=pending\n",completed.completed);
#elif defined(PS5HIP_ROUTES_TITLE)
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=8 word_comparisons=393216 finalization=pending");
#else
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=2 exact_values=146 finalization=pending");
#endif
        if(!persist()) quarantine("compute-log");
        return 0;
    } catch(const std::exception &e) { quarantine(e.what()); }
    catch(...) { quarantine("unknown exception"); }
}
