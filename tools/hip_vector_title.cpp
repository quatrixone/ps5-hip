// One fixed HIP title; startup/exit follow the existing finite shell lifecycle.
#if defined(PS5HIP_MEMORY_FILL_TITLE) || defined(PS5HIP_DEVICE_COPY_TITLE) || defined(PS5HIP_MULTI_QUERY_TITLE)
#define PS5HIP_GENERIC_TITLE 1
#endif
#include "native_runtime.h"
#include "elevation.hpp"
#include <cstdlib>
#include <fstream>

#if defined(PS5HIP_REDUCTION_TITLE) || defined(PS5HIP_ROUTES_TITLE) || defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || (defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE))
#include "ps5hip/hip_runtime_control.h"
#define ps5hip_vector_title_registration ps5hip_runtime_registration_ready
#define ps5hip_vector_title_ready ps5hip_runtime_ready
#define ps5hip_vector_title_finalized ps5hip_runtime_finalized
#define ps5hip_vector_terminal ps5hip_runtime_terminal
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
#include "hip_workers_client.h"
#include "hip_workers_agc.h"
#define ps5hip_vector_client_main ps5hip_workers_client_main
#elif defined(PS5HIP_REDUCTION_TITLE)
#define ps5hip_vector_client_main ps5hip_reduction_companion_main
#elif defined(PS5HIP_HIPIFY_TITLE)
#define ps5hip_vector_client_main hipify_client_main
#elif defined(PS5HIP_MATMUL_TITLE)
#define ps5hip_vector_client_main ps5hip_matmul_client_main
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
#include "hip_multi_query_client.h"
#define ps5hip_vector_client_main ps5hip_multi_query_client_main
#elif defined(PS5HIP_DEVICE_COPY_TITLE)
#define ps5hip_vector_client_main ps5hip_device_copy_client_main
#elif defined(PS5HIP_MEMORY_FILL_TITLE)
#define ps5hip_vector_client_main ps5hip_memory_fill_client_main
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
#if defined(PS5HIP_REDUCTION_TITLE)
extern "C" int ps5hip_vector_client_main();
#elif defined(PS5HIP_HIPIFY_TITLE)
int ps5hip_vector_client_main();
#else
int ps5hip_vector_client_main(int,char**);
#endif
#if defined(PS5HIP_VECTOR_MODEL) || defined(PS5HIP_RUNTIME_MODEL)
extern "C" const char *ps5hip_title_model_config();
#endif
namespace {
bool finalizer_installed=false;
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
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
#if defined(PS5HIP_REDUCTION_TITLE) || defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || (defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE))
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
    constexpr unsigned expected_completed=ps5hip_workers::expected_dispatches;
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
    constexpr unsigned expected_completed=3;
#elif defined(PS5HIP_DEVICE_COPY_TITLE)
    constexpr unsigned expected_completed=24;
#elif defined(PS5HIP_REDUCTION_TITLE)
    constexpr unsigned expected_completed=3;
#elif defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_MEMORY_FILL_TITLE)
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
#ifdef PS5HIP_REDUCTION_TITLE
extern "C" int ps5hip_reduction_title_main(int,char**);
#define main ps5hip_reduction_title_main
#endif
int main(int argc,char **argv) {
#if defined(PS5HIP_HIPIFY_TITLE) || defined(PS5HIP_REDUCTION_TITLE)
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
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
        const auto log=directory+(ps5hip_workers::worker_initializes ? "/ps5-hip-gfx1013-hip_workers_init.log":"/ps5-hip-gfx1013-hip_workers_main.log");
#elif defined(PS5HIP_REDUCTION_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_reduction.log";
#elif defined(PS5HIP_HIPIFY_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hipify.log";
#elif defined(PS5HIP_MATMUL_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_matmul.log";
#elif defined(PS5HIP_DEVICE_COPY_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_device_copy.log";
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_multi_query.log";
#elif defined(PS5HIP_MEMORY_FILL_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_memory_fill.log";
#elif defined(PS5HIP_GENERIC_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_generic.log";
#elif defined(PS5HIP_ROUTES_TITLE)
        const auto log=directory+"/ps5-hip-gfx1013-hip_routes.log";
#else
        const auto log=directory+"/ps5-hip-gfx1013-hip_vector.log";
#endif
        if(!ps5hip_canary::redirect_log(log.c_str())) return 2;
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
        std::puts(ps5hip_workers::worker_initializes ? "PS5HIP START target=gfx1013 kernel=hip_workers_init":"PS5HIP START target=gfx1013 kernel=hip_workers_main");
#elif defined(PS5HIP_REDUCTION_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_reduction");
#elif defined(PS5HIP_HIPIFY_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hipify");
#elif defined(PS5HIP_MATMUL_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_matmul");
#elif defined(PS5HIP_DEVICE_COPY_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_device_copy");
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_multi_query");
#elif defined(PS5HIP_MEMORY_FILL_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_memory_fill");
#elif defined(PS5HIP_GENERIC_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_generic");
#elif defined(PS5HIP_ROUTES_TITLE)
        std::puts("PS5HIP START target=gfx1013 kernel=hip_routes");
#else
        std::puts("PS5HIP START target=gfx1013 kernel=hip_vector");
#endif
        if(!finalizer_installed || !ps5hip_vector_title_registration()) quarantine("registration");
#if defined(PS5HIP_REDUCTION_TITLE) || defined(PS5HIP_GENERIC_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || (defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE))
#if defined(PS5HIP_REDUCTION_TITLE) || defined(PS5HIP_MATMUL_TITLE) || defined(PS5HIP_HIPIFY_TITLE) || (defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE))
        constexpr unsigned expected_functions=1;
#else
        constexpr unsigned expected_functions=2;
#endif
#ifdef PS5HIP_MULTI_QUERY_TITLE
        constexpr unsigned expected_modules=2;
#else
        constexpr unsigned expected_modules=1;
#endif
        const auto registered=ps5hip_runtime_observation();
        if(registered.modules!=expected_modules || registered.functions!=expected_functions || registered.completed || registered.backend_attempted)quarantine("registration-observation");
        std::printf("HIP_TITLE REGISTRATION observed_after_redirect=true modules=%zu functions=%zu constructor_gpu_calls=0\n",registered.modules,registered.functions);
#elif defined(PS5HIP_ROUTES_TITLE)
        std::puts("HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0");
#endif
        ps5hip_canary::check_runtime();
        if(!persist()) quarantine("startup-log");
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
        const int status=ps5hip_vector_client_main(argc,argv);
        const auto &workers=ps5hip_workers::report();
        // A failed join may leave a live caller: do not enter runtime/static teardown.
        if(status || !workers.passed || !workers.cleanup_complete || workers.unsafe_teardown || workers.created!=2 || workers.joined!=2)
            quarantine_workers();
        if(workers.dispatches!=ps5hip_workers::expected_dispatches || workers.active_values!=ps5hip_workers::expected_values || workers.word_comparisons!=ps5hip_workers::expected_words ||
           workers.template_comparisons!=ps5hip_workers::expected_words || workers.phase_checks!=17 || (ps5hip_workers::worker_initializes&&!workers.prebackend_verified))quarantine("worker-counts");
        if(!ps5hip_workers::validate_agc_trace())quarantine("worker-agc-attribution");
        ps5hip_runtime_finish();
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
        const int status=ps5hip_vector_client_main(argc,argv);
        const auto &multi=ps5hip_multi_query::report();
        if(status || !multi.passed || !multi.cleanup_complete || multi.dispatches!=3 || multi.active_values!=219 ||
           multi.comparisons!=522 || multi.guards!=84 || multi.queries!=32 || multi.rejections!=8 || multi.allocated!=2 || multi.freed!=2)
            quarantine("multi-client-or-ownership");
        ps5hip_runtime_finish();
#elif defined(PS5HIP_REDUCTION_TITLE)
        const int status=ps5hip_vector_client_main(); // Companion owns its one finish.
#elif defined(PS5HIP_HIPIFY_TITLE)
        const int status=ps5hip_vector_client_main();
        if(!status)ps5hip_runtime_finish();
#else
        const int status=ps5hip_vector_client_main(argc,argv);
#endif
        if(status || !ps5hip_vector_title_ready()) quarantine("client-or-ownership");
#if defined(PS5HIP_WORKERS_MAIN_TITLE) || defined(PS5HIP_WORKERS_INIT_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=ps5hip_workers::expected_dispatches)quarantine("completion-observation");
        std::printf("HIP_TITLE COMPUTE complete=true dispatches=%u exact_values=%u word_comparisons=%u template_comparisons=%u created=2 joined=2 finalization=pending\n",workers.dispatches,workers.active_values,workers.word_comparisons,workers.template_comparisons);
#elif defined(PS5HIP_MULTI_QUERY_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=3 || completed.modules!=2 || completed.functions!=2)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=3 exact_values=219 comparisons=522 guards=84 queries=32 rejected=8 finalization=pending");
#elif defined(PS5HIP_REDUCTION_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=3)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=3 exact_values=48 guards=72 finalization=pending");
#elif defined(PS5HIP_HIPIFY_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=6)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=6 exact_values=662 word_comparisons=2238 finalization=pending");
#elif defined(PS5HIP_MATMUL_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=8)quarantine("completion-observation");
        std::printf("HIP_TITLE COMPUTE complete=true dispatches=%u cases=4 rounds=2 exact_values=982 finalization=pending\n",completed.completed);
#elif defined(PS5HIP_DEVICE_COPY_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=24)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=24 copies=12 exact_values=2648 readback_words=5926 finalization=pending");
#elif defined(PS5HIP_MEMORY_FILL_TITLE)
        const auto completed=ps5hip_runtime_observation();
        if(completed.completed!=6)quarantine("completion-observation");
        std::puts("HIP_TITLE COMPUTE complete=true dispatches=6 exact_values=662 output_words=746 input_words=746 fills=24 finalization=pending");
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
