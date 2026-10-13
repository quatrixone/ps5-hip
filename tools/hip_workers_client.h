#pragma once
#include <cstddef>
namespace ps5hip_workers {
constexpr std::size_t worker_stack_bytes=512*1024;
constexpr unsigned expected_phase_checks=17;
#ifdef PS5HIP_WORKERS_INIT_TITLE
constexpr unsigned warmup_dispatches=0,expected_dispatches=4,expected_values=660,expected_words=1084;
constexpr bool worker_initializes=true;
#else
constexpr unsigned warmup_dispatches=1,expected_dispatches=5,expected_values=661,expected_words=1355;
constexpr bool worker_initializes=false;
#endif
struct Report {
    bool passed=false,cleanup_complete=false,unsafe_teardown=false;
    int error=0;
    unsigned hip_error=0,created=0,joined=0,dispatches=0,active_values=0;
    unsigned word_comparisons=0,template_comparisons=0,phase_checks=0;
    bool prebackend_verified=false;
};
// One invocation; backing stays alive when join/synchronization is uncertain.
// Never finishes the runtime: the title owner does so only after a passed report.
Report run() noexcept;
const Report &report() noexcept;
// Actual pthread identity: main=0, A=1, B=2, unknown=3. No HIP/TLS/locking.
unsigned actor() noexcept;
}
int ps5hip_workers_client_main(int,char**);
