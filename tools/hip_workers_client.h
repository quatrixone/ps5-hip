#pragma once
#include <cstddef>
namespace ps5hip_workers {
constexpr std::size_t worker_stack_bytes=512*1024;
constexpr unsigned expected_phase_checks=17;
struct Report {
    bool passed=false,cleanup_complete=false,unsafe_teardown=false;
    int error=0;
    unsigned hip_error=0,created=0,joined=0,dispatches=0,active_values=0;
    unsigned word_comparisons=0,template_comparisons=0,phase_checks=0;
};
// One invocation; backing stays alive when join/synchronization is uncertain.
// Never finishes the runtime: the title owner does so only after a passed report.
Report run() noexcept;
const Report &report() noexcept;
// Actual pthread identity: main=0, A=1, B=2, unknown=3. No HIP/TLS/locking.
unsigned actor() noexcept;
}
int ps5hip_workers_client_main(int,char**);
