#pragma once
namespace ps5hip_multi_query {
struct Report {
    bool passed=false,cleanup_complete=false;
    unsigned error=0,hip_error=0,dispatches=0,active_values=0;
    unsigned comparisons=0,guards=0,queries=0,rejections=0,allocated=0,freed=0;
};
// Single invocation, synchronous kernels; the title owns runtime finish.
Report run() noexcept;
const Report &report() noexcept;
}
int ps5hip_multi_query_client_main(int,char**);
