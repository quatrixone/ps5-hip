#pragma once
#include "ggml-backend.h"
#include <cstddef>
extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason);
namespace m0 {
constexpr size_t allocation_limit = 64u * 1024u * 1024u;
struct GateStatistics { unsigned calls, forwarded, rejected, direct_graph_calls; };
ggml_backend_reg_t registration();
ggml_backend_dev_t device_handle();
GateStatistics statistics();
void reset_statistics();
void require(bool condition);
// For synthetic standalone diagnostics; real llama calls use mandatory linker wrapper.
ggml_status release_compute(ggml_backend_sched_t, ggml_backend_t, ggml_cgraph*);
}
