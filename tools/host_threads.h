#pragma once
#include <cstddef>
namespace ps5hip_threads {
constexpr std::size_t worker_stack_bytes = 512 * 1024;
constexpr unsigned iterations = 128;
struct Report {
    bool passed = false, cleanup_complete = false;
    int error = 0;
    unsigned created = 0, joined = 0, busy = 0, counter = 0, turns = 0;
    unsigned destructors[2]{};
    unsigned main_tls = 0, worker_tls[2]{};
};
// One invocation per process. On uncertain join/cleanup, static backing remains live.
Report run() noexcept;
}
