#pragma once
#include "ps5hip/submission.h"
#include <atomic>
#include <memory>
namespace amd { class Device; }
namespace ps5hip::clr {
// Diagnostic observation only; no public HIP capability advertisement.
struct Observation {
    std::atomic<unsigned> allocations{0}, frees{0}, launches{0}, queues{0}, retired_queues{0};
    std::atomic<bool> terminal{false}, uncertain{false}, gate_destroyed{false};
};
using BackendFactory = std::unique_ptr<SubmissionBackend>(*)();
// Factory returns nullptr on failure. Observation must outlive every CLR object.
// Native callers supply the actual AgcBackend factory; models explicitly supply
// synthetic transport. There is no CPU implementation of a GPU kernel here.
amd::Device* create_device(BackendFactory, Observation&) noexcept;
}
