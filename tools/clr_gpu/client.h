#pragma once
namespace ps5hip_clr_gpu { void run(); bool passed(); }
extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason);
