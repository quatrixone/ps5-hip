#pragma once
namespace ps5hip_clr_gpu {
// One-shot deadline, inclusive range 1..120000 ms; finish/join must precede it.
void watchdog_start(unsigned milliseconds);
void watchdog_finish();
}
