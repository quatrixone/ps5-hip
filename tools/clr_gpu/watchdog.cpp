#include "client.h"
#include "watchdog.h"
#include <pthread.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <time.h>
namespace {
// Persistent storage is retained if completion or join becomes uncertain.
struct State {std::atomic<bool> stop{false};bool started=false,joined=false;uint64_t start=0,deadline=0;pthread_t worker{};} state;
uint64_t now() {
 timespec t{};
 if(clock_gettime(CLOCK_MONOTONIC,&t)||t.tv_sec<0||t.tv_nsec<0||t.tv_nsec>=1000000000L||uint64_t(t.tv_sec)>UINT64_MAX/1000000000ULL-1)
  ps5hip_clr_gpu_fail("watchdog-clock");
 return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);
}
void* worker(void*) {
 uint64_t previous=state.start;
 while(!state.stop.load(std::memory_order_acquire)) {
  const uint64_t current=now();
  if(current<previous)ps5hip_clr_gpu_fail("watchdog-clock-regressed");
  if(current>=state.deadline)ps5hip_clr_gpu_fail("watchdog-timeout");
  previous=current;
  // Relative sleep only bounds polling latency; deadline always uses monotonic time.
  timespec delay{0,10000000};
  if(nanosleep(&delay,nullptr)&&errno!=EINTR)ps5hip_clr_gpu_fail("watchdog-sleep");
 }
 return nullptr;
}
}
namespace ps5hip_clr_gpu {
void watchdog_start(unsigned milliseconds) {
 // The fixed corpus deadline is the only long-run exception; existing callers retain their limits.
 if(state.started||!milliseconds||(milliseconds>120000&&milliseconds!=1800000))ps5hip_clr_gpu_fail("watchdog-state");
 const uint64_t start=now();const uint64_t duration=uint64_t(milliseconds)*1000000ULL;
 if(start>UINT64_MAX-duration)ps5hip_clr_gpu_fail("watchdog-clock");
 state.start=start;state.deadline=start+duration;
 if(pthread_create(&state.worker,nullptr,worker,nullptr))ps5hip_clr_gpu_fail("watchdog-create");
 state.started=true;
}
void watchdog_finish() {
 if(!state.started||state.joined)ps5hip_clr_gpu_fail("watchdog-state");
 const uint64_t before=now();
 if(before<state.start)ps5hip_clr_gpu_fail("watchdog-clock-regressed");
 if(before>=state.deadline)ps5hip_clr_gpu_fail("watchdog-late-finish");
 state.stop.store(true,std::memory_order_release);
 if(pthread_join(state.worker,nullptr))ps5hip_clr_gpu_fail("watchdog-join");
 state.joined=true;
 const uint64_t after=now();
 if(after<before)ps5hip_clr_gpu_fail("watchdog-clock-regressed");
 if(after>=state.deadline)ps5hip_clr_gpu_fail("watchdog-late-join");
}
}
