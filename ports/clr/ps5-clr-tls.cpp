#include "ps5-clr-tls.hpp"
#include <pthread.h>
#include <cstdlib>
#include <new>

namespace ps5_clr {
namespace {
struct State { void* current; std::uint64_t correlation; };
pthread_once_t once = PTHREAD_ONCE_INIT;
pthread_key_t key;
int key_error;
void destroy(void* state) { delete static_cast<State*>(state); }
void initialize() { key_error = pthread_key_create(&key, destroy); }
State& state() {
  // CLR's pointer/reference accessors cannot report allocation failure.
  if (pthread_once(&once, initialize) != 0 || key_error != 0) std::abort();
  auto* value = static_cast<State*>(pthread_getspecific(key));
  if (!value) {
    value = new (std::nothrow) State{};
    if (!value) std::abort();
    if (pthread_setspecific(key, value) != 0) {
      delete value;
      std::abort();
    }
  }
  return *value;
}
}
void* current_thread() { return state().current; }
void set_current_thread(void* value) { state().current = value; }
std::uint64_t& correlation_id() { return state().correlation; }
}
