#include "ps5-clr-tls.hpp"
#include <pthread.h>
#include <cstdlib>
#include <new>

namespace ps5_clr {
namespace {
struct State { void* current; std::uint64_t correlation; void* owned; void (*destroy_owned)(void*); };
pthread_once_t once = PTHREAD_ONCE_INIT;
pthread_key_t key;
int key_error;
void destroy(void* state) {
  auto* value = static_cast<State*>(state);
  // Do not access TLS from a pthread key destructor: the key is already cleared.
  if (value->destroy_owned) value->destroy_owned(value->owned);
  delete value;
}
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
void adopt_current_thread(void* value, void (*destroy_owned)(void*)) {
  auto& s = state();
  if (!value || s.current != value || s.owned || !destroy_owned) std::abort();
  s.owned = value;
  s.destroy_owned = destroy_owned;
}
std::uint64_t& correlation_id() { return state().correlation; }
}
