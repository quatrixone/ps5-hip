#pragma once
#include <cstdint>
namespace ps5_clr {
void* current_thread();
void set_current_thread(void* value);
// Only dynamically created HostThread objects are owned; queue Thread pointers are borrowed.
void adopt_current_thread(void* value, void (*destroy)(void*));
std::uint64_t& correlation_id();
}
