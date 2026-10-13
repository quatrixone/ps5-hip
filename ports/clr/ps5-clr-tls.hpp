#pragma once
#include <cstdint>
namespace ps5_clr {
void* current_thread();
void set_current_thread(void* value);
std::uint64_t& correlation_id();
}
