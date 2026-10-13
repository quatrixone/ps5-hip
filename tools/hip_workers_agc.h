#pragma once
#include <cstdint>
namespace ps5hip_workers {
struct AgcEvent { unsigned operation, actor; std::int32_t result; };
struct AgcTrace { AgcEvent events[11]; unsigned count; bool overflow, actor_changed; };
const AgcTrace &agc_trace() noexcept;
bool validate_agc_trace() noexcept;
}
