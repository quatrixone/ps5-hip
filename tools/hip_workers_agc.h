#pragma once
#include <cstdint>
#include "hip_workers_client.h"
namespace ps5hip_workers {
struct AgcEvent { unsigned operation, actor; std::int32_t result; };
constexpr unsigned agc_event_count=1+2*expected_dispatches;
struct AgcTrace { AgcEvent events[agc_event_count]; unsigned count; bool overflow, actor_changed; };
const AgcTrace &agc_trace() noexcept;
bool validate_agc_trace() noexcept;
}
