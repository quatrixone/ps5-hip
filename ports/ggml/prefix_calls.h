#pragma once
#include <array>
#include <cstdint>
namespace ps5hip_prefix {
enum class Call : unsigned { module_load,module_function,scratch_allocate,token_read,launch,completion_sync,release_sync,scratch_free,module_unload,count };
struct CallCounter { uint64_t attempts=0,completions=0,successes=0,failures=0,requested_bytes=0;int last_result=0,first_error=0; };
using CallStatistics=std::array<CallCounter,static_cast<unsigned>(Call::count)>;
}
