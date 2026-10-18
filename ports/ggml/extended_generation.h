#pragma once
#include <cstddef>
#include <initializer_list>
namespace ps5hip_generation {
struct ExtendedCase {const char* id;const char* prompt;size_t prompt_count,budget,generated;const char* stop;std::initializer_list<unsigned> checkpoints;};
inline constexpr ExtendedCase extended_cases[]={
#define PS5HIP_EXTENDED_CASE(id,prompt,count,budget,generated,stop,...) {id,prompt,count,budget,generated,stop,{__VA_ARGS__}},
#include "extended_generation_cases.def"
#undef PS5HIP_EXTENDED_CASE
};
static_assert(sizeof(extended_cases)/sizeof(extended_cases[0])==3);
inline bool extended_checkpoint(const ExtendedCase& selected,unsigned position){for(auto value:selected.checkpoints)if(position==value)return true;return false;}
}
