#pragma once
#include <cstdint>
namespace ps5hip_dense_fixture {
struct Case {const char* name;unsigned count;float scale;};
inline constexpr Case cases[]={
#define PS5HIP_DENSE_CASE(name,count,scale) {#name,count,scale},
#include "dense_layer_cases.def"
#undef PS5HIP_DENSE_CASE
};
inline float weight(unsigned role,uint64_t index){return float(int((index*17+role*23)%61)-30)*0.000125f;}
inline float norm(uint64_t index){return 1.f+float(int(index%7)-3)*.01f;}
inline float input(unsigned index,unsigned position,float scale){return scale*float(int((index*13+position*7)%73)-36)/36.f;}
inline bool cache_checkpoint(unsigned position){return position==0||position==1||position==255||position==256||position==511;}
}
