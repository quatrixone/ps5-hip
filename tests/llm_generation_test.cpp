#include "generation.h"
#include <cassert>
#include <limits>
#include <vector>
#include <utility>
using namespace ps5hip_generation;
template<class F> void rejects(F f) { bool threw=false;try{f();}catch(const std::exception&){threw=true;}assert(threw); }
int main() {
    std::vector<float> logits(vocabulary_size, -1.f);
    const int32_t prompt[]={10,20};
    std::vector<std::pair<int32_t,int32_t>> calls;
    auto decode=[&](int32_t token,int32_t position) {
        calls.emplace_back(token,position);
        std::fill(logits.begin(),logits.end(),-1.f);
        const int32_t chosen=token==20?7:token==7?8:token==8?9:3;
        logits[chosen]=2.f;return Logits{logits.data(),logits.size()};
    };
    auto never=[](int32_t){return false;};
    auto first=generate(prompt,2,3,decode,never);
    assert((calls==std::vector<std::pair<int32_t,int32_t>>{{10,0},{20,1},{7,2},{8,3}}));
    assert(first.processed_count==4&&first.generated_count==3&&first.generated[0]==7&&first.generated[1]==8&&first.generated[2]==9&&first.stop==StopReason::step_limit);
    calls.clear();auto repeat=generate(prompt,2,3,decode,never);
    assert(first.processed==repeat.processed&&first.generated==repeat.generated); // caller-owned reset of test decoder
    calls.clear();auto eos=generate(prompt,2,3,decode,[](int32_t token){return token==8;});
    assert(eos.stop==StopReason::eos&&eos.generated_count==2&&eos.generated[1]==8&&calls.size()==3&&calls.back().first==7);
    calls.clear();auto early=generate(prompt,2,3,decode,[](int32_t token){return token==7;});
    assert(early.generated_count==1&&calls.size()==2);
    std::fill(logits.begin(),logits.end(),-2.f);logits[3]=logits[31999]=1.f;assert(greedy({logits.data(),logits.size()})==3);
    logits[3]=-2.f;assert(greedy({logits.data(),logits.size()})==31999);
    for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {logits[5]=bad;rejects([&]{greedy({logits.data(),logits.size()});});}logits[5]=-2.f;
    rejects([&]{greedy({nullptr,vocabulary_size});});rejects([&]{greedy({logits.data(),vocabulary_size-1});});
    calls.clear();const int32_t negative[]={-1},large[]={32000};
    rejects([&]{generate(prompt,2,512,decode,never);});rejects([&]{generate(prompt,2,0,decode,never);});
    rejects([&]{generate(prompt,0,1,decode,never);});rejects([&]{generate(nullptr,1,1,decode,never);});
    rejects([&]{generate(negative,1,1,decode,never);});rejects([&]{generate(large,1,1,decode,never);});assert(calls.empty());
    // Full 512-position boundary: 512th emitted token is never submitted at 512.
    const int32_t one[]={0};size_t submitted=0;
    auto boundary=generate(one,1,512,[&](int32_t token,int32_t position){assert(token==0&&position==int32_t(submitted)&&position<512);++submitted;std::fill(logits.begin(),logits.end(),0.f);return Logits{logits.data(),logits.size()};},never);
    assert(submitted==512&&boundary.processed_count==512&&boundary.generated_count==512);
    // Exceptions propagate; caller must reset/quarantine its own decoder state.
    size_t failed=0;rejects([&]{generate(prompt,2,1,[&](int32_t,int32_t)->Logits{++failed;throw std::runtime_error("decode failed");},never);});assert(failed==1);
    return 0;
}
