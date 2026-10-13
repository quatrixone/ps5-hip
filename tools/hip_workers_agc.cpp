// Experiment-only forwarding: the runtime gate serializes these entries.
#include "hip_workers_agc.h"
#include "hip_workers_client.h"
#include <ps5platform/agc.h>
#include <cstdio>
#include <type_traits>
extern "C" std::int32_t __real_sceAgcInit(std::uint32_t);
extern "C" std::int32_t __real_sceAgcDriverSubmitDcb(ps5_agc_submit_description *);
extern "C" std::int32_t __real_sceAgcSuspendPoint();
static_assert(std::is_same_v<decltype(&sceAgcInit),decltype(&__real_sceAgcInit)>);
static_assert(std::is_same_v<decltype(&sceAgcDriverSubmitDcb),decltype(&__real_sceAgcDriverSubmitDcb)>);
static_assert(std::is_same_v<decltype(&sceAgcSuspendPoint),decltype(&__real_sceAgcSuspendPoint)>);
namespace {
ps5hip_workers::AgcTrace trace{};
void record(unsigned operation,unsigned actor,std::int32_t result) noexcept {
    if(actor!=ps5hip_workers::actor())trace.actor_changed=true;
    if(trace.count<ps5hip_workers::agc_event_count)trace.events[trace.count++]={operation,actor,result};
    else trace.overflow=true;
}
}
extern "C" std::int32_t __wrap_sceAgcInit(std::uint32_t version) {
    const auto actor=ps5hip_workers::actor();
    const auto result=__real_sceAgcInit(version);record(0,actor,result);return result;
}
extern "C" std::int32_t __wrap_sceAgcDriverSubmitDcb(ps5_agc_submit_description *description) {
    const auto actor=ps5hip_workers::actor();
    const auto result=__real_sceAgcDriverSubmitDcb(description);record(1,actor,result);return result;
}
extern "C" std::int32_t __wrap_sceAgcSuspendPoint() {
    const auto actor=ps5hip_workers::actor();
    const auto result=__real_sceAgcSuspendPoint();record(2,actor,result);return result;
}
namespace ps5hip_workers {
const AgcTrace &agc_trace() noexcept {return trace;}
bool validate_agc_trace() noexcept {
#ifdef PS5HIP_WORKERS_INIT_TITLE
    constexpr unsigned actors[]={1,1,1,2,2,1,1,0,0};
#else
    constexpr unsigned actors[]={0,0,0,1,1,2,2,1,1,0,0};
#endif
    if(trace.count!=agc_event_count || trace.overflow || trace.actor_changed)return false;
    for(unsigned i=0;i<agc_event_count;++i) {
        const auto &event=trace.events[i];
        if(event.operation!=(i ? (i%2 ? 1u:2u):0u) || event.actor!=actors[i] || event.result!=0)return false;
    }
    for(unsigned i=0;i<agc_event_count;++i) {
        const auto &event=trace.events[i];
        std::printf("WORKERS_AGC ordinal=%u operation=%s actor=%u result=0x%08x\n",i,
            event.operation==0 ? "init":event.operation==1 ? "submit":"kick",event.actor,unsigned(event.result));
    }
    return true;
}
}
