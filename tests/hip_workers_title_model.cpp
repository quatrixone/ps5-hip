// Wrapper/AGC-forwarding fault model only; genuine runtime/client have separate proofs.
#include "hip_workers_client.h"
#include "hip_workers_agc.h"
#include "ps5hip/hip_runtime_control.h"
#include "elevation.hpp"
#include <ps5platform/agc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
int workers_title_main(int,char**);
extern "C" void ps5hip_install_vector_title_finalizer();
extern "C" int32_t __wrap_sceAgcInit(uint32_t);
extern "C" int32_t __wrap_sceAgcDriverSubmitDcb(ps5_agc_submit_description*);
extern "C" int32_t __wrap_sceAgcSuspendPoint();
namespace {
const char *scenario,*config;unsigned finish_calls=0,current_actor=0,forwards=0;
bool finalized=false;ps5hip_workers::Report report_data;
using namespace ps5hip_workers;
bool is(const char*s){return std::strcmp(scenario,s)==0;}
void require(bool condition){if(!condition)::_exit(91);}
void shell(){require(finalized);std::puts("PS5HIP SHELL_EXIT application-fini=completed\nPS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0");}
void cleanup(){
    require(finish_calls==1);
    if(is("skip-unregister"))return;
    std::puts("VECTOR_UNREGISTER after_main=true active=0");
    if(is("retained"))return;
    finalized=true;
    std::printf("VECTOR_FINAL registered=1 removed=1 completed=%u active=0 terminal=0 main_finished=1 hardware_validated=false\n",expected_dispatches);
}
void shutdown(){std::puts("PS5HIP SHUTDOWN atexit=observed\nPS5HIP SHUTDOWN static-dtor=observed");}
}
namespace elevation {Status request(Capability,const char*)noexcept{return is("elevation") ? Status::unavailable:Status::ok;}}
namespace ps5hip_workers {const Report &report()noexcept{return report_data;}unsigned actor()noexcept{return current_actor;}}
extern "C" const char *ps5hip_title_model_config(){return config;}
extern "C" bool ps5hip_runtime_registration_ready()noexcept{return !is("registration");}
extern "C" bool ps5hip_runtime_ready()noexcept{return finish_calls==1&&!is("finish-rejected");}
extern "C" bool ps5hip_runtime_finalized()noexcept{return finalized;}
extern "C" bool ps5hip_runtime_terminal(){require(!is("join")&&!is("unsafe"));return false;}
extern "C" void ps5hip_runtime_finish(){++finish_calls;}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation()noexcept{
    return {finalized?0u:1u,finalized?0u:1u,finish_calls&&is("completion")?expected_dispatches-1:report_data.dispatches,false};
}
extern "C" int32_t __real_sceAgcInit(uint32_t version){require(version==8);++forwards;return is("agc-return")?-1:0;}
extern "C" int32_t __real_sceAgcDriverSubmitDcb(ps5_agc_submit_description *p){require(p==nullptr);++forwards;return 0;}
extern "C" int32_t __real_sceAgcSuspendPoint(){++forwards;return 0;}
extern "C" unsigned sleep(unsigned){
    const bool finished=is("finish-rejected")||is("skip-unregister")||is("retained")||is("completion");
    require(finish_calls==unsigned(finished));
    std::printf("WORKERS_TITLE_MODEL STOP finish_calls=%u shell_requests=0 model_only=true\n",finish_calls);std::fflush(nullptr);::_exit(0);
}
int ps5hip_workers_client_main(int,char**){
    report_data={true,true,false,0,0,2,2,expected_dispatches,expected_values,expected_words,expected_words,17,worker_initializes};
    if(is("prebackend"))report_data.prebackend_verified=false;
    if(is("join")){report_data.joined=1;report_data.unsafe_teardown=true;return 1;}
    if(is("unsafe"))report_data.unsafe_teardown=true;
    if(is("cleanup"))report_data.cleanup_complete=false;
    if(is("created"))report_data.created=1;
    if(is("phase"))report_data.phase_checks=16;
    if(is("count"))report_data.active_values=expected_values-1;
    if(is("final-count"))report_data.dispatches=expected_dispatches-1;
    if(worker_initializes)std::puts("WORKERS_PREBACKEND actor=A completed=0 backend_attempted=false");
    current_actor=worker_initializes&&!is("init-main")?1:0;__wrap_sceAgcInit(8);
    constexpr unsigned actors[]={0,1,2,1,0},counts[]={1,73,257,73,257},salts[]={11,101,1009,65521,4294967280u};
    for(unsigned step=0;step<expected_dispatches;++step){
        const unsigned i=step+(worker_initializes?1u:0u);
        current_actor=is("agc-actor")?3:actors[i];
        __wrap_sceAgcDriverSubmitDcb(nullptr);if(!is("agc-missing")||step+1<expected_dispatches)__wrap_sceAgcSuspendPoint();
        std::printf("RUNTIME_COMPLETE sequence=%u kernel=thread_words blocking=true\nWORKERS_STEP index=%u actor=%s count=%u salt=%u exact=true guards=true template=true\n",step+1,step,actors[i]==0?"main":actors[i]==1?"A":"B",counts[i],salts[i]);
    }
    if(is("agc-extra"))__wrap_sceAgcSuspendPoint();
    require(forwards==agc_event_count+unsigned(is("agc-extra"))-unsigned(is("agc-missing")));
    std::printf("WORKERS_CLIENT_PASS created=2 joined=2 dispatches=%u exact_values=%u word_comparisons=%u template_comparisons=%u phase_checks=17\n",expected_dispatches,expected_values,expected_words,expected_words);return 0;
}
int main(int argc,char**argv){
    if(argc!=3)return 91;scenario=argv[1];config=argv[2];
    std::atexit(shell);ps5hip_install_vector_title_finalizer();std::atexit(cleanup);std::atexit(shutdown);
    const auto status=workers_title_main(0,nullptr);
    if(is("elevation")||is("config")||is("log-exists")){require(status==2&&finish_calls==0);::_exit(0);}
    require(status==0&&finish_calls==1);std::puts("PS5HIP SHUTDOWN main-return=0");
}
