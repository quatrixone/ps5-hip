// Actual title lifecycle, synthetic client/runtime/evidence; separate trace proof.
#include "hip_contention.h"
#include "ps5hip/hip_runtime_control.h"
#include "elevation.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
int workers_title_main(int,char**);
extern "C" void ps5hip_install_vector_title_finalizer();
namespace {
const char *scenario,*config;unsigned finish_calls=0;bool finalized=false;
ps5hip_contention::Report report_data;
bool is(const char*s){return std::strcmp(scenario,s)==0;}
void require(bool b){if(!b)::_exit(91);}
void shell(){require(finalized);std::puts("PS5HIP SHELL_EXIT application-fini=completed\nPS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0");}
void cleanup(){require(finish_calls==1);if(is("skip-unregister"))return;std::puts("VECTOR_UNREGISTER after_main=true active=0");if(is("retained"))return;finalized=true;std::puts("VECTOR_FINAL registered=1 removed=1 completed=3 active=0 terminal=0 main_finished=1 hardware_validated=false");}
void shutdown(){std::puts("PS5HIP SHUTDOWN atexit=observed\nPS5HIP SHUTDOWN static-dtor=observed");}
}
namespace elevation {Status request(Capability,const char*)noexcept{return is("elevation")?Status::unavailable:Status::ok;}}
namespace ps5hip_contention {
const Report &report()noexcept{return report_data;}
bool evidence(bool print)noexcept {
 require(!is("join")&&!is("unsafe"));if(is("evidence"))return false;
 if(print){std::puts("CONTEND_PROOF api_overlap=1 same_gate_request_while_owned=1 a_launch_calls=1 b_push_calls=1 gate_acquired_a=1 gate_acquired_b=1 kernel_blocked_wait=unproven\nCONTEND_TIME hold_ns=100 request_ns=200 release_ns=300 acquired_ns=400 lock_call_ns=200 includes_scheduling=true");
 for(unsigned i=0;i<7;++i)std::printf("CONTEND_AGC ordinal=%u operation=%s actor=%u result=0x00000000\n",i,i==0?"init":i%2?"submit":"kick",i<3?0:i<5?1:2);}
 return true;
}
}
extern "C" const char *ps5hip_title_model_config(){return config;}
extern "C" bool ps5hip_runtime_registration_ready()noexcept{return !is("registration");}
extern "C" bool ps5hip_runtime_ready()noexcept{return finish_calls==1&&!is("finish-rejected");}
extern "C" bool ps5hip_runtime_finalized()noexcept{return finalized;}
extern "C" bool ps5hip_runtime_terminal(){require(!is("join")&&!is("unsafe"));return false;}
extern "C" void ps5hip_runtime_finish(){++finish_calls;}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation()noexcept{return {finalized?0u:1u,finalized?0u:1u,finish_calls&&is("completion")?2:report_data.dispatches,false};}
extern "C" unsigned sleep(unsigned){
 const bool finished=is("finish-rejected")||is("skip-unregister")||is("retained")||is("completion");require(finish_calls==unsigned(finished));
 std::printf("CONTEND_TITLE_MODEL STOP finish_calls=%u shell_requests=0 model_only=true\n",finish_calls);std::fflush(nullptr);::_exit(0);
}
int ps5hip_contention_client_main(int,char**){
 report_data={true,true,false,2,2,3,331,813,813,0,0};
 if(is("join")){report_data.joined=1;report_data.unsafe_teardown=true;return 1;}
 if(is("unsafe"))report_data.unsafe_teardown=true;
 if(is("cleanup"))report_data.cleanup_complete=false;
 if(is("created"))report_data.created=1;
 if(is("count"))report_data.values=330;
 if(is("final-count"))report_data.dispatches=2;
 for(unsigned i=1;i<=3;++i)std::printf("RUNTIME_COMPLETE sequence=%u kernel=thread_words blocking=true\n",i);
 std::puts("CONTEND_CLIENT_PASS created=2 joined=2 dispatches=3 exact_values=331 word_comparisons=813 template_comparisons=813");return 0;
}
int main(int argc,char**argv){
 if(argc!=3)return 91;scenario=argv[1];config=argv[2];std::atexit(shell);ps5hip_install_vector_title_finalizer();std::atexit(cleanup);std::atexit(shutdown);
 const auto status=workers_title_main(0,nullptr);if(is("elevation")||is("config")||is("log-exists")){require(status==2&&!finish_calls);::_exit(0);}
 require(!status&&finish_calls==1);std::puts("PS5HIP SHUTDOWN main-return=0");
}
