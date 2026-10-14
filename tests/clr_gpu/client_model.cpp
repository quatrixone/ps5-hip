// Model-only AGC class implementation. This never links into the native title.
// Genuine CLR client/commands/adapter run unchanged; transport arithmetic is synthetic.
#define main qualified_adapter_model_main
#include "../clr_adapter_model.cpp"
#undef main
#include "ps5hip/agc_backend.h"
#include "../../tools/clr_gpu/client.cpp"
static Transport transport;
static unsigned failed_joins=0;
extern "C" int __real_pthread_join(pthread_t,void**);
extern "C" int __wrap_pthread_join(pthread_t t,void** value){
 if(!std::strcmp(mode,"join-fail")&&ps5hip_clr_gpu::observation.launches==1&&!failed_joins){++failed_joins;return EINVAL;}
 return __real_pthread_join(t,value);
}

namespace ps5hip {
AgcBackend::AgcBackend(){++backend_calls;if(!std::strcmp(mode,"init"))throw std::runtime_error("injected init failure");}
void AgcBackend::flush(void* p,size_t n){transport.flush(p,n);}
int32_t AgcBackend::submit(uint32_t* p,uint32_t n){return transport.submit(p,n);}
int32_t AgcBackend::kick(){return transport.kick();}
uint64_t AgcBackend::now_ns(){return transport.now_ns();}
void AgcBackend::sleep_us(unsigned n){transport.sleep_us(n);}
}
extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason){
 bool expected=false;
 // Before submit_and_wait only code and kernarg backing exist; command/marker allocations never occur.
 if(!std::strcmp(mode,"init"))expected=!std::strcmp(reason,"read-completion-retain-all")&&allocations.size()==2&&releases==2;
 if(!std::strcmp(mode,"noop")||!std::strcmp(mode,"corrupt"))expected=!std::strcmp(reason,"output-or-guard")&&allocations.size()==2&&releases==4;
 if(!std::strcmp(mode,"submit")||!std::strcmp(mode,"throw")||!std::strcmp(mode,"kick")||!std::strcmp(mode,"timeout")||!std::strcmp(mode,"postflush"))expected=!std::strcmp(reason,"read-completion-retain-all")&&allocations.size()==6&&releases==0;
 if(!std::strcmp(mode,"cleanup"))expected=!std::strcmp(reason,"read-completion-retain-all")&&allocations.size()==3&&releases==3;
 if(!std::strcmp(mode,"join-fail"))expected=!std::strcmp(reason,"queue-join-retain-all")&&failed_joins==1&&ps5hip_clr_gpu::joins==0&&ps5hip_clr_gpu::observation.retired_queues==1&&!ps5hip_clr_gpu::observation.gate_destroyed&&!ps5hip_clr_gpu::observation.terminal&&ps5hip_clr_gpu::observation.queues==2&&ps5hip_clr_gpu::observation.allocations==2&&ps5hip_clr_gpu::observation.frees==2&&allocations.empty()&&releases==6;
 if(!expected)std::fprintf(stderr,"CLIENT_RETENTION_DIAGNOSTIC mode=%s reason=%s live=%zu releases=%u submits=%u backend=%u\n",mode,reason,allocations.size(),releases,submits,backend_calls);
 check(expected&&!ps5hip_clr_gpu::passed()&&submits==unsigned(std::strcmp(mode,"init")!=0)&&backend_calls==1);
 std::fprintf(stdout,"CLR_GPU_CLIENT_MODEL_RETAIN mode=%s reason=%s live=%zu releases=%u gpu_execution=false\n",mode,reason,allocations.size(),releases);std::fflush(stdout);std::_Exit(0);
}
int main(int argc,char**argv){
 check(argc==2);mode=argv[1];resume=true;
 ps5hip_clr_gpu::run();
 check(!std::strcmp(mode,"normal")&&ps5hip_clr_gpu::passed()&&allocations.empty()&&releases==6&&submits==unsigned(std::strcmp(mode,"init")!=0)&&backend_calls==1);
 std::puts("CLR_GPU_CLIENT_MODEL_PASS gpu_execution=false exact_values=73 guards=8 releases=6");
}
