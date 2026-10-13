// CPU-only race check of the actual client. HIP and shader execution are synthetic.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include "ps5hip/hip_runtime_control.h"
#include "../tools/hip_workers_client.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <cerrno>
#include <pthread.h>
namespace {
struct Configuration {dim3 grid,block;size_t shared;hipStream_t stream;};
thread_local std::vector<Configuration> stack;
thread_local hipError_t error=hipSuccess;
std::mutex gate;
unsigned *allocation=nullptr,completed=0;
size_t allocated=0;
std::atomic<unsigned> creates{0},joins{0};
const char *mode="normal";
void check(bool yes){if(!yes)std::abort();}
hipError_t result(hipError_t value){error=value;return value;}
bool span(const void *pointer,size_t bytes){
    const auto begin=reinterpret_cast<uintptr_t>(allocation),at=reinterpret_cast<uintptr_t>(pointer);
    return allocation&&at>=begin&&at-begin<=allocated&&bytes<=allocated-(at-begin);
}
}
extern "C" hipError_t hipMalloc(void **pointer,size_t bytes){
    std::lock_guard<std::mutex> lock(gate);
    if(!pointer)return result(hipErrorInvalidValue);
    check(!allocation&&bytes==271*sizeof(unsigned));
    allocation=static_cast<unsigned*>(std::malloc(bytes));allocated=bytes;*pointer=allocation;
    return result(allocation?hipSuccess:hipErrorOutOfMemory);
}
extern "C" hipError_t hipFree(void *pointer){
    std::lock_guard<std::mutex> lock(gate);check(pointer==allocation);
    std::free(allocation);allocation=nullptr;allocated=0;return result(hipSuccess);
}
extern "C" hipError_t hipMemcpy(void *destination,const void *source,size_t bytes,hipMemcpyKind kind){
    std::lock_guard<std::mutex> lock(gate);
    check(bytes==271*sizeof(unsigned));
    check(kind==hipMemcpyHostToDevice?span(destination,bytes):kind==hipMemcpyDeviceToHost&&span(source,bytes));
    std::memcpy(destination,source,bytes);return result(hipSuccess);
}
extern "C" hipError_t hipDeviceSynchronize(){return result(hipSuccess);}
extern "C" hipError_t hipPeekAtLastError(){return error;}
extern "C" hipError_t hipGetLastError(){const auto value=error;error=hipSuccess;return value;}
extern "C" hipError_t __hipPushCallConfiguration(dim3 grid,dim3 block,size_t shared,hipStream_t stream){
    stack.push_back({grid,block,shared,stream});return result(hipSuccess);
}
extern "C" hipError_t __hipPopCallConfiguration(dim3 *grid,dim3 *block,size_t *shared,hipStream_t *stream){
    if(stack.empty())return result(hipErrorInvalidValue);
    const auto value=stack.back();stack.pop_back();*grid=value.grid;*block=value.block;*shared=value.shared;*stream=value.stream;
    return result(hipSuccess);
}
extern "C" void threads_launch(unsigned *output,unsigned count,unsigned salt){
    std::lock_guard<std::mutex> lock(gate);
    constexpr unsigned actors[]={0,1,2,1,0};
    check(completed<5&&ps5hip_workers::actor()==actors[completed]&&span(output,count*sizeof(unsigned))&&stack.empty());
    for(unsigned i=0;i<count;++i)output[i]=i*17u+salt;
    ++completed;(void)result(hipSuccess);
}
extern "C" bool ps5hip_runtime_registration_ready() noexcept{return true;}
extern "C" bool ps5hip_runtime_terminal(){return false;}
extern "C" bool ps5hip_runtime_gpu_unresolved(){return false;}
extern "C" ps5hip::HipRuntimeObservation ps5hip_runtime_observation() noexcept{
    std::lock_guard<std::mutex> lock(gate);return {1,1,completed,completed!=0};
}
extern "C" int tsan_client_create(pthread_t *thread,const pthread_attr_t *attributes,void*(*entry)(void*),void *argument) noexcept{
    const auto number=++creates;
    if(!std::strcmp(mode,"partial-create")&&number==2)return EAGAIN;
    return pthread_create(thread,attributes,entry,argument);
}
extern "C" int tsan_client_join(pthread_t thread,void **result){
    const auto number=++joins;const int status=pthread_join(thread,result);
    return !std::strcmp(mode,"join-error")&&number==1?EINVAL:status;
}
int main(int argc,char **argv){
    check(argc==2);mode=argv[1];const auto report=ps5hip_workers::run();
    if(!std::strcmp(mode,"normal"))check(report.passed&&report.cleanup_complete&&!report.unsafe_teardown&&report.created==2&&report.joined==2&&report.dispatches==5&&report.phase_checks==17&&!allocation);
    else if(!std::strcmp(mode,"partial-create"))check(!report.passed&&report.cleanup_complete&&!report.unsafe_teardown&&report.created==1&&report.joined==1&&report.dispatches==1&&!allocation);
    else if(!std::strcmp(mode,"join-error"))check(!report.passed&&!report.cleanup_complete&&report.unsafe_teardown&&report.created==2&&report.joined==1&&report.dispatches==4&&allocation);
    else std::abort();
    check(joins==report.created);
    std::printf("WORKERS_CLIENT_TSAN_PASS mode=%s synthetic_hip=true\n",mode);
    // The join-error injection already joined every real worker; no GPU owner exists.
    std::free(allocation);allocation=nullptr;
}
