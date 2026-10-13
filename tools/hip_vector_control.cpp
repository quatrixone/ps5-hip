// Exact historical vector contract; shared runtime owns all ABI/state/lifetime.
#include "ps5hip/hip_runtime_control.h"
namespace ps5hip {
const HipRuntimeContract &hip_runtime_contract() noexcept {
    static constexpr HipRuntimeContract contract{1,2,true};return contract;
}
NativeLaunchRequest hip_runtime_request(const Kernel &kernel,std::array<uint32_t,3> groups,
    std::array<uint32_t,3> threads,size_t shared) {
    if(groups!=std::array<uint32_t,3>{2,1,1} || threads!=std::array<uint32_t,3>{64,1,1} || shared ||
       kernel.name!="_Z10vector_addPKfS0_Pfj" || kernel.arguments.size()!=4 || !initial_launch_rejection(kernel).empty())
        throw ObjectError("unmeasured vector ABI or shape");
    return {NativeLaunchRoute::Initial,{groups,threads,1},0};
}
void hip_runtime_validate_arguments(const Kernel &,const NativeLaunchRequest &,const std::vector<LaunchArgument> &values) {
    if(values.size()!=4 || values[3].value!=73)throw ObjectError("unmeasured element count");
    for(size_t i=0;i<3;++i)if(values[i].buffer_bytes<73*4)throw ObjectError("short vector buffer");
}
}
#ifdef PS5HIP_VECTOR_MODEL
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_vector_model_backend();
std::unique_ptr<ps5hip::SubmissionBackend> ps5hip_runtime_model_backend(){return ps5hip_vector_model_backend();}
#endif
extern "C" void ps5hip_vector_finish(){ps5hip_runtime_finish();}
extern "C" bool ps5hip_vector_terminal(){return ps5hip_runtime_terminal();}
#ifdef PS5HIP_VECTOR_MODEL
extern "C" const void *ps5hip_vector_token(){return ps5hip_runtime_token();}
extern "C" bool ps5hip_vector_gpu_unresolved(){return ps5hip_runtime_gpu_unresolved();}
#endif
#ifdef PS5HIP_VECTOR_TITLE
extern "C" bool ps5hip_vector_title_registration() noexcept {return ps5hip_runtime_registration_ready();}
extern "C" bool ps5hip_vector_title_ready() noexcept {return ps5hip_runtime_ready();}
extern "C" bool ps5hip_vector_title_finalized() noexcept {return ps5hip_runtime_finalized();}
#endif
