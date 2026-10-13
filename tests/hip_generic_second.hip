// Deliberate unsupported second translation unit. Its genuine compiler-generated
// constructor must latch partial registration; it must never produce GPU work.
#include <hip/hip_runtime.h>
extern "C" __global__ __launch_bounds__(64)
void second_module_kernel(unsigned *output) {
#if defined(__HIP_DEVICE_COMPILE__)
    if(!__builtin_amdgcn_workitem_id_x())*output=17;
#endif
}
