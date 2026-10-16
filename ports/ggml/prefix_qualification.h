#pragma once
#include "ps5_backend.h"
#include "prefix_calls.h"
#include <array>
namespace m0 {
struct PrefixStatistics { uint64_t invocations=0,uploaded_leaf_bytes=0;unsigned completed_launches=0;int32_t last_token=-1; };
using PrefixCall=ps5hip_prefix::Call;
using PrefixCallStatistics=ps5hip_prefix::CallStatistics;
// Explicit diagnostic configuration: device supports_op remains false for release.
void configure_prefix(ggml_backend_t,const std::array<const char*,5>& module_paths);
PrefixCallStatistics disable_prefix(ggml_backend_t);
PrefixCallStatistics prefix_call_statistics(ggml_backend_t);
PrefixStatistics prefix_statistics(ggml_backend_t);
struct ActualPrefixQualification {
 using Readback=void (*)(ggml_tensor* const*,size_t,int32_t,void*);
 ggml_backend_t backend;Readback readback;void* user;
 ActualPrefixQualification(ggml_backend_t,Readback,void*);
 ~ActualPrefixQualification();
 ActualPrefixQualification(const ActualPrefixQualification&)=delete;
 ActualPrefixQualification& operator=(const ActualPrefixQualification&)=delete;
};
}
