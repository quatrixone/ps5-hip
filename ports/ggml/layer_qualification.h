#pragma once
#include "ps5_backend.h"
#include "prefix_calls.h"
#include <array>
namespace ps5hip_layer { struct Graph; }
namespace m0 {
struct LayerRequest { int32_t token,position; };
struct LayerStatistics {
 uint64_t invocations=0, uploaded_weight_bytes=0, uploaded_request_bytes=0, cache_resets=0;
 unsigned completed_launches=0, request_cursor=0;
 int32_t last_token=-1,last_position=-1;
};
// A layer session retains immutable weights and its own GPU KV cache across scopes.
// The original full model graph is still rejected; no CPU activation fallback.
void reset_layer_cache(ggml_backend_t);
void configure_layer(ggml_backend_t,const std::array<const char*,6>&);
ps5hip_prefix::CallStatistics disable_layer(ggml_backend_t);
LayerStatistics layer_statistics(ggml_backend_t);
ps5hip_prefix::CallStatistics layer_call_statistics(ggml_backend_t);
struct ActualLayerQualification {
 using Readback=void (*)(const ps5hip_layer::Graph&,const LayerRequest&,void*);
 ggml_backend_t backend;const LayerRequest* requests;size_t count;Readback readback;void* user;
 ActualLayerQualification(ggml_backend_t,const LayerRequest*,size_t,Readback,void*);
 ~ActualLayerQualification();
 ActualLayerQualification(const ActualLayerQualification&)=delete;
 ActualLayerQualification& operator=(const ActualLayerQualification&)=delete;
};
}
