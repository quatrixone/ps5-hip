#pragma once
#include "layer_qualification.h"
namespace m0 {
// Explicit synthetic fixture only. General placement/full-model support is unchanged.
void configure_dense_layer(ggml_backend_t,const std::array<const char*,6>&);
void execute_dense_layer(ggml_backend_t,ggml_cgraph*);
}
