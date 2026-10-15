#pragma once
#include "ps5hip/host_registration.h"
namespace ps5hip {
BundledCodeObject read_linked_host_module(const void* wrapper);
std::string linked_host_name(const char* name);
}
