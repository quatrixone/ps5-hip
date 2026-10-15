#include "ps5-embedded.h"
#include <cstring>
#include <algorithm>
#include <cstdint>
extern "C" {
extern const unsigned char __ps5hip_module_spans_begin[] __attribute__((weak));
extern const unsigned char __ps5hip_module_spans_end[] __attribute__((weak));
extern const unsigned char __ps5hip_wrappers_begin[] __attribute__((weak));
extern const unsigned char __ps5hip_wrappers_end[] __attribute__((weak));
extern const unsigned char __ps5hip_bundle_begin[] __attribute__((weak));
extern const unsigned char __ps5hip_bundle_end[] __attribute__((weak));
extern const unsigned char __ps5hip_names_begin[] __attribute__((weak));
extern const unsigned char __ps5hip_names_end[] __attribute__((weak));
}
namespace ps5hip {
namespace {
ReadableSpan span(const unsigned char* first,const unsigned char* last,size_t limit) {
 const auto lo=reinterpret_cast<uintptr_t>(first),hi=reinterpret_cast<uintptr_t>(last);
 if(!lo || hi<lo || hi-lo>limit) throw ObjectError("absent or invalid linked host span");
 return {first,size_t(hi-lo)};
}
}
BundledCodeObject read_linked_host_module(const void* wrapper) {
 const auto table=span(__ps5hip_module_spans_begin,__ps5hip_module_spans_end,32*32);
 const auto wrappers=span(__ps5hip_wrappers_begin,__ps5hip_wrappers_end,64u*1024u*1024u);
 const auto bundles=span(__ps5hip_bundle_begin,__ps5hip_bundle_end,64u*1024u*1024u);
 const auto selected=select_host_module(wrapper,table,wrappers,bundles);
 return read_single_host_wrapper(wrapper,selected.wrapper,selected.bundle,"gfx1013");
}
std::string linked_host_name(const char* name) {
 const auto names=span(__ps5hip_names_begin,__ps5hip_names_end,64u*1024u*1024u);
 const auto base=reinterpret_cast<uintptr_t>(names.data),at=reinterpret_cast<uintptr_t>(name);
 if(at<base || at-base>=names.size) throw ObjectError("kernel name outside linked immutable data");
 const auto available=std::min(size_t(4096),names.size-size_t(at-base));
 const auto end=static_cast<const char*>(std::memchr(name,0,available));
 if(!end || end==name) throw ObjectError("empty or unbounded linked kernel name");
 return std::string(name,size_t(end-name));
}
}
