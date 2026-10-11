#include "ps5hip/agc_backend.h"
#include <ps5platform/agc.h>
#include <iostream>

namespace {
int init_error = 0;
unsigned init_calls = 0, submit_calls = 0;
void check(bool ok) { if (!ok) throw std::runtime_error("AGC boundary assertion failed"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid AGC operation accepted");
}
}
extern "C" int32_t sceAgcInit(uint32_t version) { check(version == 8); ++init_calls; return init_error; }
extern "C" int32_t sceAgcDriverSubmitDcb(ps5_agc_submit_description *description) {
    ++submit_calls;
    check(description->word_count == 1 && description->words && !description->flag);
    check(!description->padding[0] && !description->padding[1] && !description->padding[2]);
    return -17;
}
extern "C" int32_t sceAgcSuspendPoint() { return -23; }
int main(int argc, char **) {
    try {
        if (argc == 2) {
            init_error = -1;
            reject([] { ps5hip::AgcBackend backend; });
        } else {
            ps5hip::AgcBackend backend;
            uint32_t word = 0;
            check(backend.submit(&word, 1) == -17 && backend.kick() == -23);
            reject([&] { backend.submit(nullptr, 1); });
            reject([&] { backend.submit(&word, 0); });
            reject([&] { backend.submit(&word, 16395); });
            backend.flush(nullptr, 0);
            reject([&] { backend.flush(nullptr, 1); });
            reject([&] { backend.flush(reinterpret_cast<void *>(UINTPTR_MAX - 8), 8); });
            check(backend.now_ns() > 0);
            backend.sleep_us(0);
            check(submit_calls == 1);
        }
        reject([] { ps5hip::AgcBackend backend; });
        check(init_calls == 1);
        std::cout << "AGC model boundary: initialization, ABI and error propagation passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
