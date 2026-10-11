#include "ps5hip/agc_backend.h"
#include <ps5platform/agc.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <ctime>

namespace ps5hip {

static_assert(sizeof(ps5_agc_submit_description) == 16 &&
              offsetof(ps5_agc_submit_description, word_count) == 8 &&
              offsetof(ps5_agc_submit_description, flag) == 12, "unexpected native AGC submission ABI");

AgcBackend::AgcBackend() {
    static std::atomic_flag attempted = ATOMIC_FLAG_INIT;
    if (attempted.test_and_set()) throw ObjectError("AGC backend already attempted in this process");
    const int32_t result = sceAgcInit(PS5_AGC_INIT_VERSION);
    if (result) {
        char message[80];
        std::snprintf(message, sizeof(message), "sceAgcInit failed: 0x%08x", unsigned(result));
        throw ObjectError(message);
    }
}

void AgcBackend::flush(void *address, size_t bytes) {
    if (!bytes) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
    constexpr uintptr_t limit = uintptr_t(1) << 47;
    if (!begin || begin >= limit || bytes > limit - begin) throw ObjectError("invalid cache flush range");
    const uintptr_t end = begin + bytes;
    __asm__ volatile("mfence" ::: "memory");
    for (uintptr_t line = begin & ~uintptr_t(63); line < end; line += 64)
        __asm__ volatile("clflushopt %0" : "+m"(*reinterpret_cast<volatile char *>(line)) :: "memory");
    __asm__ volatile("mfence" ::: "memory");
}

int32_t AgcBackend::submit(uint32_t *words, uint32_t count) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(words);
    constexpr uintptr_t limit = uintptr_t(1) << 47;
    if (!count || count > 16394 || !address || address % 4 || address >= limit ||
        uint64_t(count) * 4 > limit - address) throw ObjectError("invalid AGC command range");
    ps5_agc_submit_description description{};
    description.words = words;
    description.word_count = count;
    return sceAgcDriverSubmitDcb(&description);
}

int32_t AgcBackend::kick() { return sceAgcSuspendPoint(); }

uint64_t AgcBackend::now_ns() {
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) || now.tv_sec < 0)
        throw ObjectError("monotonic clock failed");
    return uint64_t(now.tv_sec) * 1000000000 + uint64_t(now.tv_nsec);
}

void AgcBackend::sleep_us(unsigned us) {
    const timespec delay{time_t(us / 1000000), long(us % 1000000) * 1000};
    if (nanosleep(&delay, nullptr) && errno != EINTR) throw ObjectError("completion poll sleep failed");
}

} // namespace ps5hip
