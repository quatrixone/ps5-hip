#pragma once
#include "ps5hip/submission.h"

namespace ps5hip {

// Must be constructed from the dedicated title executable, never a loaded
// anonymous payload. The title linker must import libSceAgc/libSceAgcDriver.
// One backend lifetime per process: a poisoned queue cannot be replaced in-place.
class AgcBackend final : public SubmissionBackend {
public:
    AgcBackend();
    AgcBackend(const AgcBackend &) = delete;
    AgcBackend &operator=(const AgcBackend &) = delete;
    void flush(void *address, size_t bytes) override;
    int32_t submit(uint32_t *words, uint32_t count) override;
    int32_t kick() override;
    uint64_t now_ns() override;
    void sleep_us(unsigned us) override;
};

} // namespace ps5hip
