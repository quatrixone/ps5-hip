#pragma once
#include "ps5hip/native_memory.h"
#include <memory>
#include <atomic>

namespace ps5hip {

// Native implementation supplies cache operations and AGC; tests supply a model.
// Driver calls themselves must return; the queue bounds completion polling only.
struct SubmissionBackend {
    SubmissionBackend() = default;
    SubmissionBackend(const SubmissionBackend &) = delete;
    SubmissionBackend &operator=(const SubmissionBackend &) = delete;
    virtual ~SubmissionBackend() = default;
    virtual void flush(void *address, size_t bytes) = 0;
    virtual int32_t submit(uint32_t *words, uint32_t count) = 0;
    virtual int32_t kick() = 0;
    virtual uint64_t now_ns() = 0;
    virtual void sleep_us(unsigned us) = 0;
private:
    friend class SubmissionQueue;
    std::atomic_flag queue_claimed_ = ATOMIC_FLAG_INIT;
};

enum class SubmissionStatus { Complete, SubmitFailed, KickFailed, Timeout, ClockFailed };
struct SubmissionResult { SubmissionStatus status; int32_t driver_error = 0; };

// One serialized submission at a time. Only a matching GPU marker releases holds.
// Backend and caller must outlive the queue. Completion does not prove arithmetic.
class SubmissionQueue {
public:
    explicit SubmissionQueue(SubmissionBackend &backend) : backend_(backend) {
        if (backend_.queue_claimed_.test_and_set()) throw ObjectError("backend queue already claimed");
    }
    SubmissionQueue(const SubmissionQueue &) = delete;
    SubmissionQueue &operator=(const SubmissionQueue &) = delete;
    bool poisoned() const { return poisoned_; }
    SubmissionResult submit_and_wait(const std::vector<uint32_t> &commands,
                                    const std::vector<std::shared_ptr<NativeMemory>> &resources,
                                    uint32_t timeout_ms);
private:
    SubmissionBackend &backend_;
    bool poisoned_ = false;
    std::vector<std::shared_ptr<NativeMemory>> held_;
};

} // namespace ps5hip
