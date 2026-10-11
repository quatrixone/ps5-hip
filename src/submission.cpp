#include "ps5hip/submission.h"
#include <algorithm>
#include <cstring>

namespace ps5hip {

SubmissionResult SubmissionQueue::submit_and_wait(const std::vector<uint32_t> &commands,
                                                 const std::vector<std::shared_ptr<NativeMemory>> &resources,
                                                 uint32_t timeout_ms) {
    if (poisoned_) throw ObjectError("submission queue poisoned; GPU ownership is unresolved");
    if (commands.empty() || commands.size() > 16384 || !timeout_ms || timeout_ms > 10000)
        throw ObjectError("invalid initial command size or timeout");
    auto command_memory = std::make_shared<NativeMemory>((commands.size() + 10) * 4, 256, false);
    auto marker_memory = std::make_shared<NativeMemory>(4, 64, false);
    std::vector<std::shared_ptr<NativeMemory>> pending{command_memory, marker_memory};
    for (const auto &resource : resources) {
        if (!resource) throw ObjectError("null submission resource");
        resource->range();
        if (std::find(pending.begin(), pending.end(), resource) == pending.end()) pending.push_back(resource);
    }
    auto *words = static_cast<uint32_t *>(command_memory->cpu());
    std::memcpy(words, commands.data(), commands.size() * 4);
    const uint64_t marker_address = marker_memory->range().address;
    // Drain compute before the cache-release event can publish completion.
    const uint32_t completion[] = {0xc0004600u, 0x407u, 0xc0064900u, 0x0030c514u, 0x20000000u,
                                  uint32_t(marker_address), uint32_t(marker_address >> 32), 1, 0, 0};
    std::memcpy(words + commands.size(), completion, sizeof(completion));
    auto *marker = static_cast<volatile uint32_t *>(marker_memory->cpu());
    *marker = 0;
    size_t acquired = 0;
    try {
        for (auto &memory : pending) { memory->acquire_gpu_use(); ++acquired; }
        for (auto &memory : pending) backend_.flush(memory->cpu(), memory->range().bytes);
    } catch (...) {
        for (size_t i = 0; i < acquired; ++i) pending[i]->complete_gpu_use();
        throw;
    }
    held_ = std::move(pending);
    // From the instant submit is entered, any error may mean partial consumption.
    poisoned_ = true;
    const int32_t submitted = backend_.submit(words, uint32_t(commands.size() + 10));
    if (submitted) return {SubmissionStatus::SubmitFailed, submitted};
    const int32_t kicked = backend_.kick();
    if (kicked) return {SubmissionStatus::KickFailed, kicked};
    const uint64_t started = backend_.now_ns();
    uint64_t previous = started;
    // A finite poll budget also bounds a broken clock that never advances.
    const uint32_t max_polls = timeout_ms * 10 + 1;
    for (uint32_t poll = 0; poll < max_polls; ++poll) {
        backend_.flush(marker_memory->cpu(), sizeof(uint32_t));
        if (*marker == 1) {
            for (auto &memory : held_) backend_.flush(memory->cpu(), memory->range().bytes);
            for (auto &memory : held_) memory->complete_gpu_use();
            held_.clear();
            poisoned_ = false;
            return {SubmissionStatus::Complete};
        }
        const uint64_t now = backend_.now_ns();
        if (now < previous) return {SubmissionStatus::ClockFailed};
        previous = now;
        if (now - started >= uint64_t(timeout_ms) * 1000000) return {SubmissionStatus::Timeout};
        backend_.sleep_us(100);
    }
    return {SubmissionStatus::Timeout};
}

} // namespace ps5hip
