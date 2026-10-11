#include "ps5hip/submission.h"
#include <ps5platform/kernel.h>
#include <cstdlib>
#include <iostream>
#include <map>

namespace {
std::map<int64_t, void *> allocations;
int64_t next_id = 1;
unsigned released = 0;
bool fail_unmap = false, fail_release = false;
unsigned cleanup_failures = 0;
void check(bool ok) { if (!ok) throw std::runtime_error("submission assertion failed"); }
template<class F> void reject(F f) {
    try { f(); } catch (const ps5hip::ObjectError &) { return; }
    throw std::runtime_error("invalid submission accepted");
}
struct Model : ps5hip::SubmissionBackend {
    int32_t submit_error = 0, kick_error = 0;
    uint64_t clock = 0;
    unsigned submits = 0, polls = 0, writes_after = 0;
    bool frozen_clock = false, throw_flush = false, throw_submit = false, reverse_clock = false;
    bool throw_kick = false, throw_clock = false, throw_sleep = false;
    unsigned flushes = 0, fail_flush_at = 0;
    volatile uint32_t *marker = nullptr;
    void flush(void *, size_t) override {
        if (++flushes == fail_flush_at || throw_flush) throw ps5hip::ObjectError("injected flush failure");
    }
    int32_t submit(uint32_t *words, uint32_t count) override {
        ++submits;
        if (throw_submit) throw ps5hip::ObjectError("injected driver exception");
        check(count >= 11);
        const auto *end = words + count - 8;
        check(end[-2] == 0xc0004600 && end[-1] == 0x407);
        check(end[0] == 0xc0064900 && end[1] == 0x30c514 && end[2] == 0x20000000);
        check(end[5] == 1 && end[6] == 0 && end[7] == 0);
        marker = reinterpret_cast<volatile uint32_t *>(uint64_t(end[3]) | uint64_t(end[4]) << 32);
        check(*marker == 0);
        if (writes_after == 0) *marker = 1;
        return submit_error;
    }
    int32_t kick() override {
        if (throw_kick) throw ps5hip::ObjectError("injected kick exception");
        return kick_error;
    }
    uint64_t now_ns() override {
        if (throw_clock) throw ps5hip::ObjectError("injected clock exception");
        return clock;
    }
    void sleep_us(unsigned us) override {
        if (throw_sleep) throw ps5hip::ObjectError("injected sleep exception");
        ++polls;
        if (reverse_clock) clock -= 1;
        else if (!frozen_clock) clock += uint64_t(us) * 1000;
        if (polls == writes_after && marker) *marker = 1;
    }
};
}
extern "C" int64_t sceKernelGetDirectMemorySize() { return int64_t(12) << 30; }
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t bytes, size_t alignment, int, int64_t *physical) {
    void *memory = nullptr;
    if (posix_memalign(&memory, alignment, bytes)) return -1;
    *physical = next_id++; allocations[*physical] = memory; return 0;
}
extern "C" int32_t sceKernelMapDirectMemory(void **address, size_t, int, int, int64_t physical, size_t) {
    *address = allocations.at(physical); return 0;
}
extern "C" int32_t sceKernelMunmap(void *, size_t) {
    if (fail_unmap) { fail_unmap = false; ++cleanup_failures; return -1; }
    return 0;
}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t physical, size_t) {
    if (fail_release) { fail_release = false; ++cleanup_failures; return -1; }
    std::free(allocations.at(physical)); allocations.erase(physical); ++released; return 0;
}

int main() {
    try {
        using namespace ps5hip;
        const std::vector<uint32_t> commands{0xffff1000}; // Model payload; never sent to hardware.
        {
            Model model; SubmissionQueue queue(model);
            auto resource = std::make_shared<NativeMemory>(64, 64, false);
            check(queue.submit_and_wait(commands, {resource, resource}, 10).status == SubmissionStatus::Complete);
            check(!queue.poisoned() && resource->close());
            reject([&] { SubmissionQueue second(model); });
            model.writes_after = 3;
            check(queue.submit_and_wait(commands, {}, 10).status == SubmissionStatus::Complete);
            check(model.polls == 3);
        }
        for (bool unmap : {false, true}) {
            Model model; SubmissionQueue queue(model);
            auto resource = std::make_shared<NativeMemory>(64, 64, false);
            const auto before = released;
            const auto failures_before = cleanup_failures;
            fail_unmap = unmap;
            fail_release = !unmap;
            reject([&] { queue.submit_and_wait(commands, {resource}, 1); });
            // Completion released GPU holds, but a cleanup failure must not pass.
            check(!queue.poisoned() && model.submits == 1);
            check(cleanup_failures == failures_before + 1);
            check(released == before + 2 && allocations.size() == 1);
            check(resource->close() && allocations.empty());
        }
        {
            Model model; SubmissionQueue queue(model);
            auto resource = std::make_shared<NativeMemory>(64, 64, false);
            for (unsigned round = 0; round < 2; ++round) {
                *static_cast<uint32_t *>(resource->cpu()) = 0x12340000u + round;
                const auto before = released;
                // A delayed second marker must not inherit the first completion.
                model.writes_after = round ? 3 : 0;
                check(queue.submit_and_wait(commands, {resource}, 10).status == SubmissionStatus::Complete);
                check(!queue.poisoned() && model.submits == round + 1);
                check(released == before + 2 && allocations.size() == 1);
                check(*static_cast<uint32_t *>(resource->cpu()) == 0x12340000u + round);
            }
            check(model.polls == 3 && resource->close() && allocations.empty());
        }
        for (unsigned failure = 0; failure < 3; ++failure) {
            Model model; SubmissionQueue queue(model);
            auto first = std::make_shared<NativeMemory>(64, 64, false);
            check(queue.submit_and_wait(commands, {first}, 10).status == SubmissionStatus::Complete);
            check(first->close() && allocations.empty());
            auto second = std::make_shared<NativeMemory>(64, 64, false);
            model.writes_after = 10000;
            if (failure == 0) model.submit_error = -10;
            if (failure == 1) model.kick_error = -20;
            const auto before = released;
            const auto result = queue.submit_and_wait(commands, {second}, 1);
            check(result.status == (failure == 0 ? SubmissionStatus::SubmitFailed :
                                    failure == 1 ? SubmissionStatus::KickFailed : SubmissionStatus::Timeout));
            check(queue.poisoned() && model.submits == 2 && !second->close() && released == before);
            *model.marker = 1;
            reject([&] { queue.submit_and_wait(commands, {second}, 1); });
            check(model.submits == 2 && !second->close() && released == before);
            // The model has no GPU; release quarantined backing only between cases.
            for (auto &entry : allocations) std::free(entry.second);
            allocations.clear();
        }
        for (unsigned failure = 0; failure < 11; ++failure) {
            Model model; SubmissionQueue queue(model);
            auto resource = std::make_shared<NativeMemory>(64, 64, false);
            model.writes_after = 10000;
            if (failure == 0) model.submit_error = -10;
            if (failure == 1) model.kick_error = -20;
            if (failure == 3) model.frozen_clock = true;
            if (failure == 4) model.throw_submit = true;
            if (failure == 5) { model.reverse_clock = true; model.clock = 1000; }
            if (failure == 6) model.throw_kick = true;
            if (failure == 7) model.throw_clock = true;
            if (failure == 8) model.throw_sleep = true;
            if (failure == 9) model.fail_flush_at = 4; // First marker poll after three upload flushes.
            if (failure == 10) { model.writes_after = 0; model.fail_flush_at = 5; } // Readback after marker.
            const auto before = released;
            if (failure == 4 || failure >= 6) reject([&] { queue.submit_and_wait(commands, {resource}, 1); });
            else {
                const auto result = queue.submit_and_wait(commands, {resource}, 1);
                check(result.status == (failure == 0 ? SubmissionStatus::SubmitFailed :
                                       failure == 1 ? SubmissionStatus::KickFailed :
                                       failure == 5 ? SubmissionStatus::ClockFailed : SubmissionStatus::Timeout));
            }
            check(queue.poisoned() && !resource->close() && released == before);
            if (model.marker) *model.marker = 1; // Late completion cannot silently recover a poisoned queue.
            reject([&] { queue.submit_and_wait(commands, {}, 1); });
            check(model.submits == 1 && model.polls <= 11);
        }
        {
            Model model; SubmissionQueue queue(model);
            auto resource = std::make_shared<NativeMemory>(64, 64, false);
            model.throw_flush = true;
            reject([&] { queue.submit_and_wait(commands, {resource}, 1); });
            check(!queue.poisoned() && !model.submits && resource->close());
            reject([&] { queue.submit_and_wait({}, {}, 1); });
            reject([&] { queue.submit_and_wait(commands, {}, 0); });
        }
        // Only the model can free quarantined backing: it has no GPU or pending work.
        for (auto &entry : allocations) std::free(entry.second);
        allocations.clear();
        std::cout << "submission model: completion, failures, timeout quarantine and late-marker rejection passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
