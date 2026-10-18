#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// CPU scheduling/sampling only. The caller owns decoder state, reset, and all
// tensor computation; logits must remain valid until the next decode call.
namespace ps5hip_generation {
constexpr size_t context_size = 512;
constexpr size_t vocabulary_size = 32000;
struct Logits { const float* data; size_t count; };
enum class StopReason { step_limit, eos };
struct Result {
    std::array<int32_t, context_size> processed{};
    std::array<int32_t, context_size> generated{};
    size_t processed_count = 0, generated_count = 0;
    StopReason stop = StopReason::step_limit;
};
inline int32_t greedy(Logits logits) {
    if (!logits.data || logits.count != vocabulary_size)
        throw std::invalid_argument("generation requires full logits");
    int32_t best = 0;
    for (size_t i = 0; i < vocabulary_size; ++i) {
        if (!std::isfinite(logits.data[i]))
            throw std::runtime_error("generation nonfinite logits");
        if (logits.data[i] > logits.data[best]) best = static_cast<int32_t>(i);
    }
    return best; // Strict comparison retains the lowest index on a tie.
}

template<class Decode, class IsEog>
Result generate(const int32_t* prompt, size_t prompt_count, size_t max_new_tokens,
                Decode&& decode, IsEog&& is_eog) {
    // Reserve the combined prompt/output budget before any submission, including
    // the final selected token even though it is not decoded.
    if (!prompt || !prompt_count || prompt_count > context_size ||
        !max_new_tokens || max_new_tokens > context_size ||
        max_new_tokens > context_size - prompt_count)
        throw std::invalid_argument("generation context/budget");
    for (size_t i = 0; i < prompt_count; ++i)
        if (prompt[i] < 0 || static_cast<size_t>(prompt[i]) >= vocabulary_size)
            throw std::invalid_argument("generation prompt token");
    Result result;
    auto process = [&](int32_t token) {
        const auto position = static_cast<int32_t>(result.processed_count);
        const int32_t next = greedy(decode(token, position));
        result.processed[result.processed_count++] = token;
        return next;
    };
    int32_t next = 0;
    for (size_t i = 0; i < prompt_count; ++i) next = process(prompt[i]);
    while (true) {
        result.generated[result.generated_count++] = next;
        if (is_eog(next)) { result.stop = StopReason::eos; break; }
        if (result.generated_count == max_new_tokens) break;
        next = process(next); // Feed the actual selected token back to the decoder.
    }
    return result;
}
} // namespace ps5hip_generation
