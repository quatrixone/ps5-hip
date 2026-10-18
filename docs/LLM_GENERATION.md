# Greedy generation qualification

The native full-decoder smoke already qualifies three forced-token requests,
all 22 layers and full logits. M4 adds actual sampling feedback: the next input
must come from the preceding live logits, rather than a reference token list.
The first profile uses `Once upon a time`, the model's tokenizer with special
tokens enabled and special-text parsing disabled, and at most four new tokens.
It repeats after clearing the actual KV state.

`ports/ggml/generation.h` is the shared CPU scheduling/sampling driver. Tensor
computation belongs to its decoder callback. It checks all 32,000 logits for
finite values, selects the lowest token ID on an equal maximum, and stops on
the vocabulary's end-of-generation tokens or the requested output limit.
The final selected token is emitted without a subsequent decoder call; EOS is
also emitted without being fed back. Submitted positions therefore number
`prompt_tokens + generated_tokens - 1`. The worst-case budget must fit the
512-token context before the first submission; invalid requests submit nothing.
The caller owns decoder reset and error recovery.

On Linux/WSL, the source-only Make targets fetch locked dependencies and use
the existing model/CPU-reference prerequisites:

```sh
make llm-generation-reference
make llm-generation-numeric-tests
```

Existing authenticated inputs can be selected without rebuilding libraries:

```sh
make llm-generation-reference \
  LLM_CPU_BUILD_DIR=build/llm-cpu-reference-v3 \
  LLM_FULL_GRAPH_MODEL=/tmp/ps5-hip-tinyllama-gguf-v3/tinyllama-q4_0.gguf \
  LLM_GENERATION_REFERENCE_DIR=results/m4-generation-reference-repro
make llm-generation-numeric-tests \
  LLM_CPU_BUILD_DIR=build/llm-cpu-reference-v3 \
  LLM_FULL_GRAPH_MODEL=/tmp/ps5-hip-tinyllama-gguf-v3/tinyllama-q4_0.gguf \
  LLM_GENERATION_REFERENCE_DIR=results/m4-generation-reference-repro \
  LLM_GENERATION_NUMERIC_TEST_DIR=results/m4-generation-numeric-repro
```

Builders require fresh output directories. Receipts bind source, compiler,
consumed headers, runtime libraries, model and raw outputs. References contain
every committed input/position and full logits, generated IDs and vocabulary
pieces. Python independently checks argmax, feedback, stopping and exact reset.
The HIP interpreter additionally checks all captured layer boundaries for finite
values and reset equality, and compares logits to CPU under the existing
numerical budgets. It uses the actual graph scheduler and production arithmetic
in a host interpreter; it does not execute a GPU.

Native generation still needs profile-specific packaging, image audit, frozen
reference/readback contracts, timing and .40 validation. Preserve the accepted
native smoke and its consumed candidate; never replay it. This short profile
does not replace five-prompt/16-step, 64-step, 512-token, EOS and repeated-request
release gates in `LLM_V1_PLAN.md`, nor original-format loading requirements.

## Historical first offline outcome

The CPU oracle passes: five prompt tokens and three feedback decodes per run,
two runs, 16 full-logit files and four selected tokens `, there was a`.
The host HIP interpreter matches these selected tokens and resets all 384 raw
boundaries exactly, but fails all 16 numerical logit comparisons under unchanged
budgets. The maximum absolute error is 0.19304943 and relative L2 0.00918597.
See `evidence/offline-greedy-generation-reference-and-regression.json`.

Callback scheduling and external fusion settings were ruled out experimentally.
CPU SiLU substitution restores exact logits for positions 0/1 only; adding CPU
softmax still leaves 12 failures. Both are local diagnostic substitutions, not
production GPU work. Next inspect the first remaining position-2/layer-14
intermediate and required F32 contraction/rounding. Native generation remains
unqualified; the earlier forced-token native smoke is preserved independently.

## Current result: canonical GPU-source arithmetic

`evidence/offline-greedy-canonical-arithmetic.json` records the passing follow-up:
all 16 full-logit files are bitwise CPU-equal under unchanged limits, with actual
feedback/reset and balanced cleanup. All 384 captured boundary files reset exactly;
the independent CPU numerical reference here covers logits. The original
three-request/72-output smoke regression also remains bitwise CPU-identical.

Full mode now selects separate SiLU/softmax/RoPE symbols, retaining legacy
selected-layer behavior. The nonlinear routines translate pinned AVX2 polynomial
and reduction arithmetic into ordinary GPU scalar operations; RoPE uses explicit
FMAs. No CPU callbacks or captured activation data implement these kernels.
`kernels/llm_cpu_math.h` retains upstream MIT attribution. Standalone comparisons
against actual CPU library functions pass 33 SiLU and 134 softmax cases in each of
two compiler modes, including tails, masks, finite extremes and exact aliasing.
The fresh 16-symbol fixture passes Initial admission without private memory,
LDS, spills or undefined calls. Native arithmetic and performance remain open.

Use `results/llm-layer-fixture-v8`, CPU oracle v1 and canonical numeric v1 for
current local evidence. Old failures/diagnostics are preserved; their CPU
substitutions are never native proof. Next finish native profile/audit/reference
contracts and timings, then freeze and test a fresh package on .40.
