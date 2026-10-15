# First release: native HIP LLM inference

Owner decision: 2026-10-15. This is the active release priority. It supersedes
historical next-step paragraphs that put general G5 profiling ahead of a real
inference client. The full SDK goals G1-G12 remain open with their original
acceptance criteria; completing this release does not complete the SDK.

## Release contract

Deliver a standalone native PS5 client that loads a supported GGUF model and
generates tokens through ps5-hip on the console GPU. Keep it in this repository
as an SDK example; ProsperoAI integration remains outside scope.

- One dense Llama-family architecture; no mixture-of-experts or multimodal model.
- Batch size one, one active request and one console GPU.
- Initial context ceiling: 512 tokens including prompt and generated tokens.
  Reject requests beyond the qualified limit explicitly.
- A small dense floating-point fixture for numerical bring-up, followed by one
  real model in the exact GGUF Q4_0 format. Inventory residual non-Q4 tensors;
  support their actual dtypes rather than assuming every tensor is quantized.
- Candidate checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Its model card
  identifies a Llama 2 architecture/tokenizer and Apache-2.0 license. Freeze the
  exact checkpoint, tokenizer, conversion inputs and generated GGUF hashes in
  V1.1. Metadata/tokenizer and the expected source weight hash are now locked in
  `../models/tinyllama-1.1b-chat-v1.0.json`. Full source weights/tokenizer are now
  downloaded and hash-verified. F16 and pure Q4_0 GGUF conversion are verified
  in `../evidence/offline-llm-model-conversion.json`. CPU reference inference
  passes the fixed cases in `../evidence/offline-llm-cpu-reference.json`;
  native execution remains unqualified. It is not yet a supported native model.
- CPU model loading, tokenization, scheduling and sampling are expected. Model
  tensor computation must use the PS5 HIP backend. Any temporary CPU tensor
  fallback is diagnostic and must be listed; it cannot silently satisfy final
  GPU inference acceptance.
- No predetermined tokens/second or speedup claim. Publish measured prefill,
  decode, initialization and memory results for the qualified configuration.

Model source: [TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
If that candidate cannot fit the measured budget, record the reason and select
a smaller real model with the same architecture; do not replace final model
inference with a synthetic kernel demonstration.

## Implementation order and exit evidence

### Scope rule for the first release

Useful GPU inference is the release deliverable; a complete ROCm SDK is not a
prerequisite. Accept new runtime or library work into v1 only when it closes a
demonstrated client blocker or satisfies the validation contract below. Record
the affected client call, tensor operation or acceptance check before expanding
coverage. Keep broader SDK work on its existing roadmap.

The required stack is the offline compiler/device-library subset, native HIP
allocation/copy/launch/synchronization and module loading, the model's tensor
operators, and the standalone inference client. Implement only the math-library
operations that the selected backend actually calls; complete hipBLAS/rocBLAS,
TheRock, Python and PyTorch ports are not release prerequisites. Preserve explicit
unsupported errors and existing correctness regressions.

Prioritize a working path through the whole model before optional throughput
optimization. CPU reference fixtures and kernel canaries are prerequisites, not
a substitute for native model inference. The v1 exit is V1.1-V1.5 below, including
repeatable generation and measured performance; no broader SDK percentage can
stand in for those checks.

| Milestone | Work | Required evidence |
| --- | --- | --- |
| V1.1 Client and model profile | Lock upstream llama.cpp, model/tokenizer/converter and Q4_0 outputs; inventory active HIP APIs, library calls, emitted kernel features, tensor shapes and total memory | Reproducible CPU reference, exact inputs, operator/API coverage table, compiler/link blocker report and proposed console memory budget |
| V1.2 Public runtime coverage | Connect already-proven geometry/LDS/private-memory routes to genuine public HIP where the client requires them; qualify model-scale allocations, streams and copies | Native .40 tests for the actual launch/resource shapes and sizes; errors reject unsupported cases; cleanup and uncertainty ownership preserved |
| V1.3 Required tensor operations | Implement or adapt matrix/vector multiplication, layouts, conversions, reductions, RMSNorm, RoPE, softmax, activation/gating, attention, KV updates and Q4_0 operations selected by V1.1 | Native numerical comparisons against CPU references for representative model shapes, strides, tail sizes and dtypes; declared tolerances and negative controls |
| V1.4 End-to-end inference | Integrate the native client, GGUF loader/tokenizer, weight residency, KV cache, prefill, decode and sampling | Dense fixture then real Q4_0 model; reference-checked logits and deterministic token fixtures; on-console GPU dispatch evidence |
| V1.5 Release qualification | Repeated requests, KV reuse/reset, cancellation, failure handling, unload/reload and source-only packaging | Bounded memory, correct outputs, clean teardown, reproducible build and honest performance/capability report |

These milestones overlap when independent work is possible. Start the client
compile/operator inventory now; do not require general G5 completion first.
Implement the smallest reusable HIP/library capability that unblocks the client,
not a separate fake runtime that bypasses the SDK under test.

## Client baseline and integration choices

Initial upstream source inspected from `ggml-org/llama.cpp` master:
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`. This is an inventory baseline, not a
qualified native build or a requirement to maintain a fork. CPU reference tools
and upstream quantization/RoPE tests now pass; see `LLM_V1_PROFILE.md`.
Source hashes and the
inspection scope are in `../evidence/llm-v1-planning-baseline.json`.

The upstream HIP CMake target requires HIP, hipBLAS and rocBLAS, compiles shared
CUDA operator sources as HIP and rejects `GGML_STATIC`. Therefore the first
integration spike must resolve both native linkage and actual math calls.
Deleting dependency checks or exporting success stubs does not satisfy this.
Prefer required upstream operations; where a full library port is impractical,
use a documented operation subset and explicit client/backend adaptation.
Do not call a subset a complete rocBLAS or hipBLAS port.

Initial backend profile: HIP enabled, graphs off, VMM disabled, RCCL off,
MFMA-specific quantized multiplication off, ordinary attention initially.
No managed-memory or host-registration opt-ins. The effective preprocessed
configuration and emitted ISA must be inspected; flags alone do not establish
PS5 compatibility. Preserve measured gfx1013 requirements; do not impersonate
another GPU architecture to make compilation pass.

Current public adapter blockers include its Initial launch route and advertised
64 MiB per-allocation limit. Existing geometry/LDS/private-memory proofs and
legacy GEMM tests are useful inputs, not proof those paths work through public
HIP or support model tensors. Validate and implement the required route/limit;
never merely raise advertised limits.

## V1 validation contract

- Compare intermediate tensors and logits for fixed token inputs against the
  matching CPU implementation and dtype/quantization. Freeze tolerances before
  accepting runs; investigate argmax differences rather than checking only that
  text looks plausible. Quantized output is compared with the same quantized
  reference, not assumed identical to the unquantized model.
- Freeze at least five prompt/token fixtures with at least 16 decode steps each,
  plus a 64-step generation case and a context-boundary case. Include prefill and
  incremental decode with KV reuse; resetting a session must remove prior state.
- Run at least 20 sequential requests, including different prompt lengths and
  two unload/reload cycles. Record owned model, KV, workspace and native backing
  allocations; require bounded steady-state ownership and expected release on
  unload. Do not infer total memory stability from only one counter family.
- Exercise cancellation at safe prefill/decode boundaries, invalid/unsupported
  model inputs, allocation failure and over-limit context rejection. No stale
  output, silent tensor fallback, use-after-free or uncertain-GPU force-free.
- Report initialization, prefill tokens/second, decode tokens/second, peak owned
  memory and the complete fallback inventory. A correctness preview can ship
  without a speed win; an unmeasured performance claim cannot.
- Rebuild SDK and client from a clean source checkout with fetched public inputs
  and documented model preparation. Add real Make targets when implemented,
  not placeholder commands. Freeze packages and validate important milestones
  on idle .40 under the existing console protocol.
- Update evidence/state/handoff and push each validated milestone to private
  GitHub `main`. Release publication and repository visibility are separate
  owner decisions; no public release is authorized by this plan.

## Deferred beyond this release

General command-retirement diagnostics, exact CLR command plateau profiling,
graph speedup optimization and broad operator/API coverage are not v1 gates
unless a demonstrated client issue makes them necessary. Existing graph and
ownership tests remain regressions; memory correctness is never deferred.

Also defer native Python/PyTorch, arbitrary model families/quantizations,
training/LoRA/QLoRA, unused ROCm libraries, on-console compilation and
multi-console compute. These retain their SDK goals. After useful inference,
prioritize measured performance and a separately scoped bounded LoRA release;
that requires backward kernels, gradient validation, optimizer state and
checkpoint/resume, even when base weights are frozen.

## Immediate next action

Continue V1.1 from `LLM_V1_PROFILE.md`: qualify compiler-generated embedded-image
native launch, prepare the effective HIP compile/link probe, and use the verified
CPU token/logit references to qualify required operators. The host tools and model
metadata inventory are reproducible through optional Make targets; V1.1 remains
partial. Select work from demonstrated client blockers. No native LLM, training
or first-release completion is claimed yet.
