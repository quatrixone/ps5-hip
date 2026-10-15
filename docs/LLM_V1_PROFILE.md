# Inference v1: locked inputs and current blockers

This is a partial V1.1 checkpoint, not a completed client port. Locked source
weights now convert to inspected F16 and pure Q4_0 GGUF outputs. CPU Q4_0
prefill/decode logits and greedy-token references pass repeated comparisons;
native HIP model execution remains unqualified.

## Reproduce the inputs and host tools

Run from WSL/Linux with the existing build prerequisites, including LLVM 18.1.8:

```sh
make llm-deps-fetch
make llm-cpu-reference LLM_CPU_BUILD_DIR=build/llm-cpu-reference
make llm-model-profile LLM_PROFILE_DIR=results/llm-model-profile
make llm-model-fetch LLM_MODEL_RECEIPT=results/llm-model-source/receipt.json
make llm-model-convert LLM_CPU_BUILD_DIR=build/llm-cpu-reference
make llm-reference-fixtures LLM_CPU_BUILD_DIR=build/llm-cpu-reference \
  LLM_REFERENCE_DIR=results/llm-reference
```

These are optional targets; the default native SDK build does not fetch an LLM
client or model. `llm-inputs.json` locks the upstream llama.cpp master snapshot
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`. There is no maintained client fork.
The fetcher checks repository URL, revision and clean state and expands the
earlier sparse inventory checkout into a full source checkout.

The CPU target builds `llama-completion`, `llama-tokenize`, `llama-quantize` and
`llama-perplexity`, plus upstream quantization and RoPE tests. It disables GPU
backends, BLAS, OpenMP and host-specific native tuning. This is a reproducible
host reference profile, not a claim about the fastest CPU control. Its receipt
records source, compiler/tool identities, configuration, commands and artifacts.
Reruns require identical inputs and repeat the checks; choose a fresh output
after source/tool changes. Failed reruns invalidate the prior success receipt.

The model-profile target runs network-free tooling tests, then fetches only
bounded immutable metadata and safetensors header ranges. Use a fresh profile
directory for each run. It verifies `models/tinyllama-1.1b-chat-v1.0.json`, which
contains metadata hashes, tensor inventory and the expected full weight hash
from the Git LFS pointer. Full weights are not downloaded or hash-verified by
this target. It is not a model converter or inference test.

`llm-model-fetch` downloads the full locked source weights and metadata into
ignored `models/tinyllama-source` (override `LLM_MODEL_DIR` if needed). It streams
and verifies every file's exact size/SHA before publication, preserves corrupt
existing files for inspection, and writes a receipt only after full-set checks.
It requires about 2.2 GB of storage. Reuse verified files with a new
`LLM_MODEL_RECEIPT` path; old evidence is not overwritten. This step still does
not convert or run the model.

`llm-model-convert` prepares a local Linux x86-64 CPU conversion environment,
revalidates/builds the CPU tools, then uses the exact upstream converter and
quantizer. `llm-conversion-inputs.json` and
`ports/llm-conversion-requirements.txt` lock CPython 3.12.15 and 28 dependency
wheels by SHA, including CPU-only torch 2.11.0+cpu, transformers 4.57.6 and
numpy 2.2.6. No global packages are installed. System Python 3.14 cannot use the
locked NumPy wheels, so the isolated Python is fetched automatically. Optional
terminal-database links are omitted from its archive extraction for WSL `/mnt/c`.

The environment defaults to `.deps/llm-conversion-env` (`LLM_CONVERSION_ENV`).
Matching installed package versions and dependency checks allow reuse without
another installation; each run rechecks the archive, Python and tool identities.
Earlier environment receipts are preserved by hash. A changed dependency lock
requires a fresh environment directory. `make llm-conversion-env` runs only this
preparation and the network-free tool tests.

Conversion verifies all source inputs and CPU artifacts, runs with Hugging Face
offline mode, writes F16, then invokes `llama-quantize --pure ... Q4_0 4`.
The `--pure` choice avoids additional K-quantized tensor types. Output defaults
to ignored `models/tinyllama-gguf` (`LLM_GGUF_DIR`) and must be fresh: failed or
previous outputs are preserved. A receipt is published only after both commands,
GGUF architecture/dimension/dtype checks and final source/tool checks pass.
Allow at least 7 GB total for source weights, environment and both outputs,
plus the CPU tool build. Package installation on `/mnt/c` can take several minutes.
The first local output hashes below identify the measured build; cross-machine
bit-for-bit conversion reproducibility has not yet been tested.

`llm-reference-fixtures` rebuilds/revalidates the CPU tools, then compiles the
small upstream-API harness in `tests/llm_cpu_reference.cpp`. It requires the
qualified Q4_0 output above; it does not fetch or convert a model implicitly.
`tests/fixtures/llm-reference-prompts.json` freezes six text prompts and their
token IDs, plus a repeated-token context-boundary input. The text cases are
retokenized and checked exactly with both `add_special` and `parse_special`
enabled. These are raw completion fixtures, not chat-template validation.

The seven cases cover five different prompts with 16 incremental decode calls
each, one with 64 calls, and 496 input tokens plus 16 calls filling context 512.
Each case uses batch one, four CPU threads, F16 K/V, disabled Flash Attention,
zero GPU layers and disabled operation/KQV offloading. There are two independent
process/context runs per case. All 320 generated/fed tokens match the lowest-ID
argmax of their raw logits; all corresponding repeats are bitwise identical.
The 334 rows contain 10,688,000 finite logits (32,000 vocabulary entries/row).

Raw `.f32le` files contain a full prefill logit row, then one full row after
each actual incremental decode. The final row is retained for comparison but
not sampled; the boundary case never submits a 513th token. `.tokens` files
hold the greedy tokens actually fed back. Fixed-step numerical fixtures
deliberately continue through end-of-generation tokens, so this does not qualify
normal chat termination. Each run has a fresh context: same-context reset,
unload/reload lifecycle and native execution still need their own tests.
See `../evidence/offline-llm-cpu-reference.json` for hashes and results.

Reference outputs require a fresh `LLM_REFERENCE_DIR`; raw logits stay ignored
and can be regenerated from the frozen source inputs. Receipts bind the model,
compiler, CPU libraries, harness, tokenizer inputs and output hashes. Matching
repeats on this host do not establish cross-machine bitwise equivalence or a
native floating-point tolerance. No tokens/second or native memory-fit claim
is made by this fixture target.

## Model and memory inventory

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`.

| Item | Header/config-derived result |
| --- | ---: |
| Parameter tensors | 201 |
| Parameters | 1,100,048,384 |
| Source BF16 tensor payload / hypothetical all-F16 weights | 2,200,096,768 bytes |
| Largest dense tensor | 131,072,000 bytes (125 MiB); embeddings and output weights |
| Explicit Q4_0 storage scenario | 619,094,016 bytes |
| F16 KV cache, batch one, context 512 | 11,534,336 bytes (11 MiB) |

The original Q4_0 scenario assigns 18 bytes per 32 elements to rank-two-or-higher tensors
whose last dimension is divisible by 32, and FP32 to the others. It is not a
GGUF file-size estimate. The measured pure-Q4 conversion below now confirms its
tensor payload size for this checkpoint. KV assumes 22 layers, four KV heads,
head dimension 64 and two bytes per element for both K and V. These figures
exclude activations, workspace, runtime/host allocations, padding and mappings.

The source file is 2,200,119,864 bytes; its expected SHA-256 is
`6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`.
The full source download has now verified that hash; see
`../evidence/offline-llm-model-source.json`.

Measured conversion (`../evidence/offline-llm-model-conversion.json`):

| Output | GGUF file bytes | Tensor payload bytes | Tensor types |
| --- | ---: | ---: | --- |
| F16 | 2,201,018,624 | 2,200,281,088 | 156 F16, 45 F32 |
| Pure Q4_0 | 619,831,552 | 619,094,016 | 156 Q4_0, 45 F32 |

Both outputs contain all 201 tensors / 1,100,048,384 parameters. The 45 residual
F32 tensors are 22 attention norms, 22 feed-forward norms and the final output
norm: 2,048 elements each, totaling 368,640 bytes. GGUF metadata/alignment adds
737,536 bytes to each tensor payload. F16 SHA-256:
`e657561bf17a670d0e1516d6338a7a75f65803acf58fd53bd822a08c3ddf44df`.
Pure Q4_0 SHA-256:
`c42ff3ebdfc0b8a0aa0c35a499ccf81303fb9fbdc1a0798c096a098df85003aa`.
CPU token/logit fixtures are now frozen as described above. Next measure native
client allocations and operator results; file size alone still does not
establish native memory fit or native model correctness.

The current public adapter caps individual allocations at 64 MiB. Dense
embeddings exceed that; quantizing may reduce individual tensors, but packed
model buffers can still exceed the cap. Do not infer a required single buffer
size from total weight bytes or simply raise an advertised capability.

## Source-inspected client blockers

The bounded [real compiler probe](LLM_V1_COMPILE_PROBE.md) now supplements this
inventory: twelve host objects compile and three device modules link. It also
confirms aggregate ABI/resource requirements and remaining library/version
blockers. This is not a complete client link or native execution result.

The selected profile disables HIP graphs, VMM, RCCL and MFMA-specific MMQ and
uses ordinary attention. The findings below are source inspection, not a
preprocessed/link-complete API census. Disabled runtime paths can still be
compiled or registered because upstream globs broad operator/template sets.

| Blocker | Concrete source evidence and next work |
| --- | --- |
| Build/library integration | Upstream `ggml/src/ggml-hip/CMakeLists.txt` requires `hip`, `hipblas`, `rocblas`, links all three, and rejects `GGML_STATIC`. Establish native build/link integration and real operation providers; removing package checks is insufficient. |
| Native embedded image execution | Bounded compiler-generated registration/stubs now pass genuine CLR host qualification with modeled GPU transport. Native compiler-stub execution is still pending; the cross-compiled bootstrap package alone does not prove it. Pointer-only dynamic image loading remains unsupported. See `PUBLIC_EMBEDDED_IMAGES.md`. |
| Whole-module admission | `ports/clr/ps5-device.cpp` checks every kernel against the Initial route; an unsupported unused kernel can reject a module. Measure the compiled closure rather than assuming runtime operator selection removes it. |
| Argument ABI | Current capture/admission accepts only four/eight-byte values and non-null registered global buffers. Ordinary normalization passes `uint3` aggregates and optional null buffers; softmax uses a larger structure. Inspect emitted metadata, then implement the required copying/validation semantics. |
| Geometry and LDS | Ordinary RMSNorm in `ggml/src/ggml-cuda/norm.cu` uses 256/1024-thread blocks and shared storage; RoPE uses Y-thread geometry. Current public CLR caps workgroups at 64 and admits no LDS. Integrate already-proven routes with public HIP and validate these actual operators. |
| Matmul and numerical types | `ggml-cuda.cu` retains actual GEMM fallbacks without MFMA. FP16/BF16 conversions, reductions and `rsqrtf`/`expf`/trigonometric math remain relevant. Select calls from the effective build/model, not every vendor alias: `cublasSetMathMode`, for example, is a success macro on HIP. |
| Model memory | Validate actual weight/KV/workspace placement and live allocation requests against available native memory. A metadata estimate proves neither model fit nor cleanup. |

Relevant pinned upstream files:
[HIP build](https://github.com/ggml-org/llama.cpp/blob/b76b13379d3b7d67ab0bc424516bb6fd2522bd8e/ggml/src/ggml-hip/CMakeLists.txt),
[normalization](https://github.com/ggml-org/llama.cpp/blob/b76b13379d3b7d67ab0bc424516bb6fd2522bd8e/ggml/src/ggml-cuda/norm.cu),
[RoPE](https://github.com/ggml-org/llama.cpp/blob/b76b13379d3b7d67ab0bc424516bb6fd2522bd8e/ggml/src/ggml-cuda/rope.cu),
[softmax](https://github.com/ggml-org/llama.cpp/blob/b76b13379d3b7d67ab0bc424516bb6fd2522bd8e/ggml/src/ggml-cuda/softmax.cu),
[HIP aliases](https://github.com/ggml-org/llama.cpp/blob/b76b13379d3b7d67ab0bc424516bb6fd2522bd8e/ggml/src/ggml-cuda/vendors/hip.h).

Next: a native embedded-image registration/launch test and the effective HIP
client compile/link probe, using the frozen CPU references for later numerical
comparison. Follow with the required operator ABI/geometry,
matmul and memory coverage. General G5 diagnostics remain deferred.

The static-module failure publication prerequisite and bounded compiler-image
registration now pass genuine CLR host qualification; see
`../evidence/offline-public-embedded-images.json`. GPU transport was modeled.
Remaining native compiler-launch work is documented in
[PUBLIC_EMBEDDED_IMAGES.md](PUBLIC_EMBEDDED_IMAGES.md).
