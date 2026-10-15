# Inference v1: locked inputs and current blockers

This is a partial V1.1 checkpoint, not a completed client port. No model has run
through native HIP, and no converted GGUF or reference logits are qualified yet.

## Reproduce the inputs and host tools

Run from WSL/Linux with the existing build prerequisites, including LLVM 18.1.8:

```sh
make llm-deps-fetch
make llm-cpu-reference LLM_CPU_BUILD_DIR=build/llm-cpu-reference
make llm-model-profile LLM_PROFILE_DIR=results/llm-model-profile
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

The Q4_0 scenario assigns 18 bytes per 32 elements to rank-two-or-higher tensors
whose last dimension is divisible by 32, and FP32 to the others. It is not an
observed converter policy or GGUF size. KV assumes 22 layers, four KV heads,
head dimension 64 and two bytes per element for both K and V. These figures
exclude activations, workspace, runtime/host allocations, padding and mappings.

The source file is 2,200,119,864 bytes; its expected SHA-256 is
`6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`.
Future weight download must verify it before conversion. Then lock converter
dependencies/options and actual F16/Q4_0 outputs; refresh the memory inventory
from those outputs and the client's real buffer allocation requests.

The current public adapter caps individual allocations at 64 MiB. Dense
embeddings exceed that; quantizing may reduce individual tensors, but packed
model buffers can still exceed the cap. Do not infer a required single buffer
size from total weight bytes or simply raise an advertised capability.

## Source-inspected client blockers

The selected profile disables HIP graphs, VMM, RCCL and MFMA-specific MMQ and
uses ordinary attention. The findings below are source inspection, not a
preprocessed/link-complete API census. Disabled runtime paths can still be
compiled or registered because upstream globs broad operator/template sets.

| Blocker | Concrete source evidence and next work |
| --- | --- |
| Build/library integration | Upstream `ggml/src/ggml-hip/CMakeLists.txt` requires `hip`, `hipblas`, `rocblas`, links all three, and rejects `GGML_STATIC`. Establish native build/link integration and real operation providers; removing package checks is insufficient. |
| Embedded image registration | `ports/clr/ps5-public-module-aot.patch` rejects pointer-only images (`fname_.empty()`). File-based public module tests do not establish the compiler-generated `<<<>>>` path. First qualify one bounded embedded image through the genuine registration path. |
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

Next: a genuine embedded-image registration/launch test and the effective HIP
client compile/link probe; in parallel prepare the pinned weights/conversion
and CPU reference fixtures. Follow with the required operator ABI/geometry,
matmul and memory coverage. General G5 diagnostics remain deferred.
