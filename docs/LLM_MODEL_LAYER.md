# Actual-model first-layer qualification

The selected Q4 first layer passed native validation on .40 on 2026-10-17,
source `f251366`. Candidate `results/source-public-llm-layer-console-v1` used
`build/llm-native-layer-app-v1`. See `evidence/console40-model-layer-success.json`.
This bounded milestone does not close broader M2 coverage or establish inference.

## Executed graph

Select the actual TinyLlama GGUF graph through `l_out-0`: 37 nodes, ten resident
weights, five request controls and two resident F16 caches. The 21 computed F32
outputs include embedding, normalization, Q/K/V, RoPE, attention scores and
probabilities, attention values/output/residual, FFN normalization, gate/up,
SwiGLU, down projection and final residual. Views retain upstream shapes and
strides. Unsupported layouts, operations, unowned storage and malformed
positions, indices or causal masks reject before launching.

The explicit qualification route keeps the selected layer's weights and KV on
the GPU, with a separate allocation for each of the ten weights and two caches.
This avoids flushing unrelated persistent tensors on every synchronized kernel. The CPU prepares token, position, indices and mask controls only; it
never supplies computed activations. The full original llama graph still
rejects. Its failed-decode rollback cannot establish decoder progression, so
this test drives the selected graph's declared sequence directly. Full-model
scheduler/cache integration remains a subsequent milestone.

## Sequence and evidence

Run positions 0 through 511 using tokens 0, 1, 2 and 31999 cyclically. The causal
extent grows from 256 to 512 at position 256. Clear both GPU caches and repeat
positions 0 and 1 without re-uploading weights. Read all 21 outputs at positions
0, 1, 255, 256 and 511, and at both reset positions. Read both complete F16 caches
after each sequence: 151 files totaling 2,878,464 bytes. Reset outputs must match
bitwise. The two sequence lengths total 514 requests and 16,962 GPU launches.

The CPU reference first checks the selected graph against four successful full
CPU model requests. It then runs the same persistent-cache sequence twice and
requires bitwise repeatability. CPU fusion is disabled so intermediate RMSNorm
outputs are materialized. Existing CPU libraries are reused.

The layer-specific Q4 path quantizes each 32-value activation block with the
active CPU reference's nearest-even rounding, then sums signed-nibble/int8
products. It avoids the prefix kernel's half-rounded sum correction. Existing
prefix kernels and evidence remain unchanged. Attention operands are explicitly
rounded to F16, matching the upstream F16 dot-product inputs; accumulation is
F32. Every kernel uses the already-supported 64-thread linear launch path with
no private memory or LDS.

Attention score arithmetic is checked independently against an F64 dot product
of the CPU-validated device Q and K operands after F16 conversion. The bound is
`gamma(64) * sum(abs(products))`; the CPU relative RMS check remains required.
This handles a half-rounding boundary without loosening arithmetic acceptance.
Every other sampled F32 stage retains the fixed CPU elementwise and RMS gates.

Native acceptance requires an authenticated package and references, idle .40,
a single bounded PPSA99014 run, strict transcript and raw inventory, numerical
comparison, balanced allocation/queue cleanup, restored memory and healthy
services. This cycle passed: 514 requests, 16,962 launches, all 151 raw files,
all 21 sampled stages through the final residual, exact reset and 528 balanced
allocations/frees. The largest free extent returned to 12,882,804,736 bytes.
Preserve the consumed candidate and earlier prefix recovery evidence.
The model remains external at `/data/hip/models/tinyllama-q4_0.gguf`.

## Remaining release work

M2 still requires a dense floating-point whole-layer fixture and representative
extreme inputs. One selected layer does not provide token generation. Advance
M3 across all 22 layers,
integrate decoder cache progression, final normalization/logits, sampling and
end-to-end prompt/decode validation before declaring the GGUF inference release.
Direct safetensors and PyTorch checkpoint readers remain separately required
and unimplemented. Training and full ROCm coverage remain outside this release.
