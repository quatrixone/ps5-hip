# Numerical references for the selected Q8_1/Q4_0 path

These are CPU oracles for the actual source-built GPU formats, not evidence of
GPU execution or complete model inference. They precede a frozen native package
and bounded .40 run. See [HIP_QUANTIZED.md](HIP_QUANTIZED.md) for dispatch coverage.

## Reference contract

Q8_1 uses 36-byte blocks: a binary16 scale, a binary16 sum of the original
activations, and 32 signed quantized bytes. Compute the scale in F32 before
rounding it to half; quantization uses that unrounded scale and ties away from
zero. Sum the original F32 activations in the shader's XOR16/8/4/2/1 order.
The ordinary CPU Q8 quantizer derives its sum differently and is not an exact
byte oracle for this shader. Native validation must compare every Q8 byte.

For Q4_0, the independent full-block reference evaluates
`d4 * (d8 * sum(unsigned_nibble * signed_q8) - 8 * stored_sum)` using exact
rational arithmetic and rounds the final result to FP64. A second reference
models the two-thread block split, two F32 fused operations, per-lane block
stride64, three LDS additions and five butterfly additions. Exact rational
rounding avoids host double-rounding in the F32 fused operations.

The output budget is a priori: `gamma_(ceil(blocks/64)+12) * magnitude`, where
`gamma_n = n*u/(1-n*u)` and `u=2^-24`. Magnitude sums the absolute operands
before cancellation in each partial contribution. It is not relative to the
possibly near-zero answer and is not fitted to console results. The lane model
must fit this budget before a native candidate is accepted for testing.

The supported reference domain includes finite binary16 subnormals. The
authentic first activation row requires this: one block sum is `-2^-23`, encoded
as half `0x8002`. The selected descriptors declare floating-point mode `0xf0`;
the Q4 instruction converts that half sum to F32 before arithmetic. Mode3
preserves denormals in the documented ABI. See the
[LLVM AMDGPU floating-point mode definitions](https://llvm.org/docs/AMDGPUUsage.html#floating-point-denorm-mode-enumeration-values).
This establishes the expected reference semantics; hardware validation remains
required. F32 subnormal intermediates, overflow, nonfinite values and half
underflow to zero remain outside this initial reference domain and fail explicitly.

## Assets and build

`tools/build_llm_q4_reference.py` authenticates the source-built Q8/Q4 modules,
the locked converted GGUF, the CPU tools and the prior RMSNorm reference. A
small extractor uses the genuine upstream GGUF/GGML APIs to read raw Q4 bytes;
two fresh extractions must match. No new GGUF parser is introduced.

From a source-only checkout, `make llm-q4-reference` builds the device fixtures,
fetches model inputs, prepares conversion and CPU tools, converts the model and
builds the RMSNorm prerequisite before generating these assets. Repeated runs
need fresh output overrides as described in
[SOURCE_ONLY_REBUILD.md](SOURCE_ONLY_REBUILD.md). With already authenticated
prerequisites, invoke the builder directly:

```sh
python3 tools/build_llm_q4_reference.py \
  --rmsnorm-reference results/llm-rmsnorm-reference \
  --fixture results/llm-q4-fixture \
  --model models/tinyllama-gguf/tinyllama-q4_0.gguf \
  --output results/llm-q4-reference
```

Cases cover `blk.0.attn_k.weight` at width2048 with 1/2/17/256 rows and
`blk.0.ffn_down.weight` at width5632 with 1/17 rows, plus hand-encoded synthetic
weights and standalone Q8 zero/constant/tie/padding cases. The 2048 input is the
authenticated CPU RMSNorm embedding row; the 5632 input is deterministic
synthetic data, not a traced model hidden state.

Each matrix/vector case records `input.f32le`, `weights.q4_0`, `q8.bin`,
`fp64le`, `lane.f32le` and `budget.f64le`. The frozen profile and receipt bind
inputs, outputs, tools, extraction commands and rejected corruption controls.
Use fresh output directories; failed runs are retained without a PASS receipt.

Next package both isolated Q4 execution with frozen Q8 bytes and chained
Q8-to-Q4 execution. Preserve complete raw readbacks, input bytes, guards and
ownership evidence. A passing operator run still leaves the remaining
transformer operations and full-model prefill/decode open.

## Native numerical candidate

`make public-hip-q4-package` builds the model/reference prerequisites, runs the
asset/build/raw/log/audit checks, builds the native Q4 title and audits the raw,
converted and extracted images. Use `Q4_BUILD_DIR`, `LLM_Q4_REFERENCE_DIR` and
the existing fixture/reference output overrides for fresh builds. This target
prepares a candidate; it does not deploy or establish GPU execution.

The title reuses four guarded allocations across 28 dispatches. Four standalone
Q8 cases precede eight matrix cases, each with frozen-Q8 multiplication,
quantization, and chained multiplication. The chained stage consumes the GPU's
Q8 allocation without reuploading its contents. Every stage writes all four
buffers exclusively: 112 raw files / 9,097,984 bytes. Acceptance compares 592
Q4 outputs and 80,640 Q8 bytes and checks input, weights, guards and inactive
bytes. The watchdog remains bounded at 30 seconds; completion timing is not
qualified until the actual console run.

For separately collected raw files, the authenticated offline verifier is:

```sh
python3 tools/verify_public_hip_q4.py \
  --fixture results/llm-q4-fixture \
  --reference results/llm-q4-reference \
  --dumps results/q4-readbacks \
  --output results/q4-raw-verification.json
```

The output must be fresh. Its PASS establishes numerical and byte comparisons,
not where the dumps originated. Candidate identity, collection freshness,
natural exit, service health and owned-lock cleanup remain console-control
requirements. The strict transcript parser also cannot establish those facts.

The Q4 console controls authenticate the frozen reference and all package
assets before connecting. They require all Q4 dump paths to be absent before
launch, collect exactly the expected 112 files, and compare each stage's
reported metrics with the raw results before accepting the run. Normal and
partial-failure collection share a 120-second budget. That budget rejects late
acceptance; it does not forcibly cancel a blocked FTP operation, whose existing
socket inactivity timeout still applies. Failed/partial files are retained.
