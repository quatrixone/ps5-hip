# Full-model numerical reproduction

The offline full-model scheduler executes three real TinyLlama decoder requests: tokens 0 and 1 at positions 0 and 1, then token 0 after clearing decoder memory and resetting the backend. The numerical HIP test double interprets the production layer kernels on the host. This is not GPU execution or native inference acceptance.

`tests/llm_full_numeric_test.cpp` captures all 22 layer boundaries, final weighted normalization, and logits in the scheduler's completed-node callback before activation storage is reused. It checks all scheduler backend compute interfaces to reject CPU tensor arithmetic, while permitting the explicit host interpretation of HIP kernels. All three original decoder calls must succeed, commit their actual KV positions, and clean up their owned allocations. The 72 output files must be finite and reset outputs must match exactly.

Reproduce with existing authenticated CPU libraries and the original model:

```sh
python3 tools/build_llm_full_numeric_test.py \
  --cpu-build build/llm-cpu-reference-v3 \
  --model /absolute/path/to/tinyllama-q4_0.gguf \
  --reference results/m3-full-reference-v1 \
  --output results/full-numeric-fresh
```

Build the CPU reference first with `tools/build_llm_full_model_reference.py --cpu-build ... --model ... --output ...` when it is absent. The numerical producer authenticates that reference and its original model identity, discovers compiler dependencies before and during compilation, records runtime libraries, and rechecks bound inputs after execution. It rebuilds only the test harness and backend against the existing CPU libraries. Output must be a fresh directory under `results/`.

The unchanged acceptance limits are per-element `abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)` and relative L2 at most `2e-4`. A valid execution exceeding these limits writes a `NUMERICAL_FAIL` receipt and returns exit code 1. A malformed output or execution failure also returns nonzero; it does not produce a passing receipt. Host ASan/UBSan cover the compiled harness/backend, not the prebuilt CPU libraries.

## Current result: canonical production arithmetic

`results/m3-full-numeric-canonical-v1/receipt.json` passes the original gate.
All 72 outputs are bitwise equal to the CPU reference, including every layer
and all logits for positions 0/1 and reset 0. Receipt SHA-256:
`360a7172fd5f35e3a368609b649c70aec852548e1c685f51ef9e7af166ec31b5`.
The 465 same-input quantizations match upstream bytes, and 465 projections /
1,278,720 rows satisfy independent FP64 error bounds (maximum fraction 0.03555).

The production kernels now use explicit accumulation orders: full-mode RMS
sums rounded F32 squares in FP64 before F32 square root/division; Q4 uses eight
FMA accumulators with the CPU reduction tree; F16 score/value products use 32
FMA accumulators and a fixed reduction. No CPU reference activation is injected.
Small cancellation tests distinguish these orders from the former serial dots.
The earlier corrected tree RMS mock remains the legacy selected-layer model.

`results/llm-layer-fixture-v6` contains the fresh 13-symbol device module.
Initial admission and the exact Q4 byte-load guard pass, with zero LDS/private
storage/spills. New RMS uses 6 VGPRs; score/value kernels use 57 and Q4 uses 48.
This does not establish hardware FP64 behavior, hardware nonlinear rounding,
performance or native full-decoder acceptance. Validate the native smoke app
on .40 before the long profile. The historical diagnoses below explain the
failed arithmetic and remain separate evidence.

## Native smoke profile

The `model-full` package phase reads the external model at
`/data/hip/models/tinyllama-q4_0.gguf`. It runs tokens 0/1, clears actual decoder
memory and resets the backend, then repeats token 0. Its callback captures all
24 boundaries for each request: 72 files / 949,248 bytes, 2,394 active nodes and
2,199 launches. The native watchdog remains 120 seconds including model load.
The 514-request profile is a separate helper qualification; it is not selected
by the native title until elapsed time has been measured.

`llm-full-assets` authenticates the six existing modules and the complete CPU
reference without embedding the model. `llm-native-app LLM_NATIVE_APP_PHASE=model-full`
uses that aggregate through `LLM_NATIVE_APP_FULL_FIXTURE`; the normal model,
CPU build and output-directory overrides still apply. Use a fresh output and
results directory. An existing receipt is authenticated rather than silently
replaced; stale inputs require a fresh small fixture, not an unrelated full
dependency rebuild.

The full transcript parser checks real position commits, exact raw inventory,
balanced owners, restored free extent and finalization. The controller then
compares captured bytes against the frozen CPU reference using the same
element/L2 limits above and requires exact reset equality. Transport success
alone cannot establish native numerical acceptance. Commit and freeze the
candidate before a single bounded .40 run.

## Historical failure (2026-10-17)

`results/m3-full-numeric-v4/receipt.json` records 3 committed requests, 2,394 active nodes, 2,199 interpreted launches, 2,403 checked backend compute calls, zero CPU backend arithmetic, and 16 balanced allocations/frees. All 72 outputs were finite and reset repeated exactly. **59 outputs failed the unchanged numerical limits**, with final-layer relative L2 reaching about 6.97%. Receipt SHA-256: `0b0795363739b0a3b96dac73227ef0b5db40ee5cabc256702380517665c3978a`. That historical run used its bound local `results/m3-full-numeric-v4/run.py`; the portable producer above adds full CPU reference authentication and a failing process exit, without changing the numerical thresholds. Historical receipts are not rewritten.

The first position-0 divergence occurs in layer 4's FFN projections. Layers 0-3 agree within approximately 2-3e-7 relative L2. Layer 4 FFN normalization differs by only 2.12e-7 relative L2, but feature 210 crosses a Q8 rounding threshold:

| Input | Value | Block maximum | Scaled value | Q8 value |
|---|---:|---:|---:|---:|
| CPU reference | -0.0263056624681 | 0.393037617207 | -8.4999984408 | -8 |
| Host HIP interpretation | -0.0263056829572 | 0.393037647009 | -8.5000044168 | -9 |

For each identical input separately, exported upstream `quantize_row_q8_0` and production `layer_quantize_q8_cpu` produce bitwise-equal scales and int8 values after removing the unused Q8_1 sum field. Across the two inputs, exactly one compact Q8 byte differs. An independent FP64 Q4-by-Q8 dot-product oracle using the original layer-4 gate weights agrees with the CPU gate output within max absolute 1.73e-7 / relative L2 1.21e-7, and the interpreted gate output within 3.46e-7 / 1.36e-7.

This explains the first discontinuity as quantized-activation sensitivity to incoming floating-point variation. It does not establish that all later errors share that cause, and it does not justify accepting the final error or widening the gates. The exploratory captures and same-input oracle results are in `results/m3-full-stage4-diagnostic-v1/`; they are local diagnostic artifacts, not authenticated native evidence. Resolve the end-to-end numerical failure before attempting native full-model acceptance.

## Follow-up: RMS model and every projection

The original host RMS interpreter accumulated squares in FP64. Inspection of the qualified RMS kernel (`norm-isa.stdout`, instructions B444 through B5A8) shows 1024 F32 per-thread FMA accumulators, a 32-lane XOR reduction, then another 32-lane XOR reduction across waves, followed by F32 mean/epsilon operations and hardware reciprocal square root. The test interpreter now follows that reduction and rounding sequence. Its reciprocal square root is still a rounded host surrogate: the AMD hardware instruction is not emulated bit for bit. `tests/llm_rms_numeric_mock_test.cpp` compares the tree with an independent paired reduction over 64 varied vectors and demonstrates a real difference from FP64 accumulation.

The corrected interpreter's three-request run (`results/m3-full-numeric-rms-tree-v1`) reproduced the previous 72 files exactly and still returned `NUMERICAL_FAIL`. This is a test-model correction, not a solution to the numerical gate.

Add `--same-input-oracle` to the reproduction command for an additional, independent check inside the host test. For each activation vector, it compares production Q8 scale/int8 bytes with upstream quantization of the **same** vector. For every Q4 projection row, it independently evaluates the quantized dot product in FP64 and requires the F32 result to satisfy `gamma(2*blocks+3) * sum(abs(block_terms))`, with `gamma(n)=n*2^-24/(1-n*2^-24)`. It does not substitute CPU reference activations into the backend or change the end-to-end limits. Counts must be exactly 465 quantizations and 465 projections, covering 1,278,720 output rows for these three requests. The initial diagnostic passed all rows with maximum error/bound about 0.04760.

Layer-5 snapshots show the error is already present in its weighted attention normalization (relative L2 about 8.93e-4), then grows through Q/K/V and output projections. At position 0, its softmax probabilities are identical, while the value product reflects the differing V cache. The final layer-5 residual reaches relative L2 0.00704. The same-input projection checks cover these operations, but their success does not establish complete causal equivalence of all attention/nonlinear operations or make the final CPU comparison pass.
