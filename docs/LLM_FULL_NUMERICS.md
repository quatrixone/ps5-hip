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

## Current finding (2026-10-17)

`results/m3-full-numeric-v4/receipt.json` records 3 committed requests, 2,394 active nodes, 2,199 interpreted launches, 2,403 checked backend compute calls, zero CPU backend arithmetic, and 16 balanced allocations/frees. All 72 outputs were finite and reset repeated exactly. **59 outputs failed the unchanged numerical limits**, with final-layer relative L2 reaching about 6.97%. Receipt SHA-256: `0b0795363739b0a3b96dac73227ef0b5db40ee5cabc256702380517665c3978a`. That historical run used its bound local `results/m3-full-numeric-v4/run.py`; the portable producer above adds full CPU reference authentication and a failing process exit, without changing the numerical thresholds. Historical receipts are not rewritten.

The first position-0 divergence occurs in layer 4's FFN projections. Layers 0-3 agree within approximately 2-3e-7 relative L2. Layer 4 FFN normalization differs by only 2.12e-7 relative L2, but feature 210 crosses a Q8 rounding threshold:

| Input | Value | Block maximum | Scaled value | Q8 value |
|---|---:|---:|---:|---:|
| CPU reference | -0.0263056624681 | 0.393037617207 | -8.4999984408 | -8 |
| Host HIP interpretation | -0.0263056829572 | 0.393037647009 | -8.5000044168 | -9 |

For each identical input separately, exported upstream `quantize_row_q8_0` and production `layer_quantize_q8_cpu` produce bitwise-equal scales and int8 values after removing the unused Q8_1 sum field. Across the two inputs, exactly one compact Q8 byte differs. An independent FP64 Q4-by-Q8 dot-product oracle using the original layer-4 gate weights agrees with the CPU gate output within max absolute 1.73e-7 / relative L2 1.21e-7, and the interpreted gate output within 3.46e-7 / 1.36e-7.

This explains the first discontinuity as quantized-activation sensitivity to incoming floating-point variation. It does not establish that all later errors share that cause, and it does not justify accepting the final error or widening the gates. The exploratory captures and same-input oracle results are in `results/m3-full-stage4-diagnostic-v1/`; they are local diagnostic artifacts, not authenticated native evidence. Resolve the end-to-end numerical failure before attempting native full-model acceptance.
