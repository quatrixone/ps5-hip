# First native experiment sequence

Status: offline preparation only. No console run has occurred. The user selected
title `PPSA99012`. Firmware, service endpoints, lock ownership and package hashes must be recorded
in the local experiment receipt before execution. Use the shared-console runbook
referenced by AGENTS.md; no old project's environment or lock waiver applies.

## Case 1: CPU startup control

Hypothesis: the native module loader, system libc and pinned static C++ runtime
support the startup and error-handling mechanisms required by the GPU harness.

Candidate recipe:

```sh
python3 tools/build_native_canary.py --native-source /path/to/PS5_Vulkan --target gfx1013 --kernel runtime
```

This control does not read a GPU code object, call AGC, or allocate GPU memory.
The target string labels the candidate; it does not exercise that GPU ISA.
Native module import resolution still happens during title loading.

Expected fresh log: `/download0/ps5-hip-gfx1013-runtime.log` containing, in order:

```text
PS5HIP START target=gfx1013 kernel=runtime
PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass
PS5HIP PASS kernel=runtime gpu_api_calls=0
```

Acceptance requires the exact frozen bytes to be deployed and verified, title
entry, this newly created log, normal status-zero completion, title-specific
teardown and healthy declared services. The log alone cannot prove teardown.
An existing log makes the app refuse to run: the controller must first preserve
previous evidence and remove only this exact title-owned result path.

Rejection includes a loader failure, missing/partial log, constructor or exception
failure, unexpected process status, or unclean teardown. A missing log is not a
GPU diagnosis. It can precede main or indicate a failed log open. Capture the
bounded title-specific loader/service evidence and investigate offline.

External observation deadline: 20 seconds from the confirmed launch request.
No hardware controller is implemented in this repository yet; the shared-console
protocol must bound observation and close only the configured development title.

Next decision: proceed to one GPU case only after startup and teardown pass. If
startup fails, keep all GPU cases unrun and isolate the CPU/runtime boundary.

## Case 2: indexed store

Hypothesis: the selected HIP compiler profile and native launch state correctly
establish global pointers, kernarg SGPRs, workgroup/work-item IDs, wave32 execution
and coherent completion/readback on the observed PS5 environment.

Choose one of the already-built `gfx1013` or `gfx1030` candidates using the current
hardware/source evidence, record the choice, and inspect that exact code object's
ISA before running. Do not run both automatically or substitute a hand-written
shader as the acceptance result.

The harness dispatches two groups of 64 threads for 73 output elements. Expected
output is `output[i] == i ^ 0x5a17c9e3` for every `0 <= i < 73`. The 64-byte prefix
guard and every byte of the rounded allocation after the output must be unchanged.
The expected final record reports 73 exact values, unchanged guards and closed
allocations. Require normal title teardown and service health in addition.

The completion polling deadline is two seconds. Driver calls themselves are not
bounded by that timer. Use the same 20-second external observation deadline.
Uncertain submission or completion produces `QUARANTINED`, retains memory and
parks the title. This is a failed experiment, not cancellation or permission to
reuse those allocations. A partial log ending at SUBMIT is inconclusive/failed
according to the captured process and service evidence. Never auto-retry a
suspected panic.

Next decision: exact output plus guards and clean teardown permits the vector-add
case. A failure requires offline discrimination between entry state, ISA, memory
visibility and submission; two inconclusive attempts require reassessment.

## Case 3: vector addition

Keep the successful target and platform configuration. Change only the selected
kernel and its corresponding arguments/buffers. There are 73 elements:
`a[i] = i/2`, `b[i] = 2-i/4`, and expected `output[i] = 2+i/4`. These binary
fractions are exactly representable; compare all bits, not a loose tolerance.
Both input buffers must remain unchanged and all three full allocation guards
must pass. The same completion, external deadline, quarantine and teardown rules
apply. A pass still does not qualify general HIP, LDS, atomics or scratch.

## Freeze checklist still to complete offline

- Stage `PPSA99012` and verify its metadata/package using the native build tools.
- Commit source and record the selected build receipt, compiler/runtime inputs,
  executable, code object, title metadata and package SHA-256 values and sizes.
- Prepare title-specific deployment verification, log retrieval and close control.
- Record expected environment and declared services without probing the console.

Only after this preparation is concrete should console access be requested.
