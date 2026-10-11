# First native experiment sequence

Status: the CPU shell-managed lifecycle passed one run on firmware 6.02; GPU
cases remain unrun. Ordinary C exit still faults; see the shutdown investigation
and `evidence/native-shell-exit.json`. The user selected
title `PPSA99014` after a PPSA99012 collision. Firmware, service endpoints, lock ownership and package hashes must be recorded
in the local experiment receipt before execution. Use the shared-console runbook
referenced by AGENTS.md; no old project's environment or lock waiver applies.

## Case 1: CPU startup control

Hypothesis: the native module loader, system libc and pinned static C++ runtime
support the startup and error-handling mechanisms required by the GPU harness.

The CPU shell-managed control already passed; do not rerun it or the failed
ordinary-exit diagnostics merely to begin the GPU phase. Its reproducible recipe is:

```sh
python3 tools/build_native_canary.py --native-source /path/to/PS5_Vulkan --target gfx1013 --kernel runtime --shell-exit-probe
```

This control does not read a GPU code object, call AGC, or allocate GPU memory.
The target string labels the candidate; it does not exercise that GPU ISA.
Native module import resolution still happens during title loading.

The current harness first requests filesystem access using the bundled Lapy
helper, then creates a fresh log in the directory selected with staging's
`--results-dir`. The directory must already exist. Elevation failure stops before
the runtime/GPU checks. Include helper lifecycle in klog acceptance.

Expected fresh log: `<results-dir>/ps5-hip-gfx1013-runtime.log` containing, in order:

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
Pinned title-aware controllers are available; the shared-console protocol must
bound observation and close only the configured development title.

Next decision: proceed to one GPU case only after startup and teardown pass. If
startup fails, keep all GPU cases unrun and isolate the CPU/runtime boundary.

## Case 2: indexed store

Build this candidate explicitly with `--shell-exit-probe`. On successful GPU
completion, the queue checks command and marker allocation teardown, and the
harness validates output and closes kernel/code/data allocations before returning
zero. The shell barrier then observes application callbacks, runs application fini
and requests closure. Require all six shutdown/shell records from the CPU
experiment after the GPU PASS record, title self-stop before host cleanup, no
fatal signal/loader error, sandbox/image release and healthy services. This is
an explicit shell-managed finite-job contract; it does not qualify ordinary C exit.

If command/marker cleanup fails after completion, the harness reports failure and
returns nonzero; the barrier parks for host cleanup. If completion is uncertain,
quarantine retains all potentially active allocations and never enters the barrier.
Neither path may count as success or trigger an automatic retry.

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

## Offline freeze state and remaining work

- Current next GPU candidate: `PPSA99014/gfx1013-indexed_store-3d295773d2fb`,
  source `15bfc358f014b4231fcf9f84918b79b1b301008f`, shell-managed shutdown,
  image SHA-256 `fb634e34a1a6420fbd257e3929a3fbd37b5be8f6121db1eb6ee43362015febbb`.
  It adds exception-reason logging before quarantine without changing ownership
  or retry behavior. The preceding `1288205d1a2a` package is preserved as history.
  Corrected LLVM/LLD 18.1.8 device object and native imports were inspected;
  filesystem, full extraction and local candidate/controller verification pass.
  See `evidence/offline-diagnostic-gpu-package.json`. It has not run. G1 passed
  offline (`evidence/offline-g1-acceptance.json`); reverify the exact candidate and
  current environment before console use. The earlier image with digest `aaa74094...`
  is revoked in the package catalog because it used the unintended host linker.

- Superseded CPU control (PPSA99012): `gfx1013-runtime-9914258279a8`, source `955c989`.
  Includes exact-title `lapy.elf` and `results-dir.txt`; the staged configuration
  selects `/data/ps5-hip/results`. Another normalized absolute directory can be
  selected by restaging, which creates a new package identity. UFS2 check, complete
  extraction comparison and local case verification pass. Compact provenance is
  in `evidence/offline-elevation.json`. Its image mounted on .40, but registration
  failed for a missing icon; no launch occurred. The next candidate uses PPSA99014
  and includes the required 512x512 icon in its hashed inventory.

- Five earlier `/download0` candidates were staged and packaged from source commit
  `0c2e8c14d5cdd581002b5d315889d67509e9d0fd`. Their manifests and package hashes are
  in `evidence/offline-native-packages.json`; each filesystem check and full-file
  extraction comparison passed. They are superseded for new tests by the requested
  configurable-path/Lapy integration. This is offline evidence only.
- The CPU lifecycle run established title-specific deployment, log retrieval and
  close observation. Adapt its identity guards to the current frozen GPU image;
  do not rerun old deployment scripts with stale expected image hashes.
  Prepared scripts now verify all unmatched title/PID starts and recheck idle
  immediately before launch. Controller and FTP observation I/O share an absolute
  20-second deadline; exact-title cleanup has a separate eight-second deadline.
  Timeout preserves partial evidence and discards an ambiguous FTP stream.
  Extracted-helper/fake-socket/timer checks pass without console traffic; hashes
  and remaining prelaunch I/O limits are in `evidence/offline-first-gpu-controls.json`.
- The local environment records endpoints and locks. Recheck readiness only when
  the offline milestone is complete and a bounded console experiment is ready.

Console access is authorized on either specified host only when idle.
