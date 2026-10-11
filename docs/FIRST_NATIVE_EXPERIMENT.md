# First native experiment sequence

Status: the provider-corrected indexed-store passed one GPU dispatch and clean
shell-managed lifecycle on firmware 6.02; see `evidence/native-provider-fixed-store.json`.
The preceding candidate failed with CPU SIGSEGV at RIP zero after its SUBMIT
checkpoint. Do not rerun that failed image; see `evidence/native-indexed-store-first.json`.
Vector-add remains unrun. Ordinary C exit still faults; see the shutdown investigation
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

This case follows corrected indexed-store success: exact output and guards,
clean title teardown and healthy services. That prerequisite passed once in
`results/provider-fixed-store-run-20261011T203845Z`. A fresh idle/identity preflight
is still required for vector-add; offline readiness does not establish its
execution result, and full G2 remains open.

Keep the successful target and platform configuration. Change only the selected
kernel and its corresponding arguments/buffers. There are 73 elements:
`a[i] = i/2`, `b[i] = 2-i/4`, and expected `output[i] = 2+i/4`. These binary
fractions are exactly representable; compare all bits, not a loose tolerance.
Both input buffers must remain unchanged and all three full allocation guards
must pass. The same completion, external deadline, quarantine and teardown rules
apply. A pass still does not qualify general HIP, LDS, atomics or scratch.

### Repeated vector candidate

The next frozen experiment combines this vector oracle with two serialized
dispatches through the same backend/queue, then two separately observed title
cycles of the same image. Each dispatch allocates and poisons fresh guarded
buffers; the second output cannot pass merely by retaining the first output.
Hypothesis: the qualified pointer/entry contract supports exact vector loads,
FP32 addition and stores, and completed queue ownership can be reused safely.

Source `c37f0db` adds only the profile to the successful provider-corrected
baseline. The stage is
`.local/g2-agc-provider-fix/dist/PPSA99014/gfx1013-vector_add_twice-70a3e57c6792`,
image SHA `1487a21ba6bdd65f61a67ad6739e25f8cbc9306f2fcda5602403a0f51fec67f5`.
Only the canary application object differs; device bytes and all 116 imports are
unchanged. See `evidence/offline-vector-twice-package.json`.

Expected: ordered dispatch 1/2 and 2/2 records, 73 exact values and all input/guard
checks per dispatch, allocation closure, a final 146-value summary, complete
shell-managed shutdown, title/runtime/image release and healthy services. The
in-title timeout remains two seconds per dispatch; the host launch/observation
window is 20 seconds with separately bounded exact-title cleanup. These limits
do not cancel GPU work or guarantee a driver call returns.

Any failed, missing or contradictory criterion rejects the cycle and prevents
cycle two. Preserve the store image/log before deployment. Preserve cycle one's
exact log before cycle two, which requires a reviewed acceptance receipt and a
fresh idle/identity check. No automatic retry or unrelated-title closure.
If both cycles pass, audit the unchanged G2 requirements for the explicitly
declared arithmetic launch set; hidden geometry, other workgroup shapes, waves,
scratch and a public HIP host runtime remain separate qualifications.

## Offline freeze state and remaining work

- Corrected-provider candidate: source `b8bc2b0`, stage
  `.local/g2-agc-provider-fix/dist/PPSA99014/gfx1013-indexed_store-fb47201d65dc`,
  image SHA `679fa7efaf0f4b7cc7889353d0ab2d565e72a8573055684cab256b46e6f0c442`.
  Its nine application objects, LLVM PIE and device/helper/libc bytes match the
  failed candidate. The only semantic change is the `sceAgcSuspendPoint` import
  provider, corrected to `libSceAgc`. Offline package checks and 26 host tests pass;
  one later idle-console run passed exact output, guards and clean teardown. See
  `evidence/offline-agc-provider-fix.json` and `evidence/native-provider-fixed-store.json`.
  The preceding busy-console no-run remains recorded separately. The successful
  image and log must be preserved before replacement; existing single-case
  scripts now have stale image/log preconditions and must not simply be rerun.

- Provider-fixed vector candidate: source `b8bc2b0`, stage
  `.local/g2-agc-provider-fix/dist/PPSA99014/gfx1013-vector_add-93752ddb9ec6`,
  image SHA `fb2ecdc7feded72005ef5c228b02511071de707ef87ac47c1728a7e49f55fb57`.
  Source identities, all imports and device/helper/libc bytes match the corrected
  store; only the vector-selected canary object differs among nine application
  objects. FSELF integrity, filesystem/extraction and main local case/controller
  verification pass. It is offline-ready, not deployed or run; the corrected store
  prerequisite passed once, and fresh idle/identity checks remain required.
  Exact 73-value bitwise,
  input-preservation and full-allocation guard oracles are recorded in
  `evidence/offline-provider-fixed-vector.json`; its manifest/package are appended
  to `evidence/offline-native-packages.json` with prior entries preserved.
  To minimize console cycles, prepare a fixed two-dispatch vector profile next;
  retain this single-dispatch image as the offline control. Qualify the new
  profile across separately observed clean cycles before closing full G2.

- Failed first GPU candidate: `PPSA99014/gfx1013-indexed_store-3d295773d2fb`,
  source `15bfc358f014b4231fcf9f84918b79b1b301008f`, shell-managed shutdown,
  image SHA-256 `fb634e34a1a6420fbd257e3929a3fbd37b5be8f6121db1eb6ee43362015febbb`.
  It adds exception-reason logging before quarantine without changing ownership
  or retry behavior. The preceding `1288205d1a2a` package is preserved as history.
  Corrected LLVM/LLD 18.1.8 device object and native imports were inspected;
  filesystem, full extraction and local candidate/controller verification pass.
  See `evidence/offline-diagnostic-gpu-package.json`. It ran once and failed with
  a CPU null-control-flow fault; preserve it and diagnose offline before any new
  candidate. G1 passed offline (`evidence/offline-g1-acceptance.json`).
  The earlier image with digest `aaa74094...`
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
