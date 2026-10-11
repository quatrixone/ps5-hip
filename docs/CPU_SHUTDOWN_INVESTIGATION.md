# CPU shutdown investigation

The first corrected PPSA99014 package passed its runtime checks on firmware 6.02,
then received SIGSYS in libkernel. `evidence/native-cpu-shutdown.json` records the
exact package and raw-log hashes. This is a failed CPU lifecycle test, not a GPU
result. The earlier firmware 12.70 loader failure remains a separate case.

## Offline findings

The pinned `app_crt.cpp` registers the loader teardown and `_fini` with `atexit`,
runs `main`, calls the weak `catchReturnFromMain` hook, and calls imported `exit`.
Disassembly of the tested LLVM executable places the return address of that
final call at offset 0x175 (native address 0x400175). This matches the saved stack
trace. The fault instruction address is in libkernel, not that return address;
the trace does not establish that `exit` returned.

The local native boilerplate's deployment and troubleshooting documentation
warns against returning from `main` under an unsupported loader lifecycle. The
Vulkan reference's hardware findings also report SIGSYS during exit. These are
supporting observations, not permission to classify a crash as success. Neither
identifies the failing operation in our title. Do not infer that elevation or
the loader companion is the cause merely because both are present.

## Next experiment: observe, do not replace, termination

Hypothesis: normal C/C++ exit processing reaches user callbacks and static
destructors before failing in the platform termination path. A competing
explanation is failure before those callbacks run.

The CPU-only build adds `native_shutdown_probe.cpp`. Its strong implementation
of the existing startup hook registers one function-local static destructor and
one `atexit` callback after `main` returns. It writes and syncs three checkpoints:

1. `main-return=0`: registration succeeded and the startup hook was reached.
2. `atexit=observed`: the C exit callback ran.
3. `static-dtor=observed`: that specific C++ static destructor ran.

The observer uses descriptor writes rather than stdio or allocations during
teardown. It leaves startup's `exit(status)` and loader callbacks unchanged.
The host child-process test verifies ordering and the nonzero-main path. It does
not validate the console ABI. The log checker requires all three records when
`--shutdown-probe` is specified, but still cannot certify native process exit.
Observing this destructor does not prove every global or thread-local destructor
ran, or that the loader teardown succeeded.

Before hardware: commit and freeze the candidate, verify package extraction and
hashes, and record the exact candidate in the experiment receipt. Use .30 only
after fresh idle checks under both shared locks. Preserve its prior result and
known image. Do not use .40 while the earlier process teardown is unproven.

Bound: one launch, at most 20 seconds of application observation, then the exact
title's existing close control if necessary. Capture kernel logs before launch,
retrieve the fresh result, and verify title stop, mount release and service health.
Do not retry a crash. The new bytes are an observer experiment, not a proposed
fix or a retry of the original hypothesis unchanged.

Rejection: any fatal signal, missing runtime checkpoint, persistence failure,
unexpected order, timeout or incomplete teardown rejects a lifecycle pass.
If all observer checkpoints precede SIGSYS, investigate later runtime/loader
termination. If they do not, investigate the first missing callback boundary.
Neither outcome authorizes a GPU test. Platform-managed application close is a
separate lifecycle to qualify explicitly; it does not prove return-from-main
semantics needed by future framework clients.

## Observed result

The frozen observer at source `c0bf77e`, image SHA
`e9025b4962279785ebdeeb0f4a65bdf4a55ecaf4269a637944c9a4b12845842b`, ran once on
.30, firmware 6.02. All three observer checkpoints appeared in order. PID 204
then logged exit_value=0 and received SIGSYS at the same libkernel address
0x8000003ac, with startup return address 0x400175. The title stopped and its
sandbox and image mounts were released; FTP, klog and elfldr remained reachable.
Evidence: `evidence/native-shutdown-observer.json`.

This rejects failure before these user callbacks. It leaves later destructors,
the loader callback and platform process termination unresolved. No GPU work ran.
The next step is offline analysis of that later boundary, not another run of this
observer. The failed PID 204 on .40 is a separate process on a different console;
its recovery is still unproven.
