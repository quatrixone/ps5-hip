# Combined G2 arithmetic experiment

Status: two newly controlled prospective cycles on .40 passed independent review;
G2 passed for the declared arithmetic fixture. See the six-item mapping and both
cycle bindings in [`native-g2-acceptance.json`](../evidence/native-g2-acceptance.json).
The earlier combined-image observation on .40 remains partial-pass. Both arithmetic
dispatches, all 146 values, guards, allocation closure and application finalization
passed. OS process termination and EndAppMount completion were observed, but the
older deployed loader retained its registered image and emitted no explicit
layer-release records. See
[`native-arithmetic-fw1270-cycle1.json`](../evidence/native-arithmetic-fw1270-cycle1.json).
Its original release criterion was not met; its classification is unchanged.

## Explicit lifecycle correction for future .40 controls

The original control below required disappearance of sandbox, runtime and registered
image layers after every exit. [Exact upstream1.6beta16 source](SHADOWMOUNT_16_LIFECYCLE.md)
shows that valid registered images and `/system_ex/app/<title>` mappings persist
deliberately. That literal criterion therefore tested the loader's registration
policy as well as application cleanup. The implementation plan requires clean
launch/close cycles, and the shared runbook requires runtime-layer release when
available. Future .40 controls must use the following explicit, observed lifecycle
contract. The saved first-cycle partial-pass and its original criterion remain
unchanged; this correction does not retrospectively accept it.

1. Verify the frozen deployed image bytes and title metadata independently of
   mount names. The image mount suffix hashes its source path, not image content.
2. Collect a fresh pre-launch registration/mount baseline. Record exact source,
   target, filesystem type, fsid, flags and multiplicity for visible title records.
   Report the observer's namespace and ABI limits; empty output proves no global
   absence. Do not launch if the diagnostic is incomplete or idle is ambiguous.
3. Require one fresh own-title PID/application start and matching exit, all-processes
   exit, title sandbox ProcessTerm and completed OS EndAppMount. Require the full
   arithmetic transcript, allocation closure and every finalization checkpoint,
   self-stop before cleanup, and no observed fault or cleanup failure.
4. Compare post-exit registration observations with the pre-launch baseline.
   Deliberately persistent loader registration must have unchanged identity and
   multiplicity. New or changed retained title records, including visible sandbox
   or overlay rows, reject the cycle. A positive sandbox observation followed by
   disappearance in the same view strengthens this evidence when available.
5. Require declared service health and release of only the operation's owned locks.
   Accept only the observed application/OS lifecycle and baseline equivalence;
   do not claim global mount absence, detached-process enumeration or release of
   references outside the observer's view. The snapshot itself always reports
   release unknown and never decides gate acceptance.

The matching tag and banner establish a relevant policy comparison, not the
deployed loader's exact binary identity. Fresh measured behavior must agree with
this contract. Before another arithmetic/private launch, prepare and independently
review controls that implement these prospective checks. Preserve original raw
results, scripts and one-use markers. No loader update, setting change or forced
unmount is part of this correction. Two independently reviewed clean cycles of
the same combined image remain required for G2.

The first read-only snapshot on .40 returned `count_bound` for both mount
observations. It supplied no rows or usable baseline, while confirming only the
historical PID134/application-ID queries and service health. See
[`native-teardown-snapshot-fw1270.json`](../evidence/native-teardown-snapshot-fw1270.json).
That obstruction was resolved by the frozen v3 collector: two complete snapshots
each returned 1110 records, including the same registered UFS image and persistent
title nullfs mapping. See
[`native-teardown-v3-fw1270.json`](../evidence/native-teardown-v3-fw1270.json).
This supplies a usable visible registration baseline, not proof of global mount
absence or release. New arithmetic controls must still collect fresh pre/post
snapshots and satisfy the full prospective contract above. The diagnostic's
initialized zero counters must never be treated as mount absence or successful
release; historical PID134 queries cannot prove a future title's termination.

The sections below retain the original frozen experiment contract. The explicit
prospective correction above governs newly prepared .40 controls.

The first prospective host controller incorrectly required a zero launch result.
The frozen native controller returns success for a nonnegative signed result,
which identifies the launched application. That interrupted attempt remains
inconclusive. The corrected controls validate the full response and match the
returned application ID to fresh OS/ShadowMount evidence; direct socket-response
regressions cover real successful IDs, signed errors and malformed responses.

G2 item 5 requires one frozen application to execute indexed-write followed by
vector-add. The successful store and vector images establish useful controls,
but their separate lifecycles do not satisfy that requirement. See
[`g2-readiness-audit.json`](../evidence/g2-readiness-audit.json). The pending
vector-only repeat is superseded as the next G2 experiment, not erased.

## Candidate and hypothesis

Build `arithmetic_suite` from the proven `c37f0db` arithmetic source in the isolated
`development/g2-arithmetic-suite` branch. Reuse its compiler output, loader,
launch encoder, memory allocator, submission queue and corrected import provider.
The only native application change should be its harness object. Verify this
against the earlier native object hashes before accepting a package.

The hypothesis is that the already proven store and vector entries execute
correctly in sequence through one backend and queue in one application. The
candidate runs exactly these two dispatches:

| Dispatch | Kernel | Exact oracle | Launch |
| --- | --- | --- | --- |
| 1 | `indexed_store` | `output[i] == (i ^ 0x5a17c9e3)` for 73 indices | 64x1x1 threads, 2x1x1 groups |
| 2 | `vector_add` | Binary32 sum `i * 0.25 + 2`, with both inputs unchanged | Same shape and 73 indices |

Each dispatch uses fresh poisoned allocations, validates the entire output and
guards, and closes its resources only after observed completion. The final suite
requires 146 exact values and both ordered per-dispatch passes. A marker or final
summary alone does not pass. This experiment does not establish persistent-buffer
reuse, a GPU producer/consumer dependency or asynchronous operation.

The package must be frozen before deployment. Its committed catalog entry must
bind source revision, manifest, application files, helper/libc, code object,
package and exact-title controllers. Both build and manifest must specify integer
`dispatches: 2` and `dispatch_kernels: [indexed_store, vector_add]`. Freeze the
candidate branch's checker/verifier hashes in the operation controls as well.
Until these identities and the offline tests are recorded, controls remain inert.

The implementation is frozen at `5b74ac8`. The full 24-test sanitizer run passed
after all production edits; a later test-only extension passed both affected
target cases. It injects a second-dispatch submit error, kick error and timeout,
checks first-dispatch cleanup, and verifies seven retained GPU-owned allocations
at quarantine entry with no vector or suite pass. Ten PS5 harness compile-only
variants also pass. See [`offline-arithmetic-suite-host.json`](../evidence/offline-arithmetic-suite-host.json).
The routing model uses synthetic GPU addresses and host-backed allocation doubles;
it does not execute shader ISA, title startup, or recovery after a real GPU fault.

## Preservation and console boundary

Use PPSA99014 on the user's currently selected console, .40 (firmware 12.70).
The October 12 status control established that the earlier failure PID204 was
absent; the later deployment stopped before mutation because another title was
active. See `evidence/native-private-buffer-console-no-run.json`. Read the current
runbook and local environment before contact. Acquire both shared locks with
exclusive creation and refresh idle/recovery checks before each operation; never
hold locks during offline work.

On .30, before replacing anything, verify the installed successful vector image
`1487a21b...`, its 854-byte application log `a0f414a2...`, the title metadata and
mount link. The full identities and independently reviewed first-cycle acceptance
are retained locally and in the existing evidence. On .40, preserve its existing
title image and applicable result bytes under the current reviewed deployment
control's identities. Verify and preserve the image
and log under hash-suffixed names; refuse conflicting backups or changed bytes.
Earlier store, CPU and failed-image backups remain untouched. An incomplete
promotion leaves its phase/evidence for offline review; never delete backups or
automatically repeat a partially completed deployment.

Upload to a fresh temporary name, verify bytes, recheck idle state and all
preservation preconditions, then promote and verify the new package. Require
title-specific registration evidence before selecting the separate run operation.
Deployment never launches the title. Never close an unrelated application to
obtain an idle console.

## Two separately reviewed cycles

For each cycle, recheck deployed bytes, metadata, mount link and result freshness.
Capture klog before launch. Recheck idle state immediately before sending the
title-specific launch controller. The launch/observation deadline is 20 seconds;
the application's completion polling allowance is 2 seconds per dispatch.
These host deadlines do not make a blocked native driver call cancellable.

Require the exact application transcript, all finalization checkpoints, title
self-stop before host cleanup, sandbox/runtime/image release and declared service
reachability. Use the existing exact-title close controller once with a separate
8-second allowance, then release only locks still bearing this operation's token.
No guessed process kill, reboot, setting change or automatic suspected-panic retry.

Audit cycle one before making cycle two runnable. Bind its acceptance receipt and
every referenced raw hash; verify the remote first-cycle log, preserve it under
its hash, and check idle again before the second launch. Run the same frozen image
without rebuilding, repacking or redeploying. Each cycle must supply its own fresh
lifecycle and service evidence. Never chain launch operations automatically.

Busy/ambiguous idle state, identity mismatch or backup conflict means **no-run**.
Wrong output, modified guards, failed lifecycle or service failure means **failed**.
Missing or contaminated evidence after execution means **inconclusive**, not pass.
Any failure or inconclusive execution returns to offline analysis. Service checks
prove reachability only. The current deployment control bounds individual FTP
operations with absolute deadlines; deployment and registration have separate
bounds and are not part of the 20-second launch observation.

Only two independently reviewed clean cycles of this combined image, together
with the existing parser/entry-state/rejection evidence, can close G2. Map the
unchanged plan's requirements explicitly before recording that acceptance.
G3 scratch, wave operations, LDS, generic pointers and numerical modes remain
separate qualification work.
