# Current state

Updated: 2026-10-11, America/New_York.

## Actual status

- The provider-corrected image is deployed and byte-verified on .30, but its run
  was blocked before launch when unrelated PPSA99008 PID 355 became active.
  No controller was sent; both owned locks were released. The failed GPU image/log
  remain preserved. See `evidence/native-provider-fixed-no-run.json`.
- First GPU-candidate run on .30, FW 6.02, failed with CPU SIGSEGV at RIP zero
  after the pre-submit checkpoint. No GPU execution is proven. PPSA99014 PID 341
  stopped, sandbox/runtime/image were released, and declared services remained
  reachable. No retry or process kill was sent. See `evidence/native-indexed-store-first.json`.
- Current development identity: **PPSA99014**, selected by the user after a
  pre-existing PPSA99012 installation was found on .30. Controllers and the
  elevation helper are rebuilt for the new exact title.
- First PPSA99014 run on .40 registered and created its native process, then failed
  with PRX_SCE_MODULE_LOAD_ERROR before an application checkpoint. The kernel
  reported a missing sce_module companion. All declared services stayed reachable;
  exact-title close was sent, but normal exit/complete teardown is unproven.
  No GPU work ran. See `evidence/first-native-cpu.json`.
- The builder now reproduces the pinned, independently authored libc.prx loader
  companion. Its raw and signed hashes match the native template's known hashes;
  FSELF integrity passes. Staging rejects its absence. This corrected candidate
  is packaged as `gfx1013-runtime-4a0334b256d6` at source `1b6f717`, with clean
  filesystem/extraction and artifact checks. See `evidence/offline-loader-companion.json`.
  On .30 (FW 6.02), this candidate passed constructors, heap, exception unwinding,
  TLS and clock checks with zero GPU API calls, then exited abnormally with SIGSYS
  in libkernel after logging exit_value=0. Title stop, sandbox removal and image
  unmount were observed; services remained reachable. The CPU startup/exit gate
  still fails. See `evidence/native-cpu-shutdown.json`.
- On .40, PID 204 from the earlier loader failure remained present after one
  identity-guarded SIGKILL. No repeated termination or relaunch was attempted;
  complete teardown remains unproven. Do not use .40 for another experiment
  without establishing recovery. See the first-run evidence's cleanup followup.
- A PPSA99012 CPU image was uploaded and byte-verified on idle .40 (FW 12.70).
  ShadowMount mounted UFS2 but rejected registration because icon0.png was absent.
  No title was launched. The icon fix and staging regression test pass; the old
  uploaded image remains preserved. Evidence: `results/cpu-deploy-20261011T170003Z/`.

- Repository initialization and implementation planning are complete.
- G1 passed offline; G2 is in progress. G3-G12 remain planned; no GPU gate has passed.
- LLVM 18.1.8 builds real AMDHSA V5 device-only HIP canaries for gfx1013 and
  gfx1030. A host code-object inspector/materializer and sanitizer tests pass.
- Initial launch preparation and compute packet fragments pass six host test
  cases across the two targets; no fragment is submitted or GPU-qualified.
- Pinned SDK preparation and LLVM 18 PS5 cross-build succeed. The native memory
  library and CPU-only smoke executable link. Submission lifecycle and AGC adapter
  cross-compile; fifteen C++ host sanitizer tests pass, including timeout quarantine,
  driver failure injection, AGC ABI models, guarded canary oracles and log capture.
- Initial cache/state commands and compute-drain/completion packets are encoded.
  Indexed-store and vector-add native candidates for both targets link with real
  AGC module imports and convert to integrity-checked FSELF containers.
  Packaging/runtime/GPU behavior is unverified.
- A CPU-only startup control checks constructors, heap, exception cleanup, TLS and
  the clock. All candidates use exclusive persistent logs. Native conversion uses
  hash-verified static zlib built offline from pinned source.
- No HIP runtime or hardware-qualified SDK exists yet.
- User-directed storage revision: Eden's elevation client plus an exact-title
  Lapy owned-root helper; configurable ordinary results directory, no download0
  dependency. CPU and indexed-store native builds pass. The revised CPU package
  is frozen at source 955c989; filesystem and full extraction/hash checks pass.
  See `evidence/offline-elevation.json`. Its configured directory is
  `/data/ps5-hip/results`; this is a staging choice, not an executable constant.
  Nineteen host cases pass, including the exported Eden/Lapy wire contract.
- Preflight contacted both authorized consoles; PPSA99012 and PPSA99014 images
  were deployed on .40. Only PPSA99014 was launched, once, and failed in loading.
- The original `PPSA99012` staging identity is historical; use `PPSA99014`. Hash-verified staging
  and UFS2 packaging tools pass: five packages have clean filesystem checks and
  exact extracted-file hashes. Nineteen host test cases pass (sixteen C++, three Python).
- Pinned title-aware launch/close controllers cross-compile. Host mocks and a
  bounded application-log checker are implemented; neither sends console traffic.
- The user authorized 192.168.4.30 or 192.168.4.40 only when idle. Preflight under
  the protocol's workspace `lock.txt` found FTP, klog and elfldr reachable on both.
  The later preflight established shell focus after PPSA99008 stopped on both;
  .30 became busy again before title inspection. Locks were released after every
  bounded phase. Reported firmware: .30 = 6.02, .40 = 12.70. Both the protocol
  default lock and Eden's PS5-Claude lock are now coordinated.
- No hardware throughput, compatibility or stability claim belongs to this repository yet.

## Next action

G1 passed its offline acceptance. `evidence/offline-g1-acceptance.json` maps every
requirement to evidence and records the selected LLVM/LLD 18.1.8, AMDHSA V5,
wave32, gfx1013 indexed-store route. Four whole baseline objects reproduce in a
fresh authenticated Ubuntu userspace with networking and the development compiler
filesystem excluded. The WSL kernel/CPU are shared; no second physical host claim.
The corrected recipe verifies the actual LLVM 18 device linker. Older objects
built by unintended host LLD 21 are superseded; image `aaa74094...` is revoked.

The actual pinned TheRock Clang/LLD 24 tools compiled four comparison fixtures.
Before/after executable, resolved-library, source and inspector identities match.
Fresh output paths change compilation-ID symbols and associated ELF lookup tables;
code, descriptor and metadata sections match the first comparison. Whole-object
replay equality is false, and the extra fixed-ID diagnostic was inconclusive.
New target-ID spelling is rejected by our loader; resource registers and instruction
encodings also differ from LLVM 18. Keep LLVM 18 selected. See
`docs/THEROCK_BINARY_COMPARISON.md` and `evidence/offline-therock-verified-comparison.json`.

Full TheRock `amd-llvm` target completion and staged installation remain unverified.
That broader build is separate from G1's comparison of relevant compiler tools.
The existing build is under `/tmp/ps5hip-therock-comparison`; exec session `40611`
was running at this checkpoint. Inspect that build before restarting anything.
Detailed output: `build/logs/amd-llvm_build.log`; options/pins:
`evidence/offline-therock-build-setup.json`. Do not overwrite its existing logs.

The next action is one separately prepared, bounded corrected-provider experiment.
Offline analysis confirmed that `sceAgcSuspendPoint` was imported from the wrong
provider (`libSceAgcDriver` instead of the pinned reference's `libSceAgc`). Its null
tail-import fits the captured registers; runtime GOT contents were not captured.
See `docs/FIRST_GPU_FAULT.md`.

The correction is isolated at source `b8bc2b0`, parent `15bfc35`, in local worktree
`.local/g2-agc-provider-fix` on `development/g2-agc-provider-fix`. All nine native
application objects, the entire LLVM PIE, device code, helper and libc are identical
to the failed build. Only one of 116 native import bindings changes. The converted
ELF differs in 28 provider/hash/build-ID bytes, with no code or path-string changes.
The new stage is `.local/g2-agc-provider-fix/dist/PPSA99014/gfx1013-indexed_store-fb47201d65dc`,
image SHA `679fa7efaf0f4b7cc7889353d0ab2d565e72a8573055684cab256b46e6f0c442`.
FSELF, UFS2, extracted files and local controller/candidate verification pass;
26 CTests pass, including the actual native import provider regression. See
`evidence/offline-agc-provider-fix.json`. It has not run: deployment succeeded,
but the next preflight found PPSA99008 active. Use the existing deployment only
after a fresh idle check, with `results/run-provider-fixed-store.py`; do not rerun
its deployment script, which expects the superseded image and log. No unrelated
title may be closed. See `evidence/native-provider-fixed-no-run.json`.
The current host-test
cache selects that worktree's converted ELF for the artifact import check.

Do not rerun the failed candidate. It is revoked for new runs and preserved at source `15bfc35`, stage
`gfx1013-indexed_store-3d295773d2fb`, image SHA
`fb634e34a1a6420fbd257e3929a3fbd37b5be8f6121db1eb6ee43362015febbb`.
Corrected device bytes, native imports/startup, FSELF, UFS2 and extracted-file hashes
pass local verification. See `evidence/offline-diagnostic-gpu-package.json`.
Later host/ABI work has not rebuilt or substituted this candidate. One run on .30
passed CPU runtime checks, logged SUBMIT, then faulted with SIGSEGV/RIP zero.
The checkpoint precedes actual submission; GPU execution is unproven. Frozen
return addresses are `0x400c4e` and `0x40016a`; map them using the exact native ELF.
Raw evidence: `results/indexed-store-run-20261011T201531Z`; compact receipt:
`evidence/native-indexed-store-first.json`. System cleanup released the title,
sandbox and image, and all declared services stayed reachable; this is not normal
application teardown or a successful compute result.

Prepared deployment/observation scripts and their generator are hashed in
`evidence/offline-first-gpu-controls.json`. Offline idle-history, fake-socket and
real local timer checks pass. They coordinate both locks, verify/preserve the known
CPU image/log, recheck idle immediately before launch, observe for an absolute
20 seconds and close only PPSA99014 once with a separate eight-second deadline.
No automatic retry or guessed process termination. Read `docs/CONSOLE_WORKFLOW.md`,
the shared runbook and `.local/ENVIRONMENT.md` before traffic. These single-case
scripts have now run: the remote image is the failed GPU candidate, its result log
exists, and the successful CPU image/log remain preserved. Do not reuse them as
fresh-case scripts or erase that evidence.

The CPU shell-managed lifecycle passed once on .30, FW 6.02, at source `3346166`,
stage `gfx1013-runtime-9ac5d150b5db`, image SHA
`d9a9e95cfc3e10b0fd8b547e54da3d2898291700537a27ecd9da3d63a6cc9cb0`.
Runtime checks, callbacks, destructor/fini observation, shell-request return zero,
self-stop before host cleanup, sandbox/image release and service health passed.
See `evidence/native-shell-exit.json`. Ordinary C exit still faults with SIGSYS;
do not repeat the failed diagnostics. Threaded/framework cleanup is unqualified.
The failed .40 PID 204 teardown remains unresolved; establish recovery before reuse.

Offline work ahead of G2/G3/G4:

- All 26 host CTests pass, including registered hidden-geometry and native import coverage.
  Eight real ROCm 6.2.4 header-derived vector/dimension ELFs pass launch preparation,
  twelve inspector checks pass, and PS5 native libraries cross-build. Explicit and
  thirteen hidden arguments are separated; checked uniform 64x1x1 geometry is
  generated in zeroed 256-byte implicit storage. Unsupported layouts fail. See
  `evidence/offline-hidden-geometry.json`. Existing frozen titles do not exercise it.
- Twelve real combined HIP header fixtures compile for Linux/PS5 and both targets.
  Authentic matching headers, host registration/fatbinary layouts, device ABI flags
  and a source-derived workitem-only library are recorded in
  `docs/HIP_HOST_ABI_SPIKE.md`. Full OCKL still fails; no HIP runtime is linked or run.
- Wave32 full/partial-lane probes compile at O1/O2 for both targets. Both guarded
  gfx1013 native profiles link with checked imports/startup/FSELF; gfx1030 remains
  compile-only. Neither is packaged or executed. See `evidence/offline-wave-native-link.json`.
- Generic volatile private lowering remains a G3 issue. An isolated IR contrast
  preserves volatile operations and emits buffer instructions; reserved flat-init
  registers are overwritten before use, with no hardware flat-scratch setup.
  Native scratch backing, descriptors, offsets, state and lifetime remain
  unimplemented. See `docs/VOLATILE_PRIVATE_LOWERING.md`; no workaround is accepted.

Source notices are inventoried, including the exact ps5log GPL-3.0-or-later notice.
Linked archive-member accounting and binary redistribution remain release work.
The first kernel's instruction/entry contract is in `docs/INITIAL_CAPABILITY_MATRIX.md`.
G2 still needs exact device output, guards and repeated clean lifecycle coverage;
G3 additionally requires LDS/barriers, atomics, waves and deliberate spilling.
Pushes remain reserved for coherent milestones.

## Entry points

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Research baseline](docs/RESEARCH_BASELINE.md)
- [Native launch contract](docs/NATIVE_LAUNCH_CONTRACT.md)
- [Native build](docs/NATIVE_BUILD.md)

Read only the active goal and relevant source slices. Raw logs stay in ignored
`results/`; compact, sanitized validation receipts live under `evidence/`.
`evidence/offline-resource-probes.json` records the current source hash, compiler
versions, object hashes and instruction inventories without local user paths.
`evidence/offline-native-canaries.json` records linked native candidates and their
source/dependency hashes; it is not hardware qualification.
`evidence/offline-native-packages.json` records all five `PPSA99012` package
manifests, hashes, source commit and the pinned image tool's binary provenance.
`evidence/offline-controllers.json` records controller hashes and host validation.

## Milestones

- 2026-10-11 | planning | twelve measurable goals and pinned research references recorded | next: G1 offline inventory
- 2026-10-11 | G1 progress | arithmetic and resource probes compiled for two targets; host parser checks pass; private-array flat-scratch incompatibility identified | not a completed gate
- 2026-10-11 | G1 progress | unmodified compiler emits buffer scratch for explicit-private accesses and 52 real VGPR spills; generic private pointers remain unresolved | not hardware-qualified
- 2026-10-11 | offline milestone | 0c2e8c1 | PPSA99012 | packaged, no-run: five verified images and fourteen host tests | evidence/offline-native-packages.json | next: console controller

- 2026-10-11 | CPU lifecycle milestone | 3346166 | PPSA99014 | one shell-managed startup/finalization/self-close passed on FW 6.02; 22 host cases pass | evidence/native-shell-exit.json | no G1-G12 gate passed

- 2026-10-11 | G1 passed offline | LLVM18 + gfx1013 selected; isolated reproduction, TheRock comparison, 25 host tests | evidence/offline-g1-acceptance.json | next: bounded G2 store run
- 2026-10-11 | G2 failed | 15bfc35 | FW6.02 PPSA99014 | CPU SIGSEGV/RIP0 after SUBMIT; title released, services reachable | evidence/native-indexed-store-first.json | next: offline fault mapping
