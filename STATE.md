# Current state

Updated: 2026-10-12, America/New_York.

## Actual status

- G1 passed offline; G2 remains in progress. G3-G12 are not complete. There is
  no general HIP runtime, framework port or hardware-qualified SDK yet.
- Three native HIP-compiled dispatches have passed on .30, firmware 6.02:
  one indexed-store at source `b8bc2b0`, then two vector additions in one title
  cycle at `c37f0db`. Each checked 73 exact values, full guards and allocation
  closure. Both titles finalized and self-stopped before host cleanup, released
  sandbox/runtime/image layers, and left all declared services reachable.
  See `evidence/native-provider-fixed-store.json` and
  `evidence/native-vector-twice-cycle1.json`. These observations cover only the
  declared gfx1013, wave32, explicit-argument 64x1x1-thread arithmetic fixtures;
  no general ISA, coherence, throughput or stability claim follows.
- Strict G2 audit found that numbered item 5 requires one frozen application
  candidate to execute indexed-write followed by vector-add. Existing successes
  use two images, so another vector-only cycle cannot close G2. Prepare a combined
  arithmetic suite using the unchanged proven kernels/core, then qualify two
  clean cycles of that same image. The earlier vector cycle-two control was
  blocked before launch or log preservation by unrelated
  PPSA99930 PID394. Both locks were released. See
  `evidence/native-vector-twice-cycle2-no-run.json`. The successful vector image
  and original first-cycle log remain installed; preserve them before replacement.
- Identity is **PPSA99014**; PPSA99012 was occupied. Use the Eden client and the
  bundled exact-title Lapy helper with configurable ordinary results storage.
  The current staging choice is `/data/ps5-hip/results`, not a compiled path.
- The combined `arithmetic_suite` is frozen and packaged at isolated source
  `5b74ac8`; image `28262f78...` runs store then vector on the same queue with
  fresh allocations and 146 exact values. Only the native harness object differs
  from `c37f0db`; device/core/helper/libc/import identities match. Host routing,
  corruption and second-dispatch quarantine checks pass, as do native package
  checks. See `evidence/offline-arithmetic-suite-native-readiness.json` and
  `docs/G2_ARITHMETIC_ACCEPTANCE.md`. It has not been deployed or run.
- Its deployment preflight at 22:00 UTC returned **no-run**: unrelated
  PPSA99008 PID404 was active on .30. No remote artifact read/upload/rename,
  launch or close followed; both locks were released. See
  `evidence/native-arithmetic-deploy-no-run.json`. Do not poll for availability
  while independent offline G3 work can advance.
- CPU constructors, heap, exceptions, TLS and clock checks pass under the finite
  shell-managed lifecycle. Ordinary C exit still faults with SIGSYS; threaded or
  framework teardown is unqualified. See `evidence/native-shell-exit.json` and
  `evidence/native-cpu-shutdown.json`. Do not repeat the failed exit diagnostics.
- The user now selects **.40 for useful milestone tests**. A read-only status
  check at 12:54 UTC on October 12 established that its earlier loader-failure
  PID204 was absent with ESRCH and both running/title application IDs were absent.
  The subsequent arithmetic deployment preflight at 13:07 UTC returned **no-run**:
  unrelated PPSA99008 PID117 was active. No upload, rename, directory creation,
  launch or close followed; both locks were released. See
  `evidence/native-private-buffer-console-no-run.json`. Before a later launch,
  recheck idle state and recovery under both locks. Do not close another title,
  kill a guessed PID, reboot or change settings.
- Main has 27 passing host CTests. The exact frozen vector source also passes
  ten parser/resource/launch/memory/submission/AGC sanitizer cases independently;
  see `evidence/offline-vector-repeat-host.json` and
  `evidence/offline-candidate-host-c37f0db.json`. Host checks are not GPU evidence.
- Experimental buffer scratch is implemented and packaged on isolated branch
  `development/g3-scratch`, source `1e0428f`. Its 30 sanitizer CTests passed;
  subsequent test-only checks cover both gfx1013/gfx1030 at O1 and O2. O1 needs
  712 private bytes/177 spills, O2 212 bytes/52 spills. The native profile is fixed
  to O2 and checks 4096 distinct values, inputs, scratch activity and full guards.
  Imports, startup, FSELF and package extraction pass; 622 earlier files remain
  unchanged. See `evidence/offline-scratch-native-readiness.json`. No scratch
  execution has occurred; AGC ring state, wave offsets and visibility are unproven.

Experimental LDS implementation is frozen separately at `27165fd` on
`development/g3-lds`. All 41 integrated host CTests pass with ASan/UBSan,
including static/dynamic/mixed output and guard oracles. Four LLVM 18 objects
(O1/O2, gfx1013/gfx1030) and all three PS5 harness object compiles pass;
independent review checked uniform barriers and separate mixed static/dynamic
addresses. See `evidence/offline-lds-integrated.json` and the isolated branch's
`docs/LDS_LAUNCH_PLAN.md`. All three native packages now pass startup/import/FSELF/device-ISA/UFS
checks and complete extraction comparison; all 730 prior artifacts remain unchanged.
See `evidence/offline-lds-native-readiness.json`. They are cataloged as unrun
candidates. No LDS hardware execution or G3 qualification follows from these checks.

Contended atomic probes now compile at O1/O2 for both targets: 32/64-bit
fetch-add and CAS, plus runtime-address additions that avoid wave aggregation.
The CAS64 fixture uses a nonzero upper token word. Exact permutation/winner
oracles pass 13 synthetic mutation tests; actual objects fit the existing entry
structure. See `evidence/offline-atomic-contention-probe.json`. A six-dispatch
macro-only native harness is now frozen at `06956e4`, with 28 sanitizer CTests
and two SDK object compiles. Its actual-harness models cover reordered modular
tickets, full 64-bit CAS tokens, rounded guards, immutable offsets and quarantine
at every dispatch. See `evidence/offline-atomic-native-harness.json`. Native
linking/packaging, GPU execution and memory-scope qualification remain pending;
these results do not advertise atomic support.

Retained private-memory calls now have an isolated offline launch implementation
at `6f7d9a1` (`development/g3-flat-calls`). It supplies eight user words with the
raw flat base separate from descriptor flags, preserves existing routes, and uses
the actual cumulative 288-byte fixed frame requirement. All 47 sanitizer tests
passed; final compiler-audit operand/dataflow fixes and all four rebuilt fixtures
passed focused checks, and both changed core files compile with the PS5 SDK.
The strengthened source reloads a mutated private element after a retained call;
16,448 outputs and 96 boundary cases distinguish a lost store from register
forwarding. See `evidence/offline-private-call-launch.json`. No native flat-call
harness is packaged. A macro-only harness is now frozen at `716c8eb`, with 13
sanitizer tests covering four real compiler profiles and nine SDK object compiles.
Its models exercise the actual harness, parser and queue with synthetic GPU
callbacks; they do not execute ISA. It remains absent from package-selection
tools. See `evidence/offline-private-calls-native-harness.json`.
Private-aperture and flat-register support remain unproven;
prior graphics flat-scratch faults remain adverse evidence. A separate read-only
aperture probe compiles offline, but has not run on hardware. Continue compiler
specialization/backend research before considering the platform diagnostic.

Private-call compiler research now includes a bounded specialization pass:
64 new clones per invocation and eight descendants per original function, with
ABI-attribute and tail-call fixes, conservative fallback and deterministic budget
tests. A separate volatile fixture preserves nested calls, post-store reloads and
both valid global/private branches; all four compiler profiles and independent
CPU oracles pass. Its accesses remain flat and are not eligible for the current
specialization. See `evidence/offline-private-call-compiler-research.json`.
A default-off LLVM18 backend proof now compiles as a standalone pass object;
review corrected instruction-encoding pseudo handling, bundle rejection and the
pre-frame-lowering alignment check. Installed-compiler controls cover ordinary,
generic, volatile, reordered, late-spill and MIR-reload cases. See
`evidence/offline-private-backend-object.json`. Clean and patched LLVM builds now
complete. Actual enabled testing found two issues: conservative call-mask
clobbers kept flat initialization despite a proven graph, and the module barrier
disrupted IPRA's callee-first register-usage collection. Both failed revisions,
binaries and strict comparison evidence remain preserved. Revision 4 keeps real
register uses protected and rejects proof use unless IPRA is explicitly disabled.
Its selected paired baseline/default-off/enabled controls now pass: 24 matrix
cases per mode, negative/stress/reload/ineligible cases, ten valid MIR controls,
four real flat-register controls and 14 guard configurations. Eight independently
measured ordinary/volatile profiles require 304 nested private bytes (144 + 160 +
0); mixed volatile remains 144. All eight ordinary linked graphs preserve calls,
stores/reloads, ABI saves, descriptors and resources while removing only the four
flat-initializer instructions plus a separately proved frame-MOV scheduling change.
The extended audit checks 48 cases; 42 rejected cases remain byte-identical.
Six positive controls and six linked alias objects pass narrowly bounded scheduling
and actual-byte/call-target checks. Parent verification rehashed 1,409 artifact
references. See `evidence/offline-private-backend-ipra-off-controls.json`; earlier
failed research remains in `evidence/offline-private-backend-source-builds.json`.
Fresh portable reproduction now passes at `fa8788e`: 61 cases per mode, eight
ordinary linked graphs, 48 extension comparisons, 14 guard configurations, four
register controls and six linked alias objects. Independent verification rehashed
1,792 artifacts and bound the current binaries to retained source-build evidence.
The recipe is included under `tools/experimental/private-calls`; see
`evidence/offline-private-compiler-reproduction.json`. This used supplied
source-built binaries on the same WSL host, without rebuilding LLVM or claiming
independent-host reproduction.

The separate 304-byte `private_calls_buffer` harness and connected package tools
are frozen at `56dcbcf`. All 61 sanitizer CTests and eight checks on the four fresh
objects pass. The experimental profile lock pins the exact compiler receipt,
input IR and object bytes; equal private-byte metadata cannot substitute a generic,
volatile or proof-disabled shader. Its gfx1013 O2 image `c539b220...` passes
startup/import/FSELF/UFS2 checks and full extraction comparison; see
`evidence/offline-private-buffer-native-readiness.json`. It has not executed.
The original 288-byte native control is preserved; generic and volatile
private-call support, the production compiler and hardware capabilities remain
unchanged.

A separate read-only aperture diagnostic is frozen at `669d4fc` on
`development/g3-aperture`. It uses the existing six-user-SGPR launch path and
reports 64 raw base/limit observations while checking lane witnesses, within-wave
consistency, full rounded guards, ownership and closure. Distinct raw-field poison
catches omitted-store patterns without imposing an invented aperture range.
All 29 integrated sanitizer CTests passed, followed by four and five affected
checks after test and ISA-audit improvements; both target harnesses compile with
the PS5 SDK. See `evidence/offline-private-aperture-diagnostic.json`. The same
source now has a verified native package, image `7565cc2d...`, with startup,
116 import bindings, exact device ISA, FSELF and full UFS extraction checks.
All 1,012 preservation files remain unchanged; the catalog appends this as its
24th candidate. See `evidence/offline-private-aperture-native-readiness.json`.
No console execution has occurred. Even plausible observations
would not establish flat addressing or private-memory support.

Historical failures and superseded packages remain preserved. The first GPU
image `fb634e34...` faulted on a CPU import boundary; the isolated provider fix
then passed. See `docs/FIRST_GPU_FAULT.md`. Its revoked image and the earlier
unintended-LLD21 image must not be rerun. The package catalog retains their
identities, original PPSA99012 candidates and all later PPSA99014 packages.
No prior raw logs or artifacts were removed when this status was consolidated.

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

The configured TheRock `amd-llvm` build, stage installation and component `dist`
assembly have completed. Both component logs end in exit zero; original session
`40611` also returned zero. The four compiler/inspection tools and resolved
libraries match the prior comparison exactly. Installed executable code matches
the build tree, with recorded RPATH changes. The composed component `dist` runs
Clang/LLD/readobj/objdump/Flang; the bare stage needs its composed dependencies.
See `evidence/offline-therock-component-completion.json`. Full ROCm distribution,
artifact packaging and upstream tests remain separate work. The build and logs
under `/tmp/ps5hip-therock-comparison` must not be overwritten or restarted.

The next hardware candidate is the frozen `5b74ac8` combined arithmetic image,
as G2 item 5 explicitly requires. Its offline readiness and host evidence are
recorded, and the package catalog preserves all 19 earlier entries. Reviewed
controls are bound to committed catalog `2ad6d9b` and the exact candidate tools;
their first deployment invocation stopped at the busy-console idle guard.
The portable private-call compiler reproduction and exact 304-byte native package are
now ready offline. Check user-selected .40 once, with both coordination locks,
declared services, foreground idle and the frozen read-only PID204 status probe.
Require recovery evidence separately from foreground idle. If available, run the
unchanged arithmetic control before the experimental buffer-call image. Qualify
two separately reviewed clean arithmetic cycles for G2; a private-call pass is a
separate G3 observation. This
supersedes the planned vector-only cycle-two action;
preserve that control and its first-cycle evidence, but do not spend console time
on it as a G2-closing test. The installed vector image and log must still be
verified and preserved before any replacement. No console polling is needed while
offline work can advance. Full G2 remains open for the combined-candidate proof,
repeated lifecycle check and final acceptance audit.
Full/partial wave32 probes are now packaged at frozen source `5581dbd` with the
corrected AGC import provider. All 27 sanitizer tests, four O1/O2 target launch
checks, native startup/import/ISA and package extraction checks pass. Both measured
descriptors use six user SGPRs with kernargs in s[4:5]. The catalog now includes
both images; see `evidence/offline-wave32-readiness.json`. No wave hardware run
has occurred. Parser/launch changes since the proven G2 source are explicit;
preserve older receipts and never deploy the older incorrectly bound binaries.

Matching TheRock Clang24, full HIP7 headers and all seven real driver-selected
device libraries compile the combined vector fixture for Linux and PS5. The newer
bundler requires the explicit `x86_64-sie-ps5-unknown` spelling. Original and expanded
PS5 host-only IR match after normalizing only the triple and generated HIP IDs.
The original failure is preserved; no compiler/header/library patch was needed.
See `evidence/offline-therock-hip-host-abi.json`. This proves compilation of one
fixture, not native linking/loading or runtime compatibility. The current loader
still rejects the newer device target spelling; keep LLVM18 selected.
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
`evidence/offline-agc-provider-fix.json`. After an earlier busy-console no-run,
it passed once in `results/provider-fixed-store-run-20261011T203845Z` (PID 362).
See `evidence/native-provider-fixed-store.json`. Its image and successful log
were hash-verified and preserved before the vector candidate replaced it.
Do not reuse deployment/run scripts with stale image or log-absence assumptions.
The current host-test
cache selects that worktree's converted ELF for the artifact import check.

The following vector-add case is ready offline at the same source `b8bc2b0`:
`.local/g2-agc-provider-fix/dist/PPSA99014/gfx1013-vector_add-93752ddb9ec6`,
image SHA `fb2ecdc7feded72005ef5c228b02511071de707ef87ac47c1728a7e49f55fb57`.
Its full import map matches the corrected store, and device/helper/libc bytes
are identical; only the vector-selected canary object changes among the nine
application objects. FSELF, UFS2, extracted-file and local case/controller checks
pass. The corrected store prerequisite has now passed once, with exact output,
guards, clean teardown and healthy services, in
`results/provider-fixed-store-run-20261011T203845Z`. This single-vector package
remains an unrun control; the later two-vector image passed one clean cycle.
The combined arithmetic candidate now supersedes both as the next G2 case.
The store candidate remains unchanged. See `evidence/offline-provider-fixed-vector.json`.

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
scripts have now run: the remote image is the two-dispatch vector candidate,
its successful first-cycle log exists, and the successful store, failed GPU and CPU evidence
remain preserved. Do not reuse them as
fresh-case scripts or erase that evidence.

The CPU shell-managed lifecycle passed once on .30, FW 6.02, at source `3346166`,
stage `gfx1013-runtime-9ac5d150b5db`, image SHA
`d9a9e95cfc3e10b0fd8b547e54da3d2898291700537a27ecd9da3d63a6cc9cb0`.
Runtime checks, callbacks, destructor/fini observation, shell-request return zero,
self-stop before host cleanup, sandbox/image release and service health passed.
See `evidence/native-shell-exit.json`. Ordinary C exit still faults with SIGSYS;
do not repeat the failed diagnostics. Threaded/framework cleanup is unqualified.
The October 12 read-only check established that the failed .40 PID204 was absent;
the later deployment was blocked by another title. Recheck current recovery and
idle state before reuse; no new GPU qualification follows.

Offline work ahead of G2/G3/G4:

- The fixed `vector_add_twice` profile reuses one queue for two fresh guarded
  vector dispatches and requires ordered per-dispatch records plus a final summary.
  Repeated resource ownership/completion and second-round failure quarantine tests
  pass. Build/stage/case metadata enforce the profile; single-run contracts remain
  unchanged. All 27 host CTests pass; see `evidence/offline-vector-repeat-host.json`.
  Its isolated native source is `c37f0db`, parent `b8bc2b0`, on
  `development/g2-vector-repeat` in the existing `.local/g2-agc-provider-fix`
  worktree. Only harness/build/stage profile changes enter that source; no later
  hidden-geometry or wave implementation enters the experiment. Its package
  `gfx1013-vector_add_twice-70a3e57c6792` passed imports/startup/FSELF/ISA/UFS2,
  extracted-file and main catalog/controller verification; all 76 previous
  artifacts are unchanged. See `evidence/offline-vector-twice-package.json`.
  Image SHA `1487a21ba6bdd65f61a67ad6739e25f8cbc9306f2fcda5602403a0f51fec67f5`.
  Ten parser/resource/launch/memory/submission/AGC sanitizer tests also pass from
  this exact isolated source, separately from main's 27 tests. Source and all
  114 existing native build/stage files remain unchanged; see
  `evidence/offline-candidate-host-c37f0db.json`.
  Deployment preserved the successful store image/log. Cycle one then passed
  two native dispatches and clean teardown. Reviewed acceptance and separately
  tested cycle-two controls are preserved; the initial idle guard rejected cycle
  two before log preservation because another application was active. The
  combined arithmetic suite supersedes that vector-only cycle as the next case.
- Host CTests include registered hidden-geometry and native import coverage.
  Eight real ROCm 6.2.4 header-derived vector/dimension ELFs pass launch preparation,
  twelve inspector checks pass, and PS5 native libraries cross-build. Explicit and
  thirteen hidden arguments are separated; checked uniform 64x1x1 geometry is
  generated in zeroed 256-byte implicit storage. Unsupported layouts fail. See
  `evidence/offline-hidden-geometry.json`. Existing frozen titles do not exercise it.
- Twelve real combined HIP header fixtures compile for Linux/PS5 and both targets.
  Authentic matching headers, host registration/fatbinary layouts, device ABI flags
  and a source-derived workitem-only library are recorded in
  `docs/HIP_HOST_ABI_SPIKE.md`. Full OCKL failed on LLVM 18; the later matching
  TheRock/HIP 7 spike compiled its fixture with complete matching device libraries. No HIP
  runtime is linked or run; the newer code-object format remains rejected.
- Wave32 full/partial-lane probes compile at O1/O2 for both targets. Both guarded
  gfx1013 native profiles link with checked imports/startup/FSELF; gfx1030 remains
  compile-only. Both gfx1013 profiles are now packaged at `5581dbd`, but neither
  has executed. See `evidence/offline-wave32-readiness.json`.
- Generic volatile private lowering remains a G3 issue. An isolated IR contrast
  preserves volatile operations and emits buffer instructions; reserved flat-init
  registers are overwritten before use, with no hardware flat-scratch setup.
  The isolated `1e0428f` spill experiment implements buffer scratch backing,
  descriptors, offsets, state and lifetime for six-user-SGPR entries only; it
  remains unrun. The eight-user-SGPR indexed-private layout and generic lowering
  remain unsupported. See `docs/VOLATILE_PRIVATE_LOWERING.md`; no generic-pointer
  workaround is accepted.

Source notices are inventoried, including the exact ps5log GPL-3.0-or-later notice.
Linked archive-member accounting and binary redistribution remain release work.
The first kernel's instruction/entry contract is in `docs/INITIAL_CAPABILITY_MATRIX.md`.
G2 still needs the combined image's exact output, guards and two clean cycles;
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
`evidence/offline-native-packages.json` preserves the five original `PPSA99012`
packages and later `PPSA99014` manifests, hashes, source revisions and revocations,
including the provider-fixed store and vector candidates.
`evidence/offline-controllers.json` records controller hashes and host validation.

## Milestones

- 2026-10-11 | planning | twelve measurable goals and pinned research references recorded | next: G1 offline inventory
- 2026-10-11 | G1 progress | arithmetic and resource probes compiled for two targets; host parser checks pass; private-array flat-scratch incompatibility identified | not a completed gate
- 2026-10-11 | G1 progress | unmodified compiler emits buffer scratch for explicit-private accesses and 52 real VGPR spills; generic private pointers remain unresolved | not hardware-qualified
- 2026-10-11 | offline milestone | 0c2e8c1 | PPSA99012 | packaged, no-run: five verified images and fourteen host tests | evidence/offline-native-packages.json | next: console controller

- 2026-10-11 | CPU lifecycle milestone | 3346166 | PPSA99014 | one shell-managed startup/finalization/self-close passed on FW 6.02; 22 host cases pass | evidence/native-shell-exit.json | no G1-G12 gate passed

- 2026-10-11 | G1 passed offline | LLVM18 + gfx1013 selected; isolated reproduction, TheRock comparison, 25 host tests | evidence/offline-g1-acceptance.json | next: bounded G2 store run
- 2026-10-11 | G2 failed | 15bfc35 | FW6.02 PPSA99014 | CPU SIGSEGV/RIP0 after SUBMIT; title released, services reachable | evidence/native-indexed-store-first.json | next: offline fault mapping
- 2026-10-11 | G2 progress | b8bc2b0 | FW6.02 PPSA99014 | one indexed-store GPU dispatch and clean teardown passed | evidence/native-provider-fixed-store.json | next: repeated vector dispatch
- 2026-10-11 | G2 progress | c37f0db | FW6.02 | two vector dispatches and one clean cycle passed | evidence/native-vector-twice-cycle1.json | next: idle-gated second cycle
- 2026-10-11 | G3 offline | 1e0428f | gfx1013 | forced-spill image packaged; no hardware run | evidence/offline-scratch-native-readiness.json | next: combined G2 acceptance candidate
- 2026-10-11 | G2 offline | 5b74ac8 | gfx1013 | combined arithmetic image packaged | evidence/offline-arithmetic-suite-native-readiness.json | next: two idle-gated cycles
- 2026-10-11 | G3 offline | 716c8eb / 669d4fc | retained-private-call harness modeled; read-only aperture package verified | evidence/offline-private-calls-native-harness.json; evidence/offline-private-aperture-native-readiness.json | no new hardware qualification
- 2026-10-11 | G3 offline | 06956e4 | six-dispatch atomic harness modeled; both SDK objects compiled | evidence/offline-atomic-native-harness.json | no package or hardware qualification
- 2026-10-11 | compiler research | source-built LLVM comparisons exposed call-mask and IPRA ordering issues; guarded revision 4 compiled | evidence/offline-private-backend-source-builds.json | full paired acceptance pending
- 2026-10-11 | compiler milestone | guarded revision 4 passes selected paired IPRA-off controls and eight linked ordinary graphs; generic/volatile fallback retained | evidence/offline-private-backend-ipra-off-controls.json | next: fresh portable reproduction and isolated304-byte native harness
- 2026-10-12 | G3 offline | fa8788e / 56dcbcf | portable recipe and304 image verified; .40 deployment no-run while another title active | evidence/offline-private-buffer-native-readiness.json; evidence/native-private-buffer-console-no-run.json | next: arithmetic control then private candidate when idle
