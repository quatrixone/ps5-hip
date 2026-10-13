# Current state

Updated: 2026-10-13, America/New_York.

## Actual status

- **Visibility computation and delayed cleanup independently audited; original cycle inconclusive:** the recovered
  result reports six dispatches, three rounds and 771 exact outputs over persistent
  buffers, intact guards, closed allocations and complete finalization. Two read-only
  fetches matched. The original controller rejected interleaved spawn records and
  exhausted its deadline before result retrieval or a post snapshot. Both operations
  ended with reachable services and released locks; no GPU rerun occurred. Recovery
  review passed. A separate delayed snapshot matched the original visible mount
  baseline with unchanged result and installed bytes; independent audit passed.
  The post snapshot was 864.105950 seconds after launch, with unlocked/klog gaps;
  it does not repair the original 20-second cycle. See
  `evidence/native-visibility-computation.json` and
  `evidence/native-visibility-execution-incomplete.json`.
- Refreshed full/partial wave32 candidates are frozen at `c3972cb`, using current
  allocation/submission cleanup. Device code and wave harness objects match the
  older candidates; the new images need their own hardware qualification. All 38
  configured sanitizer tests, eight actual-object launch/packet checks and both
  native package validations passed. Parent artifact review rehashed sources,
  logs, staged payloads and images. See `evidence/offline-wave32-current-readiness.json`.
- **Mixed LDS fixture passed on .40 and independent review:** 257 exact
  sum-XOR-tag outputs from 16,448 unchanged inputs, static/dynamic/allocated LDS
  256/1280/2048 bytes, stride five. Guards, allocation closure and finalization
  passed; PID159/application `0xe018` stopped naturally. Visible mount identities
  matched, services stayed healthy and locks released. See
  `evidence/native-lds-mixed.json`. Independent audit rehashed 25 raw files and
  165 bindings and reproduced acceptance. The dynamic image and result were preserved.
- **Persistent-buffer producer/consumer candidate ready offline:** frozen commit
  `a049853` on `development/g3-dependent-visibility` reuses three guarded buffers
  across six dispatches and three changed-input rounds. All 54 sanitizer tests,
  native link/import checks and package extraction passed; independent review is
  clear. See `evidence/offline-dependent-visibility-readiness.json`. The subsequent
  hardware observation is above; this fixture includes CPU cache maintenance.
- **Dynamic LDS fixture passed on .40 and independent review:** 257 exact sums
  from 16,448 unchanged inputs, static/dynamic/allocated LDS 0/1280/2048 bytes,
  stride five, intact guards and closed allocations. PID150/application `0xc018`
  finalized and stopped naturally inside the original bounded cycle. Visible mount
  identities matched, services stayed healthy and both locks released. Independent
  audit verified 25 raw files and 160 bindings and reproduced acceptance. See
  `evidence/native-lds-dynamic.json`. Mixed deployment preserved its image and log.
- **Static LDS fixture passed on .40:** 257 exact sums from 16,448 unchanged
  inputs, intact guards, allocation closure, full finalization and natural PID141 /
  application `0xa018` stop. Visible mount identities matched; services remained
  healthy and both locks released. Parent saved-evidence audit reproduced acceptance
  and rehashed 25 raw files. Post-run independent review was interrupted by a content
  filter; preflight and deployment reviews passed. See `evidence/native-lds-static.json`.
  The later dynamic deployment preserved its image and 616-byte result. Full G3 is open.
- LDS offline revalidation passed: ten host tests, four byte-identical compiler
  profiles and three byte-identical oracle files. Static LDS checks 257 sums from
  16,448 inputs; see `docs/LDS_EXPERIMENT.md` and
  `evidence/offline-lds-reproduction-v2.json`. Those checks preceded the static LDS
  observations above; those native results cover only their exact frozen profiles.
- **Compiler-spill fixture passed on .40:** 4,096 exact outputs, modified scratch,
  unchanged guards, closed allocations, complete finalization and natural PID132 /
  application `0x8018` stop. The bounded cycle has matching visible mount snapshots,
  healthy services and released locks. Parent offline audit rehashed all 25 raw
  files and reproduced acceptance. Independent post-run review subsequently passed
  after the earlier subagent content-filter failure; all 115 bindings and the
  original bounded lifecycle were verified. See `evidence/native-register-spill.json` and
  `evidence/native-spill-deployment.json`. This covers one O2 two-wave spill shape,
  not full G3. Next: static LDS, then dynamic/mixed LDS. Preserve installed spill
  image/log and earlier private304 backups. Private304 milestone pushed as `e431228`.
- **Private304 computation passed on .40:** all 16,448 exact outputs, modified
  scratch, unchanged guards, closed allocations and full finalization. A separate
  independently audited completion capture recovered the unchanged application log
  without rerunning the GPU and found eventual visible mount-baseline equivalence.
  Its post snapshot was 452.68 seconds after launch, with an unlocked/klog gap;
  the original bounded cycle remains inconclusive. This qualifies only the exact
  O2 buffer-private nested-call fixture, not full G3. See
  `evidence/native-private304-computation.json`. Next: frozen compiler-spill case,
  then static/dynamic/mixed LDS; use the reviewed strict-idle wait correction.
- The first private304 title launched as PID123/application `0x6018` and
  self-stopped with complete observed OS teardown. Its host controller then
  refused cleanup during a short shell-focus lag; no close was sent, and the
  application result/post-run snapshot were not fetched. The failure postcheck
  observed idle; services were healthy and locks released. This attempt remains
  inconclusive as an original host-control cycle. The separate completion evidence
  above establishes computation while preserving this record and disclosing the
  delayed post-observation. See `evidence/native-private304-execution-incomplete.json`.
- The private304 mounted image is now independently verified on .40, with exact
  package, signed executable, device object, metadata and mount-link bytes.
  The read-only recovery used acknowledged connection-local raw SELF transfer;
  preserved arithmetic artifacts matched, idle/health passed and locks released.
  See `evidence/native-private304-deployment-verified.json`; this verification
  preceded and was bound into the first private-memory run above.
- The private304 image was promoted on idle .40 after preserving the arithmetic
  image/log and observing normal old-image unmount. Fresh UFS/nullfs registration
  followed, but the first mounted executable read exceeded its signed-SELF size
  bound. That original deployment verification was incomplete; no application was launched.
  Services remained reachable and locks were released; final idle could not be
  reread after the failed FTP transfer. A connection-local SELF-to-ELF conversion
  is the documented hypothesis; the later read-only raw-byte verification above
  established mounted identity without reconstructing the discarded response.
  See `evidence/native-private304-deployment-incomplete.json`.
- **G2 passed for the declared arithmetic fixture.** Two independently reviewed
  cycles of the same frozen image on .40 checked 146 exact values each, unchanged
  guards, allocation closure and full finalization. PID105/application `0x2018`
  and PID114/application `0x4018` self-stopped before host close; ordered OS
  teardown, visible pre/post mount equivalence, service health and owned lock
  release passed. The six-item requirement map is in
  `evidence/native-g2-acceptance.json`. Scope is gfx1013 wave32, the named
  relocation-free store/vector kernels, 64x1x1 threads and two X groups over
  73 elements. The subsequent private-memory observation is recorded above.
- The new prospective arithmetic cycle on .40 was **inconclusive** because of a
  host-controller defect: it treated successful launch return `0x18` (the new
  application ID) as failure and closed PID99 before natural self-stop. No fresh
  arithmetic log or post-run mount snapshot was obtained. The installed image
  and positive pre-run baseline verified; the prior log was preserved. Exact-title
  host-close teardown completed, idle returned, all services stayed reachable and
  both locks were released. That consumed controller is preserved; the separately
  corrected controller and direct response regressions enabled the two passes above.
  See `evidence/native-arithmetic-prospective-cycle1.json`; this is not evidence
  of a GPU failure and does not change the earlier partial-pass.
- G1 passed offline; G2 passed its declared native fixture. G3-G12 are not complete. There is
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
  used two images, so another vector-only cycle could not close G2. The combined
  arithmetic suite using the unchanged proven kernels/core subsequently passed
  two clean cycles of the same image, as recorded above. The earlier vector cycle-two control was
  blocked before launch or log preservation by unrelated
  PPSA99930 PID394. Both locks were released. See
  `evidence/native-vector-twice-cycle2-no-run.json`. The successful vector image
  and original first-cycle log were preserved before subsequent replacements.
- Identity is **PPSA99014**; PPSA99012 was occupied. Use the Eden client and the
  bundled exact-title Lapy helper with configurable ordinary results storage.
  The current staging choice is `/data/ps5-hip/results`, not a compiled path.
- The combined `arithmetic_suite` is frozen and packaged at isolated source
  `5b74ac8`; image `28262f78...` runs store then vector on the same queue with
  fresh allocations and 146 exact values. Only the native harness object differs
  from `c37f0db`; device/core/helper/libc/import identities match. Host routing,
  corruption and second-dispatch quarantine checks pass, as do native package
  checks. See `evidence/offline-arithmetic-suite-native-readiness.json` and
  `docs/G2_ARITHMETIC_ACCEPTANCE.md`. Its first .40 run is recorded below.
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
- At 13:47 UTC the unchanged combined arithmetic image ran on idle .40, firmware
  12.70: both dispatches checked 146 exact values, full guards, allocation closure
  and all finalization checkpoints. PID134 self-stopped before host cleanup;
  process exit, title sandbox ProcessTerm and OS EndAppMount completion were
  observed, services stayed reachable and both locks were released. Independent
  review classified this **partial-pass**: ShadowMount v1.6beta16 retained its
  registered UFS image and emitted no explicit layer-release records. This is
  not proof of a leak, but does not satisfy the declared release prerequisite.
  That evidence gap held further arithmetic/private304 launches until the explicit
  prospective lifecycle contract and two new accepted cycles resolved the G2
  prerequisite. The original partial-pass remains unchanged. See
  `evidence/native-arithmetic-fw1270-cycle1.json`.
- The current host suite passes 46 sanitizer CTests with real HIP bundle
  fixtures, registration/capture models, OCML objects, directed MPFR certificates,
  owned native data and telemetry v2 enabled. The earlier bundle/inspector
  changes also passed all three affected cases. The exact frozen vector source passes
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
at every dispatch. See `evidence/offline-atomic-native-harness.json`. The connected
suite is now frozen at `2b863a7` and packaged: 29 sanitizer tests, four real
compiler profiles, startup/import/FSELF/UFS checks and extracted-object host
checks pass. Independent review verified all 5,771 protected files unchanged.
See `evidence/offline-atomic-native-readiness.json`. GPU execution and
memory-scope qualification remain pending; these results do not advertise atomic
support. The package catalog now has 26 entries. Earlier console controls pin
the 25-entry catalog and intentionally reject this change. Version2 controls
were independently reviewed, verified against both frozen profiles and used
for the .40 arithmetic observation; original scripts and receipts are unchanged.

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

The bounded HIP offload-bundle reader and inspector are frozen at `89c0644`.
They select exact `hipv4` gfx1013/gfx1030 payloads, check the complete directory
and copy the selected linked V5 ELF into owned storage before inspection.
Twelve real LLVM18/ROCm6.2.4 Linux/PS5 host objects select and materialize unchanged;
truncation, malformed directories, ambiguous targets and 2,000 mutation trials
pass sanitizer checks. The reader cross-compiles with the pinned PS5 SDK.
See `evidence/offline-fat-binary-reader.json` and
`docs/HIP_FAT_BINARY_BOUNDARY.md`. No HIP registration exports, normal host launch
or new native execution is implemented by this unit.

The single-TU registration control is frozen at `08a5f52` and integrated into the
development branch. Actual ROCm6.2.4 compiler constructors register an owned
module and host-token binding on Linux, with normal and partial-failure teardown
checked through genuine compiler callbacks. Launch returns `hipErrorNotSupported`
and submits no work. PS5 compilation, raw PIE linking and conversion preserve the
bounded wrapper/bundle, relocations, constructor entry and converted RELRO.
Independent review verified 68 outputs, 59 imports and 242 untouched baseline
files; all 31 integrated sanitizer tests pass. See
`evidence/offline-host-registration-control.json` and
`docs/HIP_HOST_REGISTRATION_CONTROL.md`. The measured SDK lacks the destructor-bearing
TLS import; this serialized control does not qualify threads, native exit, GPU
launch or G4 acceptance.

Historical failures and superseded packages remain preserved. The first GPU
image `fb634e34...` faulted on a CPU import boundary; the isolated provider fix
then passed. See `docs/FIRST_GPU_FAULT.md`. Its revoked image and the earlier
unintended-LLD21 image must not be rerun. The package catalog retains their
identities, original PPSA99012 candidates and all later PPSA99014 packages.
No prior raw logs or artifacts were removed when this status was consolidated.

The G4 architecture comparison now selects one small native HIP compatibility
runtime for the first serialized examples. All five unchanged pinned CLR units
compile on Linux; the PS5 SDK compiles NDRange, with concrete OS/semaphore branch
failures elsewhere. Parent verification checked71 raw artifacts and24 upstream
source files. Reassess CLR before substantial asynchronous scheduling or deferred
memory management; no production CLR/HSA implementation is being started. See
`docs/RUNTIME_ARCHITECTURE.md` and `evidence/offline-runtime-architecture.json`.
No runtime link, native execution or additional gate acceptance follows.

Compiler argument capture is frozen at `0edd623` and integrated: actual compiler
stack values are copied into owned scalar/pointer arguments, interior pointer
extents are resolved through the permitted live-data inventory, and the existing
planner packs all288 kernarg bytes and56 packet words. The genuine stub still
returns `hipErrorNotSupported`, with synthetic addresses and no GPU calls.
All32 isolated and34 integrated sanitizer tests pass; parent verification checked
147 probe artifacts and3,728 untouched baseline files. See
`evidence/offline-host-launch-capture.json` and `docs/HIP_HOST_LAUNCH_CAPTURE.md`.

Owned native data and the separate full-header HIP memory control are frozen at
`b4ab3cf` and integrated. Logical extents exclude rounded backing padding; checked
interior pointers, H2D/D2H copies and exact-base free use actual NativeMemory.
Clean setup failures preserve existing buffers; retained cleanup poisons the
registry, including supported zero-size ABI calls. GPU-owned free/copy reject,
completed queue ownership permits free, and closed holds cannot be resubmitted.
All 37 isolated sanitizer tests pass; the ordinary typed HIP client also passes
ASan/UBSan and PS5 compile/link/conversion checks with five direct-memory imports
and no HIP/AGC imports. Independent review verified 57 artifacts; 3102 protected
files stayed unchanged. Registration/capture controls remain separate and launch
still returns unsupported. See `evidence/offline-native-data-memory.json` and
`docs/NATIVE_DATA_REGISTRY.md`. Native execution/cache visibility remain unqualified.

The exact public ShadowMount1.6beta16 source is now verified at `8566c02`:
normal exit deliberately retains valid registration images and title nullfs
mappings. The earlier all-layer-disappearance expectation conflicts with that
policy. `docs/G2_ARITHMETIC_ACCEPTANCE.md` explicitly corrects the prospective
contract to observed process/OS teardown plus unchanged registration baseline;
the historical partial-pass remains unchanged. Snapshot visibility and deployed
loader identity remain unqualified. See `docs/SHADOWMOUNT_16_LIFECYCLE.md` and
`evidence/offline-shadowmount16-lifecycle.json`. Review fresh before/after controls
before another arithmetic/private launch; do not force unmounts or change settings.

The pinned OCML-only recipe is frozen at `3730f16` and integrated. It builds
authentic OCML plus conservative configuration constants without the blocked
full OCKL build. Four gfx1013/gfx1030 O1/O2 objects exactly reproduce the earlier
independent outputs; prelink IR proves direct sin/log/tanh OCML calls. Two focused
tests pass, with actual artifact mutations. No GPU, numerical-accuracy, complete
device-library or hermetic-toolchain qualification follows. See
`docs/OCML_MATH_PROBE.md` and `evidence/offline-ocml-math-probe.json`.

The guarded math candidate is now frozen at `ef9f296`, packaged as image
`55ce66ac...` and appended as unrun catalog entry 27. Its 257 immutable inputs yield
1542 results, with full rounded guards and 378 inactive output words checked.
An independent MPFR 4.2.2 oracle fixes exact FMA/sqrt results, declared 4/3/3/5
engineering ULP envelopes for sin/exp/log/tanh, special values and subnormal
requirements before any hardware result. Five numerical tests, 45 isolated and 46
integrated sanitizer CTests pass, including all 280 kernarg bytes of four actual
objects. Native startup, 116 imports, FSELF, UFS and retained extraction checks pass;
5783 prior files are unchanged. Loader/launch objects postdate the proven G2
candidate, so this is not a harness-only difference. See
`evidence/offline-native-math-readiness.json` and `docs/NATIVE_MATH_PROFILE.md`.
No math GPU execution, all-domain accuracy or G3 completion follows.

The read-only diagnostic is frozen at `05b1723`; two focused sanitizer checks,
SDK compilation and a16-import audit pass. Its one idle .40 observation at14:33UTC
confirmed historical PID134 ESRCH and empty application-ID queries, but both
mount collectors returned `count_bound` before any rows. Initialized count0 is
not an empty mount table. The framed transcript completed, services stayed
reachable and locks were released; the kernel also emitted an FD-budget warning
whose cause is unestablished. This is inconclusive and does not supply a usable
registration baseline or permit another GPU launch. Preserve the one-use marker.
See `evidence/offline-teardown-snapshot.json` and
`evidence/native-teardown-snapshot-fw1270.json`. This motivated the separate
raw-return telemetry candidate; the original marker remains consumed.

Telemetry v2 is frozen at `0bf7fb9` with the same provider, record limits and
query sequence. It stores exact signed/hex returns and errno for the initial
query and three bounded fill slots. Source reconstruction recovers v1 after
removing telemetry, and native imports/call sites match. Two focused tests and
the final 36-test integrated sanitizer suite pass. See
`evidence/offline-teardown-snapshot-v2.json`. Independent review cleared the
fresh one-use controller. Its 14:52 UTC .40 preflight returned no-run because
unrelated PPSA99008 PID177 was active. No diagnostic send, title action or remote
mutation followed; services were reachable and both locks released. The v2
marker then remained absent and the v1 marker consumed. See
`evidence/native-teardown-v2-no-run.json`; do not poll for availability.

At 15:36 UTC, after the memory/math offline milestone, idle .40 executed the frozen
telemetry v2 once. Both initial queries returned 1110/errno 0, exceeding its 1024
record ceiling; no allocation or fill query occurred. PID134 was ESRCH and both
app-ID queries absent before/after; declared services stayed reachable and both
locks released. The FD-budget warning recurred without an established cause.
This remains inconclusive with release unknown; the v2 marker is now consumed.
See `evidence/native-teardown-v2-fw1270.json`. Reassess the collector offline
using the measured 1110 return and SDK contract; no automatic third observation
or GPU launch is authorized by these results.

Offline reassessment proposes a separate 2048-record collector ceiling using
the measured return and pinned SDK contract, with unchanged provider/layout/calls
and bounded memory/output. FreeBSD reference source confirms count/byte units but
also skipped busy/hidden mounts and possible identity redaction; no firmware ABI
or global completeness follows. Collector v3 is now frozen at `d6587ae`, built
against the pinned SDK, and passes two focused sanitizer/import checks. Parent
review verified the small production diff, source/audit hashes and native ELF.
See `evidence/offline-teardown-count1110-design.json` and
`evidence/offline-teardown-v3-readiness.json`. A fresh sender passed 17 mocked
checks and independent review. Its first .40 preflight at 2026-10-13 01:01 UTC
was no-run: FTP2121 and klog3232 refused connections, while9021 was reachable.
No payload was sent; both locks were released and the v3 marker remains absent.
See `evidence/native-teardown-v3-no-run.json`. Release remains unknown; require
available services and fresh idle/firmware checks before any observation.
A new user-authorized service preflight at 01:37 UTC had the same no-run result:
no payload, both locks released, and the v3 marker still absent. The user has
been asked to start FTP/klog; see `evidence/native-teardown-v3-service-recheck-no-run.json`.
After the user started services, all three passed at 01:40 UTC. The fresh log
reported FW12.70 and initial home focus7, but the frozen parser requires a later
focus transition, so it stopped before payload dispatch. This is a log-format
gap, not evidence of an active game. See `evidence/native-teardown-v3-initial-focus-no-run.json`.
At 01:47 UTC a separately reviewed host parser accepted the initial focus format,
and the unchanged v3 payload completed once. Both complete 1110-record snapshots
contained the same registered UFS image and title nullfs mapping. This establishes
a usable visible registration baseline under the prospective G2 criterion, not
global namespace completeness or resource release. The console remained idle,
all services stayed reachable and both locks were released. The shared v3 marker
is now consumed. See `evidence/native-teardown-v3-fw1270.json`; prepare a new
bounded arithmetic cycle with fresh pre/post evidence, not another standalone snapshot.

## Next action

Finish prospective interleaved-log parser qualification, then qualify the refreshed
wave32 candidates. Preserve the successful visibility image/result and the original
inconclusive receipt. The additive 2D/3D
geometry implementation is being tested in `.local/g3-geometry`. Atomic, numerical and broader
visibility qualification remain open; do not treat isolated fixture passes as G3.

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

The exact older-loader source explains persistent registration, and the future
lifecycle criterion is explicitly corrected in `docs/G2_ARITHMETIC_ACCEPTANCE.md`.
The original first combined arithmetic cycle remains partial-pass. The read-only
mount observation hit its count bound. Raw-query instrumentation v2 was later
executed once while idle and measured 1110 twice, still above the 1024 ceiling.
All three diagnostic one-use markers are consumed. The bounded v3 collector
now has two complete stable observations with positive registration anchors.
The prospective baseline prerequisite is established; the next reviewed arithmetic
cycle must still obtain its own fresh same-context pre/post snapshots. Do not infer an empty table from
initialized fields, a leak from registration persistence, or deployed identity
from the banner. Preserve the installed arithmetic image, successful result and
earlier unique CPU backup. No setting change or automatic repeat is authorized
by these inconclusive observations.

The portable compiler reproduction and exact304 native package remain ready
offline. Once the control's lifecycle prerequisite is established, test private304
once with fresh idle/recovery checks under both locks. Two separately accepted
clean arithmetic cycles and final mapping to the written requirements are still
needed for G2; private-call execution is a separate G3 observation. Existing version2
GPU controls bind the previous 26-entry catalog; the current 28-entry catalog needs
fresh reviewed controls before deployment. Never reuse consumed receipts, reset their
markers or apply old log-absence assumptions. The connected G4 vector control is
now frozen at `5509e31`: genuine compiler constructor/stub, owned native data/code/
kernargs, lazy AGC, blocking queue, synchronization and genuine unregister.
All 26 host scenarios and 48 integrated CTests pass; the PS5 executable passes
link/conversion/layout/import checks. Explicit cleanup failures prevent reuse,
while normal in-flight NotReady errors no longer poison the runtime. See
`evidence/offline-hip-native-vector.json`. This is modeled transport, not GPU
execution. The follow-up title is now frozen at `038fdb6`, integrated as `c62fbe2`,
and packaged with Eden and shell-managed lifecycle. It passed22 genuine-constructor
lifecycle cases,26 standalone cases,50 integrated sanitizer CTests and11 recipe
tests including actual-artifact mutations. Final checks require actual unregister,
complete State destruction, closed allocations and persisted logs before shell exit.
The UFS image passed filesystem and full extraction checks;38 source and140 raw
artifact hashes were independently verified. The signer omits one final unmapped
24-byte SIE note; every other reconstructed ELF byte matches, and the original
whole-file-equality failure remains preserved. See `evidence/offline-hip-native-vector-title.json`.
No native execution or G4 completion is claimed. Qualify the arithmetic lifecycle
control before a new reviewed HIP title launch. Separate earlier controls remain preserved.
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

- 2026-10-13 | G3 visibility | a049853 | .40:771 exact outputs; recovered result and delayed cleanup audited, original cycle inconclusive | evidence/native-visibility-computation.json | wave32 next
- 2026-10-13 | G3 wave32 offline | c3972cb | refreshed cleanup,38 sanitizer tests,8 object checks,both packages verified | evidence/offline-wave32-current-readiness.json | hardware pending

- 2026-10-13 | G3 mixed LDS | 27165fd | .40:257 exact outputs, intact guards, bounded natural teardown; independently audited | evidence/native-lds-mixed.json | remaining G3 features
- 2026-10-13 | G3 visibility offline | a049853 | 54 tests, native package and independent review passed | evidence/offline-dependent-visibility-readiness.json | hardware validation pending

- 2026-10-13 | G3 dynamic LDS | 27165fd | .40:257 exact sums, intact guards, bounded natural teardown; independently audited | evidence/native-lds-dynamic.json | mixed LDS next; G3 open

- 2026-10-12 | G3 static LDS fixture | 27165fd | .40:257 exact sums,inputs/guards intact,natural teardown; parent-audited | evidence/native-lds-static.json | dynamic/mixed LDS next; G3 open

- 2026-10-12 | G3 exact spill fixture | 1e0428f | .40:4,096 exact outputs,guards intact,full bounded teardown; parent-audited | evidence/native-register-spill.json | next static LDS; G3 open

- 2026-10-12 | G3 exact private-call observation | 56dcbcf | .40:16,448 exact outputs,304 private bytes/lane,guards intact; delayed composed capture audited | evidence/native-private304-computation.json | full G3 open; next forced spills

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
- 2026-10-12 | G4 preparation | 89c0644 | bounded fat-binary reader and inspector consume12 real HIP objects; sanitizer and SDK checks pass | evidence/offline-fat-binary-reader.json | next: bounded constructor registration; no runtime or hardware qualification
- 2026-10-12 | G3 offline | 2b863a7 | six atomic kernels packaged and independently verified | evidence/offline-atomic-native-readiness.json | GPU qualification pending
- 2026-10-12 | G2 partial-pass | 5b74ac8 | FW12.70 arithmetic146 passed; layer release unproven | evidence/native-arithmetic-fw1270-cycle1.json | hold launches
- 2026-10-12 | G4 offline | 08a5f52 | genuine compiler registration controls, PS5 link checks and31 sanitizer tests pass | evidence/offline-host-registration-control.json | native launch unsupported
- 2026-10-12 | G4 architecture | pinned CLR/HIP6.2.4 comparison selects one small native runtime with explicit reassessment threshold | evidence/offline-runtime-architecture.json | implementation and hardware qualification pending
- 2026-10-12 | G4 offline | 0edd623 | compiler arguments copied, packed and encoded;34 integrated sanitizer tests pass | evidence/offline-host-launch-capture.json | no GPU submission
- 2026-10-12 | G3 offline | 3730f16 | four OCML math objects reproduce exactly;36 integrated sanitizer tests pass | evidence/offline-ocml-math-probe.json | numerical/GPU checks pending
- 2026-10-12 | diagnostic | 0bf7fb9 | telemetry v2 built/reviewed; .40 busy at14:52UTC, no payload sent | evidence/native-teardown-v2-no-run.json | await useful idle milestone
- 2026-10-12 | read-only .40 | 05b1723 | historical PID134 absent;mount count bound prevents baseline;services healthy | evidence/native-teardown-snapshot-fw1270.json | inspect raw query offline
- 2026-10-12 | G4 offline | b4ab3cf | owned native data and full-header memory ABI pass;37 isolated tests | evidence/offline-native-data-memory.json | connect genuine vector launch
- 2026-10-12 | G3 offline | ef9f296 | math image/oracle verified;46 integrated tests | evidence/offline-native-math-readiness.json | GPU qualification pending
- 2026-10-12 | diagnostic | 0bf7fb9 | idle .40 returned1110 twice;1024 cap prevents baseline | evidence/native-teardown-v2-fw1270.json | offline reassessment
- 2026-10-12 | G4 offline | 5509e31 | genuine HIP vector path:26 host cases,48 integrated checks,PS5 link | evidence/offline-hip-native-vector.json | package lifecycle next
- 2026-10-12 | diagnostic no-run | d6587ae | .40 FTP/klog refused;no payload;locks released | evidence/native-teardown-v3-no-run.json | services then fresh idle checks
- 2026-10-12 | G4 offline | 038fdb6 | HIP title packaged;50 CTests;22 lifecycle cases | evidence/offline-hip-native-vector-title.json | native qualification pending
- 2026-10-12 | read-only .40 | d6587ae | two complete1110 tables;visible registration baseline established | evidence/native-teardown-v3-fw1270.json | new prospective arithmetic cycle
- 2026-10-12 | G2 inconclusive | 5b74ac8 | .40 host launch-parser bug caused early close;services healthy | evidence/native-arithmetic-prospective-cycle1.json | fix host parser offline
- 2026-10-12 | G2 cycle pass | 5b74ac8 | .40 exact146,self-stop,visible mount equivalence | evidence/native-arithmetic-prospective-cycle1-pass.json | second same-image cycle
- 2026-10-12 | G2 passed | 5b74ac8 | .40 two clean same-image cycles,292 exact values | evidence/native-g2-acceptance.json | G3 private304 validation
