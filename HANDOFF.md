# ps5-hip agent handoff

Updated: 2026-10-15, America/New_York. This is the resume point for the private
`ps5-hip` repository. Read `AGENTS.md`, `STATE.md`, and the active goal in
`docs/IMPLEMENTATION_PLAN.md` before working. The mission is a native PS5 HIP /
ROCm compute stack; a bounded result does not close a broader acceptance gate.
For a new computer with GitHub source only, read `docs/SOURCE_ONLY_REBUILD.md`:
`make` builds a new public HIP package from source on `main`. The historical
installed image and D evidence cannot be recreated by Make.

The root `Makefile` now fetches locked public inputs and builds the SDK,
zlib, UFS2Tool, Lapy helper, controllers, host suite, native core, and a
74-object public HIP bootstrap package. The
ProsperoEden and protocol locks refer to their public `main` snapshots; old
local-only commits are no longer build prerequisites. The protocol launch
controller now returns a status without printing a result, so the historical
one-use E control remains inert pending revised lifecycle checks and review.

## Frozen launch candidate

`/home/denis/ps5-hip-module-package/results/source-launch-console-v1` is frozen
at `6f5167e` and passed independent offline identity review: all 29 bound files,
seven packaged files and 86 build inputs match. See
`evidence/offline-frozen-public-launch.json`. Its `offline-review.json` explicitly
sets `approved:false`; operational `review.json` is absent. This is not deployment
approval. Preserve this candidate and its source/control identities while further
implementation proceeds; later source changes require a distinct candidate.

Recovery status from the owner remains pending. After recovery, obtain complete
read-only installed-image and preserved-memory-log hashes, verify idle/health
state, and request the independent operational review before any deployment.
Expected baseline hashes in this manifest are requirements, not current console
observations. Do not replay prior consumed controls or automatically retry .40.

## Console preparation checkpoint

`make console-controls` now builds the missing teardown helper from pinned source.
Its successful artifact is `build/source-make-snapshot/teardown-snapshot.elf` in
the WSL worktree, with source/import receipt and repeated Make verification.
Failed-receipt and modified-artifact/log regressions fail closed. The freezer can
select a separately preserved helper using `--snapshot-build`. See
`evidence/offline-source-console-controls.json`. No console contact occurred.
The owner was asked whether .40 and its services have been restored; await that
status before any retry. Continue offline preparation meanwhile.

## Native launch package checkpoint

The native public launch client is integrated with Make, package inventory,
freeze controls and strict log validation. `make public-hip-launch-package`
completed all 74 objects in `/home/denis/ps5-hip-module-package/build/public-hip-launch-v1`;
session30848 is finished. Cached revalidation passed. All 62 host tests and
17 console-control test methods pass. See `evidence/offline-source-public-launch.json`
for package identities, raw logs and review. No candidate has been frozen or
sent to the console. The next hardware step requires .40 recovery and the
installed-image/preserved-log checks below, then a new committed, frozen and
independently reviewed candidate. Do not replay earlier deployment controls.

## Current implementation checkpoint

The public launch admission candidate passed fresh qualification in
`/home/denis/ps5-hip-module-package/results/public-launch-source-v4`.
Session9627 finished successfully. The69-unit base model and alternate launch
harness passed12base modes, exact73values/8guards, captured arguments, clean
ownership, and both output negative controls. See
`evidence/offline-public-launch-admission.json` for receipt and review identities.
Earlier v1/v2 were deliberately stopped for fixes; v3 had a runner path error.
Their records remain preserved; v4 is the fresh acceptance attempt.

## What is done

The public HIP loader now admits bounded owned raw AOT files, validates the copy
before publication, keeps bytes stable through program destruction, and uses
`addDeviceProgram`/`load` without unavailable runtime compiler services. Pointer-only
images return NotSupported without dereference. Single-device admission and
one extraction attempt are explicit. See `evidence/offline-public-module-aot.json`.

Fresh source model `results/public-module-aot-model-v2` passed all twelve cases
under ASan/UBSan across 69 genuine HIP/ROCclr units, binding 681 included files.
The complete fresh run includes real gfx1030 rejection, simultaneous modules,
owned-file unlink, malformed/size errors and effective unbundler flag assertions.
It supersedes the earlier v1 and incremental extension as current acceptance;
those raw results remain preserved. All 61 host tests passed; native
hip_fatbin.cpp compiled. The focused actual-extractor harness injects
read/close/allocation/parser failures and repeated attempts.

The native module package and fixture-specific console controls are complete offline.
`make public-hip-module-package MODULE_BUILD_DIR=build/public-hip-module-v2
RESULTS_DIR=/data/ps5-hip/results` built 74 fresh objects and a checked seven-file
package in `/home/denis/ps5-hip-module-package`. Session18327 exited successfully;
do not restart it. Cached revalidation and all 62 host tests passed. The v1 build
failed because the generic import verifier required AGC imports; its artifacts
remain preserved. The module-specific policy requires the exact shell binding
and rejects AGC dependencies while retaining serialized import-hash checks.
See `evidence/offline-source-public-module.json` for artifact identities/review.

The module client performs two rounds of simultaneous module load/lookup/unload,
checks survivor isolation, and rejects backend creation, allocations, queues or
GPU work. No candidate has been frozen, deployed or executed. The .40 hold below
still applies. Next offline implementation: public launch capability admission;
the preserved `results/public-launch-probe` diagnostic isolates a zero workgroup
limit in the public adapter, but its candidate is not production qualification.

The host qualifier is reproducible from source with
`make public-hip-host-test QUALIFICATION_DIR=results/<new-directory>`; no old
workspace receipts are needed. Platform providers and transport are modeled,
so this does not establish native PS5 loading or GPU execution. The earlier
source-only ten-case model and symbol/error-guard milestones remain recorded in
`evidence/offline-public-hip-source-model.json`,
`evidence/offline-public-module-symbols.json`, and
`evidence/offline-public-module-errors.json`.

Latest source milestone, pushed to `main`: `7880705` adds a source-built public
synchronous HIP copy fixture and propagates failed completion/status as an error
before releasing the command. All 58 host tests and the 74-object native package
passed; independent package/ISA review approved deployment only. See
`evidence/offline-source-public-copy.json`. Async copies and public HIP GPU
dispatch remain unqualified.

**Current console hold:** `results/source-copy-console-v1` deployment failed on
an FTP read bound before recorded image preservation/promotion. No copy launch
occurred. The prior memory log was renamed to its SHA-suffixed `.preserved` path;
the uploaded temporary image and all receipts must be retained. A read-only
installed-image fetch exceeded 120 seconds; a bounded follow-up timed out and
the final health probe returned `No route to host`. All owned locks were released.
Current installed-image integrity is unknown; service health from earlier phases
does not prove it. Do not replay v1 deployment or bind its execution approval.
After connectivity is restored, establish complete installed-image and preserved
log hashes read-only. Any further deployment needs a distinct frozen, reviewed
candidate that explicitly handles the already-preserved baseline log. Preserve
raw `deploy`, `inspect`, and `measure` phase receipts. Offline implementation can
continue while console testing is held.

Recovery controls now support a new reviewed candidate with
`--baseline-log-preserved` and the prior log digest/fixture. They verify the
existing preserved bytes without renaming, reject conflicting fresh logs and
recheck before replacement and after settling. FTP receipts retain partial-read
diagnostics with an explicit completion flag. Fifteen focused control tests and
the three related CTest cases pass offline. This changes frozen control hashes;
v1 stays consumed/inert. No new deployment has been attempted.

Latest progress: source `be6e521` adds the memory fixture to the supported Make
pipeline. All 57 host tests and 74 fresh-object native build/package checks
passed. The independently reviewed `results/source-memory-console-v1` case
then passed on .40: two distinct allocations (257 and 4097 bytes), interior-free
rejection with no retirement, two frees, double-free rejection, joined watchdog,
balanced finalizer and clean shell-requested shutdown. See
`evidence/native-source-public-memory-success.json`. Copied data and public HIP
GPU dispatch remain unqualified. The owner requests pushing every validated
milestone to `main`; implementation and evidence are published there.

Latest continuation: source-only Make at `7342726` passed all 56 host tests,
native core, 74 fresh objects, serialized SCE import/hash checks and package
validation. Independently reviewed v2 ran once on .40 as PID639/app `0x4018`:
`hipInit=0`, count=1, complete finalization, shell-exit return=0 and natural
stop passed, with matching mount observations, healthy services and owned lock
release. See `evidence/native-source-make-bootstrap-success.json` and raw
`results/source-make-console-v2`. Public HIP allocation is now qualified above;
this bounded startup result does not complete G5.

The preceding v1 run failed with a shell-exit SIGSEGV. Its package and log
were preserved before v2 deployment; its gate remains failed. The fix applies
the converter import-library patch omitted by Make, repairs import hash metadata
and verifies serialized bindings before signing. See
`evidence/native-source-make-bootstrap-failure.json` and raw
`results/source-make-console-v1`.
The user confirmed this host is the sole .40 controller and authorized an
independent review agent. The prior original-workspace recovery block is obsolete
for new source-built candidates; historical phases remain consumed/inert.

- G1 passed offline; G2 passed its named arithmetic fixture. G3 has substantial
  bounded native compute evidence, but broader qualification remains open. G4
  has genuine registered HIP clients, reusable runtime, HIPIFY translation,
  threading, fills, copies, reduction, and exact tiled matmul runs on `.40`, but
  broad G4 acceptance remains open. `STATE.md` and the linked compact evidence
  give each exact claim.
- Genuine upstream ROCclr passed a native CPU command/lifecycle workload and
  one GPU-backed A-write/B-copy/A-kernel/B-read cycle on `.40` (73 exact active
  values, eight copied guards, natural stop). The integrated GPU evidence is
  `evidence/native-clr-gpu-dispatch-success.json`, image SHA-256
  `84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da`.
  This is not public HIP GPU support.
- Genuine exported upstream `hipMalloc`/`hipFree` passed the 69-unit Linux
  PS5-registry model, ten ASan/UBSan and ten TSan modes, and independent review.
  Integrated commits `209664d`–`debf465`; see
  `evidence/offline-hip-public-memory-bridge.json`. Native public API behavior
  now has the bounded native allocation/free result above; copies and GPU launch remain open.
- A **clean 74-object native public HIP bootstrap** title was built, strictly
  linked, converted, packaged, and independently reviewed. Its one-use D control
  installed and registered image SHA-256
  `836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8`
  as PPSA99014 on `.40`. Independent D review confirmed exact mounted EBOOT and
  libc bytes, preservation of the prior GPU image/log and all 34 historical
  backups, idle title, healthy services, and owned-lock release. **It was not
  launched.** The compact record is
  `evidence/native-public-hip-bootstrap-deployment.json`.
- A relocatable source-only recipe on `main` independently builds a new
  74-object bootstrap package with verified FSELF and UFS2 extraction. This
  original package was an offline candidate. The corrected source-built v2
  package now passed the bootstrap above. A fresh GitHub-only clone at `7ef291c` passed
  default `make` (54 host tests, native core, 74 objects and package audit);
  see `evidence/offline-main-source-build.json`.
- Native public HIP allocation/free and native ROCclr FP32 GEMM candidates have
  passed offline native package and independent review. Neither was run on a
  console. The allocation title includes real `hipInit`/count/malloc/free symbols
  in 74 freshly compiled units. The corrected GEMM candidate passes GeometryLds
  admission, a 39-unit sanitizer model, a 46-unit strict native build, and
  package/control review. It targets a 9×11×13 FP32 result with 99 active values
  and 45 padding guards. No native GEMM numerical result exists.

Overall progress is roughly **20–22% of the twelve-goal ambition**, a judgment
about the full roadmap rather than a formal gate metric. G5–G12 are open.

## Historical branches and local material

| Work | Branch / frozen HEAD | Local checkout | Status |
| --- | --- | --- | --- |
| Supported source build | `main` / bootstrap `7342726`, memory `be6e521` | repository root | Full source builds and bounded .40 bootstrap/memory passed; pushed to GitHub. |
| Public HIP bootstrap D/E | `development/g5-hip-native-link` / `3ac8385` | `.local/g5-hip-native-link` | D consumed and independently cleared; E inert. |
| Public HIP memory title | `development/g5-native-public-memory` / `1f969f6` | `.local/g5-native-public-memory` | Reviewed offline; no native run. |
| Draft memory controls | `development/g5-native-public-memory-controls` / `1f969f6` | `.local/g5-native-public-memory-controls` | Untracked draft controls only; inspect before relying on them. |
| ROCclr GEMM title | `development/gemm-native-bridge` / `a0af56e` | `../ps5-hip-gemm-bridge` | Reviewed offline; stale console baseline, no native run. |

These branches preserve historical reviewed candidates; the supported
source-only bootstrap build needs only `main`. All three reviewed candidate branches above were published to the private
GitHub remote alongside the integrated branch. Nested worktree `origin` remotes
still point to other **local** checkouts, not GitHub; use the private GitHub
remote explicitly when publishing later work.
Do not assume an isolated branch has been merged into the integrated baseline.
The fresh native memory package SHA-256 is
`4a4ea8e14543e074b2324badf55f3138f35ddc418677547df716bea5c0c15818`;
its tracked evidence is `.local/g5-native-public-memory/evidence/offline-native-public-hip-memory.json`
and independent review is local at
`.local/g5-native-public-memory/results/native-memory-independent-review-v1/review.json`
(SHA-256 `f96cc4a5d5cf4330ff2582d0465149eb2d6d681a114596bf313662447a78b425`).
The GEMM package image SHA-256 is
`9c229e4221570716ec1cc2ee9bea08d41299bd6b433aad46a30ebf121f5777e4`;
see `../ps5-hip-gemm-bridge/evidence/offline-clr-gemm-admission-repair.json`,
package review at `../ps5-hip-gemm-bridge/results/gemm-package-independent-review-v1/package-review.json`
(SHA-256 `d7b80aadcd9188d2ad1e50d06ffe11da404946e42f3782db103affbd44c89915`),
and controls review SHA-256
`02ccafaeaa25a033364628955c4ad8b85182191f058b04319675bdf7e447a12f`.

## Historical console and one-use control state

The earlier read-only D review classified registration as
`clear-live-public-HIP-bootstrap-deployment-D`. Its raw D receipt is local and
untracked at
`.local/g5-hip-native-link/tools/public_hip_controls/deployment/deployment-20261014T025025.424073Z/deployment.json`
(SHA-256 `d665cf262827764cc0babc7ba645cbad3d61cb00cf20f2658bbca3f44afdf532`);
the independent review is local at
`.local/g5-hip-native-link/results/public-hip-bootstrap-deployment-independent-review-v1/review.json`
(SHA-256 `2295552024df10e0f8cab93e1bf8494b1c9e9f57c104e949a810f338b9fdc3fe`).
The first D attempt failed in local preflight due an incomplete prior GPU-log
hash, before remote mutation. The corrected D passed; **do not repeat either D**.

The D worktree also contains four untracked `*-consumed.json` phase markers under
`tools/public_hip_controls/deployment/`. Keep those and the raw receipt in place;
they are local-only and are not included in a fresh GitHub clone. A new agent
working from a fresh clone needs this preserved workspace evidence before
binding E; do not fabricate or replay D to replace it. Do not clean the
worktree or replay a consumed phase. The E manifest at
`.local/g5-hip-native-link/tools/public_hip_controls/execution/manifest.json`
has `deployment_binding: null`; `run.py` still raises `INERT` at its activation
point. Its `verify_local()` currently expects D review classification `clear`,
while the actual reviewed classification is the longer value above. This was
fixed in the source-only continuation below. The historical E remains inert;
the fresh v2 control now supplies the qualified startup result.

The last D observation found `.40` idle with FTP 2121, klog 3232, and ELF loader
9021 healthy. Both shared lock files (`C:\Users\denis\Documents\PS5\lock.txt`
and `C:\Users\denis\Documents\PS5-Claude\lock.txt`) were absent at the last
local check. This is historical state; recheck immediately before any console
use. **Do not use console `.30`**. Use `.40` only under the current runbook and
ignored `.local/ENVIRONMENT.md`. The user permits .40 proof-of-concept tests when
useful, but wants candidates finished and reviewed offline first. Preserve the
installed image, old GPU image/log, backups, and any user data.

## Scope update — 2026-10-14

The owner removed ProsperoAI integration. G7 now targets a standalone ggml/llama
inference client and SDK preview; application-specific ABI, UI and model coverage
are no longer acceptance requirements. Historical source/evidence records remain
provenance. No implementation gate is closed by this scope change.

## Fresh-checkout continuation � 2026-10-14

This checkout is now on local `main`, with the standalone scope and corrected
source-build pipeline integrated. An isolated historical bootstrap checkout
at `.local/g5-hip-native-link` is on `codex/public-hip-bootstrap-resume`, commit
`98a6887`. It fixes the known D-review classification mismatch and adds seven
binding tests; these and the inert-entry test pass under WSL. Execution remains
inert and unbound, with independent control review still pending.

The owner identified the original workspace as
`C:/Users/denis/Documents/PS5/workspace/dev/ps5-hip`. Both Windows and WSL report
that path absent on this host. The raw deployment receipt/review, consumed markers,
frozen package and local controller/helper dependencies have not been recovered.
The new nested checkout contains source only; do not mistake it for the original
preserved workspace or replay deployment. Recovery is not required for the new
Make-built v2 package and its fresh D/E receipts.

The current installed PPSA99014 package on .40 is
`37decef81b2d32dace7662f0c003322a09da234275a92191628524a027a1e1b5`.
Its current memory log hashes to
`1425fabbd3a010b3be3bf3dc656c5c6fc1b39e4c4c8d9e4fe7491b82bf06791b`.
Both phases under `results/source-memory-console-v1`, and both prior bootstrap
cases, are consumed and must not be replayed. Previous image/log pairs remain
preserved. Services were healthy and the owned lock released at completion.
Use the current dedicated-controller environment in `.local/ENVIRONMENT.md`;
the sole lock for this session is `C:/Users/Denis/Documents/PS5/lock.txt`.

## Next work, in order

1. Keep the supported source pipeline on `main`; preserve both consumed fresh
   cases and their independent reviews. Do not reactivate historical D/E phases.
2. Implement and qualify the next public HIP data-copy/module/launch path using
   the current memory image/log baseline. Preserve the bounded memory result;
   allocation/free does not establish byte correctness or GPU execution.
3. Refresh GEMM D/E controls against the actual then-installed PPSA99014 image,
   keep the corrected GeometryLds source frozen, and independently review before
   a numerical `.40` run. No GEMM GPU success is presently claimed.
4. Continue G5 public HIP GPU launch and the later stream/event/library,
   standalone inference/SDK, framework, training, performance, on-console
   compilation, and
   multi-console gates in `docs/IMPLEMENTATION_PLAN.md`.

Before any hardware experiment, follow `AGENTS.md` and
`.deps/sources/protocol/docs/RUNBOOK.md`: state the hypothesis,
expected result, rejection criterion and timeout; inspect ISA; freeze and hash
the candidate; check console idleness and the current environment's lock; use one bounded
launch; audit title stop, services, mounts, and owned lock release. Keep raw
logs local and commit compact sanitized evidence. The latest qualified console
observation is the source-Make public memory case above.
