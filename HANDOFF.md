# ps5-hip agent handoff

Updated: 2026-10-14, America/New_York. This is the resume point for the private
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

## What is done

Latest continuation: source-only Make at `7342726` passed all 56 host tests,
native core, 74 fresh objects, serialized SCE import/hash checks and package
validation. Independently reviewed v2 ran once on .40 as PID639/app `0x4018`:
`hipInit=0`, count=1, complete finalization, shell-exit return=0 and natural
stop passed, with matching mount observations, healthy services and owned lock
release. See `evidence/native-source-make-bootstrap-success.json` and raw
`results/source-make-console-v2`. Public HIP allocation/GPU execution is next;
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
  beyond initialization/discovery is still untested.
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
| Supported source build | local `main` / tested source `7342726` | repository root | Full Make and bounded .40 bootstrap passed; local continuation not pushed. |
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
`403d8d0364ec57dea5342a77676fa7fd15ce74d7c7b1dbf0f823ce9784e01118`.
Its current bootstrap log hashes to
`42d39b5728e57d752937b023a49ccf4eceabbdff463747c68159a09e4696be1b`.
Both v2 phases under `results/source-make-console-v2` are consumed and must not
be replayed. The previous failed image/log and all historical backups remain
preserved. Services were healthy and the owned lock released at completion.
Use the current dedicated-controller environment in `.local/ENVIRONMENT.md`;
the sole lock for this session is `C:/Users/Denis/Documents/PS5/lock.txt`.

## Next work, in order

1. Keep the supported source pipeline on `main`; preserve both consumed fresh
   cases and their independent reviews. Do not reactivate historical D/E phases.
2. Update the native public memory control baseline to the current v2 image/log, review its
   controls independently, and test its genuine exported API title separately.
   Do not infer native malloc/free from the Linux model or bootstrap startup.
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
observation is the source-Make v2 bootstrap above.
