# ps5-hip agent handoff

Updated: 2026-10-14, America/New_York. This is the resume point for the private
`ps5-hip` repository. Read `AGENTS.md`, `STATE.md`, and the active goal in
`docs/IMPLEMENTATION_PLAN.md` before working. The mission is a native PS5 HIP /
ROCm compute stack; a bounded result does not close a broader acceptance gate.
For a new computer with GitHub source only, read `docs/SOURCE_ONLY_REBUILD.md`:
the exact bootstrap package is not yet reproducible end-to-end from a clean
clone, and historical D evidence cannot be built from source.

## What is done

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
  is still untested.
- A **clean 74-object native public HIP bootstrap** title was built, strictly
  linked, converted, packaged, and independently reviewed. Its one-use D control
  installed and registered image SHA-256
  `836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8`
  as PPSA99014 on `.40`. Independent D review confirmed exact mounted EBOOT and
  libc bytes, preservation of the prior GPU image/log and all 34 historical
  backups, idle title, healthy services, and owned-lock release. **It was not
  launched.** The compact record is
  `evidence/native-public-hip-bootstrap-deployment.json`.
- Native public HIP allocation/free and native ROCclr FP32 GEMM candidates have
  passed offline native package and independent review. Neither was run on a
  console. The allocation title includes real `hipInit`/count/malloc/free symbols
  in 74 freshly compiled units. The corrected GEMM candidate passes GeometryLds
  admission, a 39-unit sanitizer model, a 46-unit strict native build, and
  package/control review. It targets a 9×11×13 FP32 result with 99 active values
  and 45 padding guards. No native GEMM numerical result exists.

Overall progress is roughly **20–22% of the twelve-goal ambition**, a judgment
about the full roadmap rather than a formal gate metric. G5–G12 are open.

## Exact branches and local material

| Work | Branch / frozen HEAD | Local checkout | Status |
| --- | --- | --- | --- |
| Integrated baseline | `development/offline-bringup` / current remote HEAD | repository root | Documentation handoff published to the private GitHub remote. |
| Public HIP bootstrap D/E | `development/g5-hip-native-link` / `3ac8385` | `.local/g5-hip-native-link` | D consumed and independently cleared; E inert. |
| Public HIP memory title | `development/g5-native-public-memory` / `1f969f6` | `.local/g5-native-public-memory` | Reviewed offline; no native run. |
| Draft memory controls | `development/g5-native-public-memory-controls` / `1f969f6` | `.local/g5-native-public-memory-controls` | Untracked draft controls only; inspect before relying on them. |
| ROCclr GEMM title | `development/gemm-native-bridge` / `a0af56e` | `../ps5-hip-gemm-bridge` | Reviewed offline; stale console baseline, no native run. |

All three reviewed candidate branches above were published to the private
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

## Current console and one-use control state

The latest read-only D review classified registration as
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
while the actual reviewed classification is the longer value above. The next
agent must fix that exact check and bind the receipt/review identities before
freeze and independent review. **No public HIP E control has been activated.**

The last D observation found `.40` idle with FTP 2121, klog 3232, and ELF loader
9021 healthy. Both shared lock files (`C:\Users\denis\Documents\PS5\lock.txt`
and `C:\Users\denis\Documents\PS5-Claude\lock.txt`) were absent at the last
local check. This is historical state; recheck immediately before any console
use. **Do not use console `.30`**. Use `.40` only under the current runbook and
ignored `.local/ENVIRONMENT.md`. The user permits .40 proof-of-concept tests when
useful, but wants candidates finished and reviewed offline first. Preserve the
installed image, old GPU image/log, backups, and any user data.

## Next work, in order

1. In the bootstrap branch, create a new E-only binding to the exact D receipt
   and review above. Correct `verify_local()`'s review classification, retain the
   one-use/freshness guards, run focused offline checks, freeze source/artifacts,
   and obtain an independent review. Only then consider one bounded `.40` launch
   if it is idle and all service, identity, lock, and result-freshness checks pass.
   Verify public HIP initialization/result and natural teardown, and record a
   compact execution outcome. A failure must be recorded as a failure, not
   replayed blindly.
2. Depending on E, update the native public memory control baseline, review its
   controls independently, and test its genuine exported API title separately.
   Do not infer native malloc/free from the Linux model or bootstrap startup.
3. Refresh GEMM D/E controls against the actual then-installed PPSA99014 image,
   keep the corrected GeometryLds source frozen, and independently review before
   a numerical `.40` run. No GEMM GPU success is presently claimed.
4. Continue G5 public HIP GPU launch and the later stream/event/library,
   ProsperoAI, framework, training, performance, on-console compilation, and
   multi-console gates in `docs/IMPLEMENTATION_PLAN.md`.

Before any hardware experiment, follow `AGENTS.md` and
`../../docs/ps5-homebrew-dev-protocol/docs/RUNBOOK.md`: state the hypothesis,
expected result, rejection criterion and timeout; inspect ISA; freeze and hash
the candidate; check console idleness and both lock locations; use one bounded
launch; audit title stop, services, mounts, and owned lock release. Keep raw
logs local and commit compact sanitized evidence. This handoff records no new
console observation.
