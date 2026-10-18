# Current state

Updated: 2026-10-18, America/New_York. See `HANDOFF.md` for the exact
resume point, branch map, and evidence identities.

## Native corpus deployment/readback integration

Corpus candidate freezing, nested capture, numerical readback and full application
log validation are implemented. The manifest binds 1,830-second observation,
600-second capture and 4,200-second whole-phase limits, with a corpus-only 128 MiB
reference bound. The controller verifies all 6,539 captured files and supports
2 MiB corpus logs, including preservation for the next deployment. 277 focused
controller/model/readback/transcript tests pass; see
`evidence/offline-native-corpus-controller.json`.

Next build one fresh corpus package, inspect all three linked-image audits, freeze
the exact candidate and test on idle .40. No new native package or console result
exists yet. Reuse the existing model, CPU oracle and kernel modules; do not rerun
the completed host numerical corpus solely for these controller changes.

## Native corpus title and asset integration

The native build now admits the explicit corpus profile: ABI operand 3, the
same six GPU modules and external model, 272 decoder commits, 6,539 output files
and 86,091,236 total bytes. The title creates five fresh case directories, rejects
unexpected output names and preserves exclusive writes. Corpus selection uses
the fixed 30-minute watchdog; smoke and short-generation limits stay unchanged.
The image auditor requires the generation call chain for corpus as well.
All 107 focused tests pass, including compiled production writer/dispatch tests.
See `evidence/offline-native-corpus-profile.json` for source and test-log hashes.

This is offline integration, not a linked package or hardware result. Next finish
corpus support in candidate freezing, nested readback collection/comparison,
application log parsing and controller deadlines. Existing deployment admission
still rejects corpus until those pieces are complete. Then build once, inspect
the actual linked images and run on idle .40. The arithmetic milestone is pushed
as `086e370`; no console contact occurred during this integration.

## M4 production-helper corpus arithmetic passes offline

`results/m4-corpus-native-generation-numeric-v1` completes with final PASS.
The actual production helper and ggml scheduler execute all five cases through
host interpretation of HIP arithmetic: 272 commits, 22 layers and 32,000 logits
per commit. All 6,528 float files are byte-for-byte equal to CPU corpus reference
v2 (86,065,152 bytes); maximum absolute and relative-L2 errors are zero. Exact
metadata, live greedy feedback, reset repeats and all numerical gates pass.

The corpus transcript validates 285 timing intervals, 11 metadata files / 26,084
bytes, 217,056 nodes and 199,376 interpreted launches. Ownership is balanced:
16 allocations/frees and one created/freed backend instance. ASan/UBSan found no
diagnostics; prebuilt CPU libraries are not instrumented. Host execution took
542.509 seconds including diagnostic capture; this is not PS5 performance.
63 generation tests, 18 Make tests and four watchdog tests pass. See
`evidence/offline-production-helper-corpus-numerics.json` and the local receipt.
`make llm-native-generation-corpus-tests` now selects the corpus oracle and
production helper while preserving the short target. The watchdog admits the
fixed 1,800,000-ms corpus duration; existing title arguments remain unchanged.

Next coordinate explicit native corpus admission: title writer/selection,
asset profile, three-image audit, frozen nested inventory/readback, application
log/baseline bounds, watchdog selection and controller deadlines. The exact map
is in `docs/LLM_GENERATION.md`. Then build one fresh package and validate on idle
.40. No console contact or native package build occurred in this milestone;
the installed short-generation proof remains the native baseline. No process
from this offline corpus run remains active; do not restart the completed run.

## M4 five-prompt CPU corpus and production-helper transport pass

`results/m4-generation-corpus-reference-v2` passes real TinyLlama CPU capture:
five fixed continuation prompts, 16 generated tokens each, two reset rounds,
272 actual decoder commits, 6,528 float outputs / 86,065,152 bytes plus 11 metadata
files. Every layer, normalization and logits boundary repeats bitwise within
each case. Shared cases are in `ports/ggml/generation_cases.def`; Make exposes
`llm-generation-corpus-reference` with a separate output directory. Existing CPU
libraries and model were reused. The v1 science instruction stopped at EOS and
was correctly rejected; preserve that failed attempt and its raw evidence.

The production helper implements corpus profile 3 with one resident model/context
and case-specific paths/logs. `results/m4-corpus-native-helper-v1` passes changing
mock-logit feedback, ten resets, one created/freed instance, exact file inventory,
balanced ownership, EOS rejection and clock rejection under ASan/UBSan. Sequence,
smoke and short-generation regressions also pass. This mock does no tensor math.
55 generation test methods and 17 Make tests pass; independent native-helper
review found no defect. See `evidence/offline-five-prompt-generation-corpus.json`.

Next integrate corpus in the HIP numerical builder/comparator and strict transcript
parser, then run the actual production helper against the CPU oracle offline.
The numeric builder currently rejects corpus explicitly. After that pass, extend
native title writer/selection, asset profile, image audit, frozen readback inventory
and bounded application/capture deadlines. Only then build one fresh native
candidate and validate on idle .40. No console contact or native package build
occurred for this milestone; the installed short-generation baseline is unchanged.

## Native-client duplicate identity reads removed

Initial artifact, object and tool comparisons now reuse the native client's
freshly verified input digests. Unbound artifacts still require a byte read;
path, size, inventory and membership checks remain. Final input and artifact
checks still read every byte afresh. All 56 focused client/app/loader tests pass,
including six new trust-boundary tests; independent source review found no defect.
The retained v7 receipt implies 170,977,143 redundant bytes avoided per client
authentication call (169 reads); this is a static count, not a timing claim.
See `evidence/offline-native-client-identity-reuse.json`.

No native rebuild or console contact was needed. Next implement the fixed
five-prompt / 16-token M4 corpus across oracle, production helper and validators,
with reset/repeat and all-boundary comparison offline before one native build.
Retain the short generation regression. Corpus-specific bounded application and
capture deadlines must account for measured diagnostic time; do not reuse the
150-second short-case budget. Further producer authentication overlap remains
open, but these tooling changes do not need another standalone console cycle.

## Frozen manifest verification now overlaps filesystem reads

The controller verifies frozen manifest files with four bounded workers. Every
invocation still reads fresh bytes and checks resolved path containment; manifest
identity is checked first, worker failures propagate, and all workers finish
before control returns. Producer authentication and final checks remain intact.
All 288 source/controller regression tests pass, including seven fresh-read,
tamper, symlink, I/O failure and worker-lifetime cases.

A warm-cache alternating benchmark of 512 existing manifest inputs (7,967,282
bytes) measured median 6.643 seconds serial versus 1.121 seconds parallel.
This measures the manifest scan only, not total deployment or execution time.
See `evidence/offline-parallel-frozen-verification.json`. No native rebuild or
console contact was needed. Repeated producer authentication remains the next
bottleneck; preserve all consumed native proof and obtain fresh source-bound
receipts for future candidates.

## Build prerequisite rereads reduced after the native milestone

Initial native-app prerequisite closures now merge their already-authenticated
path/digest identities without hashing the same bytes immediately again. Unverified
operands, consumed headers and image-audit inputs retain ordinary hashing; fresh
final consumers, symlink/inventory checks and uncached final recheck are unchanged.
The locked generation case avoids three GGUF reads and one safetensors read,
4,059,614,520 bytes plus other closure rereads. No elapsed-time speedup is claimed.
All 73 focused builder, freeze and Make test methods pass; no native rebuild or
console cycle was needed for this tooling change. See
`evidence/offline-authenticated-prerequisite-merge.json`.

The consumed native generation proof below remains bound to source `42f74b3`.
Current tooling/test identities changed, so retain its historical receipts and
raw outputs; obtain a fresh current-source app receipt for any new candidate.
Next reduce repeated producer validation in the freeze/console-control path;
this small builder change alone does not resolve that larger bottleneck.

## M4 short native greedy generation passes on .40

The installed PPSA99014 generation app produces `, there was a` from
`Once upon a time` using actual live-logit feedback, then repeats after a real
KV/backend reset. All 16 commits execute 22 layers and 32,000 logits through
public HIP: 12,768 nodes and 11,728 GPU launches. All 384 captured float outputs
are bitwise CPU-oracle equal; both metadata files and reset/token histories match.
Native kernel fixture v8's generation arithmetic is now qualified for this case.

Independent combined execution review passes: 411 host allocations/releases,
16 GPU allocations/frees, one retired queue, restored largest free extent,
normal title stop, unchanged mounts, healthy services and both locks released.
The 12,882,804,736-byte largest free extent recovers; total free memory remains
unknown. Decode timings range 4.401-4.579 seconds including boundary readbacks
and file writes; this is a diagnostic correctness run, not throughput evidence.

See `evidence/console40-native-short-generation-success.json` and
`results/m4-native-generation-build-review-v1/execution-review.json`.
`results/source-public-llm-generation-console-v1` is consumed: never replay it.
The installed baseline is package SHA
`4032fac68e2761ac4d1bb10aceb2ad38155a0150ca05958559c35dc4da70bca3`,
22,478,848 bytes, results `/user/homebrew/ps5-hip/m4-greedy-v1`, application log SHA
`9940d2ce2bcf654f786e53c9b29090dd4929664a2b0f19676d434f730bfeb095`.
The external model was verified and reused at `/data/hip/models/tinyllama-q4_0.gguf`.
Preserve the prior smoke package/log recovery and all raw generation evidence.

Before wider console tests, reduce repeated host-side build/control input
verification while retaining frozen package/reference/controller identities,
model validation and native lifecycle checks. The package upload took 2.414
seconds; deployment took 1,783 seconds and execution control 1,489 seconds,
including fresh local authentication. Cost analysis and an unapplied small
builder improvement are in the review folder. M4 wider prompts/context, M2
remaining dense/extreme fixtures, M5 and original-format native loading remain open.

## M4 native generation package and linked audit pass

`build/llm-native-generation-app-v1` finishes with 149 authenticated objects,
201 commands and a 415-function raw/converted/extracted-image audit. Generation
profile operand 2 and the actual helper/sampler/process/decode/EOS/greedy edges
pass. Independent review reproduces the audit and extracts all 15 package files,
confirming six GPU modules and external `/data/hip/models` model storage. The
package is 22,478,848 bytes; weights are not embedded. Native client v7 and the
prepared public source/compiler paths were reused.

See `evidence/offline-native-generation-linked-package.json` and
`results/m4-native-generation-build-review-v1/linked-package-review.json`.
This is offline package qualification: no generation execution or new native
arithmetic is claimed. Next freeze/review a fresh candidate, verify idle .40,
and validate 16 commits, all 384 float boundaries and both token metadata files,
reset/repeat, timings and clean ownership. Keep the consumed smoke baseline.
The exact freeze command is drafted in `.local/native-generation-freeze-command-v1.json`.
Repeated input authentication, rather than recompilation, is a measured build
phase bottleneck; a read-only cost analysis is in the same local review folder.

## M4 native generation profile and audit contract prepared

The builder now derives the title's compile-time generation selection from its
authenticated six-module asset receipt. Smoke stays the default. Generation checks
16 requests, 386 files/5,064,354 bytes, 12,768 nodes and 11,728 launches, and emits a
profile-specific application completion marker. Make selects the generation CPU
oracle with `LLM_FULL_PROFILE=generation` and rejects a reused aggregate with the
wrong profile.

The native generation title object compiles with consumed headers bound. Its
actual ABI operand is 2, and the helper/sampler/process/decode/EOS/greedy symbols
match the new audit selectors. The auditor adds their selected direct call edges,
prompt/feedback process callsites and matching profile operand while retaining
common scheduler/HIP, allocator, loader, callback-address and cleanup checks.
Complete indirect feedback dataflow is not claimed by this static audit.
See `evidence/offline-native-generation-profile.json`; 129 focused test methods
pass. This is object/contract qualification, not a linked-image audit.

Native client `build/llm-native-client-v7` passes current authentication and its
70 objects can be reused; no replacement client build is needed. Reuse prepared
public source/compiler paths from `build/llm-native-full-smoke-app-v1`. Preserve
`results/m4-native-title-preflight-v1`, including the initial section-count
inspection failure: the general object reader was replaced with llvm-nm for this
preflight, without weakening linked-image bounds.

Next build the generation package once with those authenticated inputs, inspect
its actual raw/converted/extracted images and finish the full audit. Then commit,
freeze/review and test the fresh candidate on idle .40. No native package build or
console contact occurred in this contract milestone. The installed smoke baseline
and consumed candidate remain untouched.

## M4 generation freeze, readback and application contracts integrated offline

The generation profile now selects its CPU oracle during reference freezing and
binds all 384 float outputs plus both metadata files. Admission requires a native
image audit that explicitly identifies `generation` and proves the actual
generation call chain; the old smoke audit cannot authorize it. The default smoke
profile and its raw contract are retained.

The controller collects the exact 386 files, rejects stale metadata before launch,
uses hash-bound frozen token policy, and requires numerical comparison of the same
captured bytes. The generation application parser checks startup, four memory
snapshots and recovery, phase/request timing, ownership balance, watchdog join and
finalization. Shared framing/memory checks retain the smoke contract. These are
offline parser/controller tests, not native-origin evidence.

`make llm-native-generation-tests` now runs generation application/readback/freeze/
controller checks and the production helper. The refreshed numerical helper is
`results/m4-native-generation-numeric-v6`: all 384 outputs remain bitwise CPU-equal,
with balanced storage and no CPU backend tensor arithmetic. CPU oracle v3 and
six-module assets v1 remain current. See
`evidence/offline-generation-freeze-and-controller.json` for qualification details.
The real reference-only freeze/readback case is
`results/m4-generation-reference-freeze-v1`: original/copy authentication and
386-file comparison pass, all 384 float outputs bitwise CPU-equal. Those readbacks
were copied from offline v5 output; they are not console captures. Preserve the
reference manifest, comparison, receipt and ignored qualification script.

Removed only v5's superseded host executable and two objects (4,457,720 bytes),
after process and current-receipt dependency checks. Ledger:
`results/cleanup-generation-numeric-v5-1792288749477757541.json`. Raw outputs,
receipts, logs and sources remain; v5 requires rebuilding pruned host outputs for
a complete artifact rehash.

Next wire the native title/builder compile-time generation selection and extend
the linked-image audit to its real helper/sampler/decoder path. The current title
still selects smoke, and the audit cannot yet produce generation proof. Do not
build/deploy a generation candidate until those checks pass. No native package
build or console contact occurred; preserve the installed .40 smoke baseline.

## M4 generation now compares every CPU boundary offline

The greedy CPU oracle now captures all 22 layer outputs, final normalization and
32,000 logits for every committed request, before activation reuse. Its completed
callback logits must equal the actual sampler input. All 384 outputs from the
production native generation helper under numerical HIP interpretation are
bitwise CPU-equal, including reset/repeat. Token histories still produce
`, there was a`. The existing absolute/relative/L2 budgets are unchanged.

Use `results/m4-generation-reference-v3` and
`results/m4-native-generation-numeric-v5`; earlier references/numerical receipts
are historical after these source changes. A copied oracle authenticates using
its bound original producer command paths while checking all copied artifacts;
a deliberately mutated intermediate is rejected. This makes the reference usable
inside a frozen candidate. The complete reference is about 5.4 MB.

Full assets now accept an explicit `generation` profile, bind its CPU oracle,
384 float files and both actual metadata files, and retain the default smoke
contract. The actual six-module closure passes and is recorded in
`results/m4-generation-assets-v1`. See
`evidence/offline-generation-all-boundaries.json`. The native title,
linked image audit, freezer, application transcript/controller and readback
consumer still need generation profile integration. No native package build or
console contact occurred; host interpretation proves no new device arithmetic.
Preserve the installed .40 smoke baseline and its consumed candidate.

## M4 production generation helper integrated and timed offline

`NativeFullProfile::generation` now uses the shared native model/context lifecycle,
actual tokenizer and live-logit greedy feedback. It captures every layer/norm/logit
at every short-sequence position, writes actual generation/commit metadata, clears
real KV/backend state before each round and compares repeat token histories.
Captured logits must equal the exact logits consumed by the sampler. The helper
reports checked monotonic load/context/reset/decode/run/cleanup timings; decoder
intervals include callback readback/file writes. Run overhead combines scheduling,
sampling and logging; it is not isolated sampling or GPU throughput.

The production helper passes numerical HIP interpretation against the refreshed
CPU oracle: 16 committed decodes, all 16 full-logit files bitwise CPU-equal, 384 raw
boundaries plus 2 metadata files, exact reset and balanced GPU-storage ownership.
See `evidence/offline-native-generation-helper.json`. Transport tests also pass
514-request sequence, smoke 3, greedy 16 and forced-EOS 10 profiles. The EOS case
observes only prompt inputs: selected EOS is never submitted. A regressing clock
fails in an isolated host process. These transport mocks are not GPU arithmetic.

The shared driver now reserves the full combined 512-token prompt/output budget,
including the final selected token. A 512-token prompt plus one output is rejected
before any decoder submission. This fixes the prior one-token budget discrepancy;
the short generated history is unchanged. Preserve CPU oracle
`results/m4-generation-reference-v2`, numerical helper
`results/m4-native-generation-numeric-v4`, and mock qualification
`results/m4-native-generation-mock-v9` with their logs/receipts/raw data.

Next wire the explicit generation profile through the native title/builder,
profile-specific assets/reference, linked image audit, freezer, full application
transcript/ownership and controller. The current title still selects smoke;
helper transcript validation alone proves neither native origin nor finalization.
Do not deploy a generation candidate until those contracts pass offline and a
fresh cache-backed client/package is frozen/reviewed. The installed .40 smoke
baseline remains unchanged. No console use or native package build occurred.

## M4 short greedy arithmetic now passes offline

The new full-mode GPU-source SiLU, softmax and RoPE routines pass the unchanged
16-request generated-history gate. All 16 complete 32,000-logit outputs are
bitwise CPU-identical, with actual token feedback, KV reset/repeat, 12,768 nodes,
11,728 interpreted launches and balanced 16/16 storage plus 1/1 instances.
All 384 raw boundary files are finite and reset exactly; CPU numerical comparison
in this profile covers logits. There is no CPU activation substitution in this
passing production test. See `evidence/offline-greedy-canonical-arithmetic.json`.

Diagnostics isolated SiLU differences and a passing combination of polynomial
exponential/softmax arithmetic and RoPE contraction. Full mode now uses the
pinned AVX2 lane arithmetic and explicit RoPE FMAs; legacy selected-layer symbols
and its 12-function-query route are unchanged. The original full smoke regression
still matches all 72 CPU outputs bitwise. Standalone math passes 33 SiLU / 134 softmax
cases in both sanitized and optimized modes; 37 focused test methods and direct
kernel/node programs pass.

`results/llm-layer-fixture-v8` is the current 16-symbol device module: all kernels
pass Initial admission, no undefined symbols/LDS/private/spills, and the Q4
16-byte-load contract is retained. New SiLU/softmax/RoPE use 12/23/18 VGPRs.
Preserve `results/m4-generation-canonical-v1`, `results/m4-cpu-math-v2`,
`results/m4-full-smoke-regression-v1` and their raw evidence. Fixture v7 has
identical shader bytes but a superseded math-test source closure; use v8.

Next integrate an explicit native generation profile, full short-sequence
capture/feedback/EOS/reset contracts and monotonic phase/request timing. Bind
its reference through assets, image audit, freezer and controller before a fresh
cache-backed native build and .40 test. Current client/app receipts are historical
for the old backend: do not pretend they authenticate these changed sources.
The accepted installed smoke baseline below is unchanged. No new native package
or console run has occurred; GPU denormal/transcendental results remain unproved.

## Historical first M4 numerical failure (resolved offline)

The shared bounded generation driver and source-only Make targets now execute
actual tokenizer/prefill/greedy feedback, with four selected tokens and explicit
KV reset/repeat. The CPU oracle for `Once upon a time` passes 16 real decoder
commits and produces `, there was a`; full logits repeat bitwise. All 24 focused
tests pass, including sanitized driver checks, EOS, overflow-before-submit,
independent argmax/feedback, malformed evidence and reset drift.

The HIP host interpreter completes the same 16 commits, 12,768 active nodes and
11,728 launches with 16 balanced allocations/frees and no CPU backend tensor
operations. Token IDs match and all 384 boundary outputs reset exactly, but
**all 16 full-logit comparisons fail** unchanged numerical budgets: maximum
absolute error 0.19304943, maximum relative L2 0.00918597. Do not call this GPU
generation acceptance or deploy it. See
`evidence/offline-greedy-generation-reference-and-regression.json` and
`docs/LLM_GENERATION.md`.

Preserve CPU oracle `results/m4-generation-reference-v1`, failed numerical case
`results/m4-generation-numeric-v1`, independent review and local diagnostic
folders. CPU callback splitting and fusion-environment changes do not explain
the failure. A diagnostic-only upstream CPU SiLU substitution restores bitwise
logits at positions 0/1 in both runs, but 12 outputs still fail; substituting
softmax as well does not finish the fix. These injected CPU operations establish
no GPU capability and must never enter the accepted runtime.

The current offline fix above supersedes that numerical failure; preserve it
as history. Native generation still needs packaging, timings, image audit, a
new frozen candidate and .40 test. The accepted installed native
smoke baseline below remains unchanged; no console contact or native rebuild
was needed for this offline milestone. Longer context and all M4/M5 gates stay open.

## Native full decoder and logits validated on .40

The native PPSA99014 app passed at frozen source `3ef7c74` on 2026-10-17.
Three real decoder commits (token 0/position 0, token 1/position 1, then token 0
at position 0 after actual KV/backend reset) execute all 22 TinyLlama Q4 layers,
final weighted normalization and 32,000 logits. All 72 raw outputs (949,248
bytes) are bitwise identical to the authenticated CPU reference; reset is exact.
The audited HIP path executes 2,394 active nodes and 2,199 GPU launches.

The aligned native staging fix resolves the earlier model-load failure:
411 host allocations/releases and 16 GPU allocations/frees balance; one queue
retires normally, with no allocation failure or poisoned storage. Largest free
extent is 12,882,804,736 bytes before load and after cleanup, 12,253,069,312
after load and 12,206,473,216 after context creation. These are largest extents,
not total-free-memory measurements. The external model at
`/data/hip/models/tinyllama-q4_0.gguf` was verified and reused, not embedded.
The app stops normally, services remain healthy and both owned locks release.
See `evidence/console40-full-decoder-smoke-success.json`.

Preserve consumed `results/source-public-llm-full-smoke-console-v2` and
`build/llm-native-full-smoke-app-v2`; never replay the candidate. The installed
baseline is package SHA-256
`0c203527c52ab9e1645ae0428df1dc596ee081e64ed250c57322bf015d957571`,
log `eaf552b7651620275bc56b18000a21c637a8423472d5232d1b3fb13a2c2c39a8`,
in `/user/homebrew/ps5-hip/m3-full-smoke-v2`. Final execution receipt is
`17c3d3572f66d5b0ea2f13984d944f2c36b09491be583b0c209ca97bb0e6f57a`;
independent terminal review is `results/m3-native-full-build-review-v2/execution-review.json`.

Reuse client v7, app v2's v1 prepared source paths, full assets v2, vocabulary v3,
GPU layer fixture v6 and CPU full reference v1. Keep their receipts and inputs.
No native rebuild is needed just to record this milestone. Source-only
reproduction builds new receipts; local historical evidence is not in GitHub.

Next: short deterministic greedy generation using GPU logits and actual KV
progression, with monotonic phase/per-request timings and CPU reference checks.
The observed approximately 17-second whole-app lifecycle is not tokens/second
or per-request latency; do not extrapolate it to the 514-request profile or
run that profile under unchanged deadlines. Generated text, long full-model
context, remaining dense/extreme-input fixtures and release acceptance remain
open. Numerical-only GPU-origin and audit dataflow limitations are unchanged;
native acceptance combines image, deployment, audited routing and execution.

## Historical first full model-load failure (resolved)

The consumed v1 candidate failed before load_complete with ordinary-heap
`std::bad_alloc`. Source review identified a 36,864,000-byte upload staging
vector; the native aligned replacement passed v2. There is no captured
allocation backtrace. Native-only RAII staging now passes constrained-heap,
sanitizer, failure-path,
actual machine ABI/cleanup audits and the successful v2 console execution.
Keep `evidence/console40-full-model-load-allocation-failure.json`, consumed v1,
its package and logs as history; package `700c673e179b111de7106297fb410667fc97505b76b9cbbeb59c8fe81c8aa894`
is the superseded baseline. Do not rewrite failed receipts or replay v1.

## Native selected-layer milestone

On 2026-10-17, the selected TinyLlama Q4 first layer passed on .40 as the native
PPSA99014 app at source `f251366`. All 37 nodes ran for 512 positions plus reset
and two repeats: 514 requests, 16,962 GPU launches and 151 validated raw files.
All 21 sampled stages, including the final residual, passed; reset outputs
matched exactly. Twelve resident owners and 528 allocations/frees balanced,
and the largest free extent returned to 12,882,804,736 bytes.
See `evidence/console40-model-layer-success.json` and `docs/LLM_MODEL_LAYER.md`.

Preserve `build/llm-native-layer-app-v1` and consumed candidate
`results/source-public-llm-layer-console-v1`, including receipts and readbacks.
The previous prefix app v4 and its consumed candidate remain recovery history;
do not replay either candidate. Reuse authenticated external model bytes and
source-built fixtures rather than uploading or rebuilding unchanged inputs.

This completes the native selected Q4-layer milestone, not the broader M2 gate
or inference. M2 still needs the dense floating-point whole-layer fixture and
representative extreme inputs. The full-decoder milestone above now establishes
three actual decoder commits; this older selected-layer sequence alone did not.

## Full-decoder numerical milestone (offline)

Production canonical arithmetic now passes the complete three-request portable
numerical gate: positions 0 and 1, then position 0 after actual KV clear/reset.
All 72 captured outputs (22 layers, final normalization and 32,000 logits per
request) match the authenticated CPU reference bit for bit. The real decoder
commits all positions; 2,394 active nodes and 2,199 interpreted launches complete
with zero CPU backend tensor arithmetic and balanced ownership. Same-input
checks pass all 465 quantizations and 465 projections without injecting CPU
activations. Acceptance tolerances remain unchanged.

The fix gives full mode a serial FP64 sum of F32 squares for RMS, eight fixed
FMA accumulators for Q4 projection, and 32 fixed FMA accumulators for F16 dot
products. The selected-layer normalization route remains unchanged. The fresh
13-kernel GPU fixture passes Initial admission, keeps exactly 16 byte loads per
Q4 block and uses no LDS, private memory or spills. Host interpretation does
not prove native GPU arithmetic, performance or full-model inference.

See `evidence/offline-full-decoder-canonical-arithmetic.json` and
`docs/LLM_FULL_NUMERICS.md`. Reuse `results/m3-full-reference-v1`,
`results/m3-full-numeric-canonical-v1` and `results/llm-layer-fixture-v6`.
Historical NUMERICAL_FAIL receipts remain unchanged.

The native helper has smoke (3 requests/72 files) and long (514 requests/168
files) profiles. Smoke now passes natively as recorded above. Measure phase and
request timings before attempting the long sequence. The long
profile already passed mock lifecycle testing with synthetic outputs; that
result does not establish long-profile numerical accuracy. Native smoke image,
frozen candidate and .40 numerical/lifecycle acceptance now pass.

## Native full smoke preparation

The `model-full` phase now packages the six authenticated modules and selects
three real decoder requests (0/1/reset0), with 72 raw outputs and unchanged
numerical gates. Full image audit, copied-reference freezer and controller
routes have focused positive/negative tests and independent review. The
extended auditor also accepts the preserved selected-layer image. This is
source/control preparation, not acceptance of a newly linked full image.
See `evidence/offline-native-full-smoke-contracts.json`.

Reuse full assets `results/m3-full-assets-v2`, vocabulary v3, native client v7
and CPU reference v3. Only the stale small vocabulary fixture was refreshed.
The native package now passes its final receipt and 393-function image audit:
149 objects, 8,212 authenticated inputs, 15 files and 22,478,848 package bytes.
See `evidence/offline-native-full-decoder-package.json`. Preserve
`build/llm-native-full-smoke-app-v1`. The consumed full candidate failed during model loading as described above.
The v2 milestone above supersedes that failure and establishes native smoke
all-layer/logit acceptance. The v1 command plan remains historical only.

## Full-decoder control flow (offline only)

Full mode now configures llama's own backend instances, explicitly offloads
embedding weights and forwards the original scheduler graph. Both whole-graph
and single-node callback modes pass three successful mock-HIP decodes at
positions 0, 1 and reset 0: 2,394 active nodes, 2,199 launches, real KV commits,
no CPU tensor arithmetic and balanced 16/16 allocations and 1/1 instances.
Exact graph/spans, ordered fragments and live controls are checked. A dedicated
64 KiB workspace handles in-place softmax, and copied controls may be reused
only after their last consumer. Existing layer/prefix regressions still pass.
See `evidence/offline-full-decoder-integration.json` and
`results/m3-full-backend-v11`, `results/m3-full-dispatch-v7`,
`results/m3-layer-regression-v4`.

Portable numerical integration passes as described above. The first native
full-model candidate failed during model loading; corrected v2 now passes. See
the current accepted baseline and preserved failure identities above. Preserve the earlier selected-layer package for recovery.

## Full-model numerical reference

All 514 original CPU decoder requests now have a boundary reference: all 22
layer residuals, final weighted normalization and all 32,000 logits at seven
checkpoints. The 168 raw files total 2,214,912 bytes; reset positions 0 and 1
match all 24 boundaries bitwise. See
`evidence/offline-full-model-cpu-reference.json`. Reuse authenticated
`results/m3-full-reference-v1`; no native package was rebuilt for this gate.
Native full GPU smoke integration now passes as recorded above; the long
CPU reference does not by itself qualify long native execution.

## Full-model integration prerequisite

The original CPU decoder now has an authenticated 514-request/reset trace.
All seven complete graphs pass strict 22-block metadata/topology admission
and ten mutation classes under sanitizers. The 801-node graph contains three
inactive mixed-input nodes; their reused bytes must not drive GPU operations.
See `docs/LLM_FULL_MODEL.md` and `evidence/offline-full-model-graph-admission.json`.
Reuse `results/m3-full-graph-trace-v4` after authentication;
`results/m3-full-graph-admission-v3` records the host gate. Full GPU scheduler
instance configuration, ordered split execution and KV progression now pass
the offline mock gate above. Full numerical and native validation remain open.

## Layer arithmetic follow-up

Host extreme-input tests found and fixed an invalid integer conversion for
finite activations whose Q8 scale underflows to half zero. Such blocks now
become canonical zeros before reciprocal calculation; ordinary-input bytes
remain unchanged. Sanitizers and fresh layer fixture v5 ISA/admission pass.
See `evidence/offline-layer-extreme-inputs.json`. This is not a new native
qualification: the installed app remains the accepted `f251366` build.
Use `results/llm-layer-fixture-v5` in a refreshed aggregate/candidate. Dense
whole-layer, near-zero RMS and native extreme-input validation remain open.

## Current scope

The first release is standalone TinyLlama 1.1B Q4_0 GPU inference, batch one,
with a 512-token context. Follow [the release plan](docs/LLM_V1_PLAN.md).
Training, framework ports and complete ROCm library coverage are deferred;
new runtime work must unblock this client or its release validation.

As requested on 2026-10-14, ProsperoAI integration is removed. G7 is a standalone
inference client and SDK preview. All other compute, library, framework and training
gates remain open as previously recorded; historical evidence is unchanged.

## Plan review and resumption

M0's bounded HIP-call qualification passed on .40 as native title PPSA99014.
All eleven exact counter rows matched: ten successful sites and the expected
zero cleanup-free row. The guarded 16-byte fill preserved all 48 surrounding
bytes; copies, full clear, synchronization and 1/1 allocation/queue cleanup
passed. The app exited normally, mounts matched, all eight installed assets
were checked twice, services stayed healthy and both owned locks were released.
See `evidence/console40-native-hip-call-success.json`.

Together with scheduler-v10's full-model/context/rejection/reuse result and the
fresh host failure cases, this closes the bounded M0 client-boundary gate.
Cleanup-failure paths remain host-tested only. M1 remains complete. Neither
result demonstrates model GPU tensor computation or inference.

The actual-model embedding/normalization/QKV prefix now passes native GPU
validation on .40 as PPSA99014, using app v4 at source `50df1fd` and frozen
candidate `results/source-public-llm-prefix-console-v3`. Tokens 0, 1, 2 and
31999 each ran at position zero through the real model graph/backend: embedding
lookup, RMSNorm, learned normalization weights and full Q/K/V projections.
All 24 raw output tensors passed numerical validation across 36 GPU launches.
The model was reused from `/data/hip/models`, and the native app completed
normal cleanup. See `evidence/console40-model-prefix-success.json` for final
acceptance and exact identities; `docs/LLM_MODEL_PREFIX.md` documents the scope.

This prefix result remains valid historical evidence. The selected complete
Q4 layer now passes the separate native milestone above; the full original
graph still rejects unsupported work. M2's broader numerical fixtures, M3's
model-scale forward path and native original-format readers remain open.

Preserve app v4, its frozen candidate, deployment/recovery receipts and raw
readbacks. Both candidate phases are consumed; do not replay them. Prepare a
fresh candidate for the next native milestone. The model does not need another
upload while its authenticated bytes remain unchanged.

Earlier prefix evidence remains valid history: four real graphs and 36 mocked
HIP dispatches passed offline; 361 Python regression methods and sanitized
GET_ROWS/Q8/Q4 admission tests passed. See
`evidence/offline-model-prefix-integration.json`. App v3 / frozen candidate v2
then loaded the model/context but rejected the first embedding launch with
`hipErrorInvalidValue` before GPU submission, quarantined and completed a
healthy controlled close. That run validated no model tensor output; see
`evidence/console40-model-prefix-launch-rejection.json`. The narrow four-row
GET_ROWS admission fix and four resource tests are recorded in
`evidence/offline-public-getrows-admission.json`. Preserve the earlier recovery
artifacts and v1 prepared sources used by v3; never replay consumed candidate v2.

Build-performance work now uses relative generated source/include paths from
each fresh prepared tree, allowing unchanged runtime units to hit ccache even
when another unit changes. A real unchanged runtime TU hit across v1/v4 trees
and matched uncached object bytes; isolated source/header/optional-header changes
invalidated correctly. Default builds no longer depend on automatic prior-tree
hints; the explicit prepared-source option remains available. Dependency scans,
absolute input identities and before/after checks remain enforced. Historical
123-test path-resolution/automatic-tree work remains recorded in
`evidence/offline-build-overhead-reduction.json`; automatic selection is now
superseded by per-unit reuse. One warm build took 1637.75 seconds with 79 hits
and two misses. Freeze dedup reduced a measured run from 1299.29 to 942.888
seconds (27.43%); neither result establishes whole-build savings from relative
paths. The applied cache change passes 127 focused tests, including real SDK
cache integration; see `evidence/offline-relative-compiler-cache.json` and
`docs/SOURCE_ONLY_REBUILD.md` for controls and limits.

A new prefix-only candidate would need regenerated prefix reference/aggregate
receipts after the builder changes. The active layer candidate has its own fresh
reference and aggregate; it reuses the CPU libraries, native client, SDK and model.
The accepted v4 package predates the relative-path cache change; a complete
package using the new paths has not yet been built or timed.

Disk cleanup removed 3167 superseded native/experiment object files (1.06 GiB).
Receipts, sources, models, logs and recovery packages were retained; old
receipts referencing pruned objects require a rebuild for full reauthentication.
The active call candidate's inputs were excluded. Local deletion ledgers are
under `results/m0-hip-call-console-preparation-v1`.

The owner's additional source-format requirement is direct native loading of
safetensors and PyTorch checkpoints, not desktop conversion. The stages and
acceptance criteria are in `docs/LLM_V1_PLAN.md`; native readers are unimplemented.
The desktop prototype was removed from active targets and retained locally.

External model loading is now validated on .40 with native model-app-v8
(source `08857a6`). The 21 MiB package contains `model-path.txt`, not weights;
the existing Eden/Lapy helper grants access to
`/data/hip/models/tinyllama-q4_0.gguf`. Deployment uploaded the locked
619,831,552-byte model once and verified its SHA-256; execution verified it
again without uploading. The app loaded the full model, created the 512-token
context, completed both expected graph rejections/reuse and balanced 219/219
owned allocations. The largest free extent returned to its pre-load value.

The same context heap warning still rejects strict M0 acceptance. Finalizers,
shell-exit request and ordered title stop completed; services remained healthy
and both locks were released. Post-run mount/asset checks were skipped after
transcript rejection. This validates external loading, not inference or a clean
M0 pass. See `evidence/console40-native-external-model-load.json`.
Candidate `results/source-public-llm-external-console-v1` is consumed for both
deploy and execute. Preserve its recovery files and installed v8 package.
Next investigate the scheduler malloc warning offline; do not replay this run.

Previous checkpoint: model-app-v7 loaded the full locked TinyLlama
model on .40 and created its 512-token context. Both mandatory graph rejections
and second-call reuse completed, with 219 owned allocations/releases, zero
live owned bytes/failures, and a 663,545,360-byte requested peak. The measured
direct pool was 12 GiB; its largest free extent returned from 12,214,861,824
bytes after context creation to the original 12,882,804,736 bytes after cleanup.

Overall execution acceptance remains false: context creation emitted one
`[ScePthread/System] Internal Memory is running out.` warning. The strict parser
rejected it; post-execution mount/asset checks consequently did not run.
Application finalizers and its shell-exit request completed, the title stopped,
services remained healthy, and both locks were released. This is validated
model-loading/allocator progress, not a clean M0 pass or GPU inference.
See `evidence/console40-native-model-load-partial.json`.

Next trace the scheduler's unchecked ordinary `malloc(context_buffer_size)`:
its null result can cause `ggml_init` to obtain owned aligned backing instead.
This is a source-supported explanation for the warning, not proven attribution.
Do not whitelist the warning or replay consumed model-console-v3. A correction
needs a fresh candidate. Per-HIP-call accounting and M2-M5 remain open; M1 is complete.
The older checkpoints below are historical, not the current next action.

Model-app-v5 linked and packaged, but Make failed its final audit: the earlier
external-only wrapping assumption was wrong. lld also redirects same-unit GGML
context allocation/free calls; the original aligned allocator symbols are removed.
All selected aligned pairs (contexts, CPU buffers, threadpools) preserve matching
sizes. The corrected auditor passes 17 tests and a 45-function three-image
diagnostic on retained v5 images, including the indirect threadpool allocation.
This diagnostic does not create a successful v5 receipt or deployment candidate.
The runtime log now accurately says `scope=ggml_aligned`; ordinary libc allocation
and the HIP limit remain unchanged. Native capacity and model loading are unproven.
Next build fresh model-app-v6, independently admit it, then freeze and review a
new installed-app candidate for idle .40. Preserve consumed v2 and installed v4
recovery evidence. M0 call accounting remains open; M1 remains complete.

The corrected native model run failed during model loading. Startup and the
watchdog guard passed, then `posix_memalign` returned ENOMEM for the
619,094,016-byte CPU model buffer. The app quarantined; the controller's bounded
exact-title close restored idle, healthy services and released both locks.
Candidate model-console-v2 execution is consumed: do not replay it.
Model loading, graph rejection/reuse and natural finalization remain unproven.
Next diagnose the native allocator route and implement/test an owned host-storage
correction offline before a fresh package. This is not evidence of insufficient
total console RAM. M0 remains open; M1 remains complete.
See `evidence/console40-native-model-allocation-failure.json`.

The watchdog-corrected model-app-v4 is installed and independently verified on
.40: 22 readbacks, two transfer records, nine mounted assets, preserved failed
model package/log, idle snapshots, healthy services and both locks released.
See `evidence/console40-native-model-watchdog-fixed-deployment.json`.
Candidate `results/source-public-llm-model-console-v2` deployment is consumed;
its approval now binds one bounded execution to the exact deployment receipt.
Next run the installed PPSA99014 model/reuse test. No app launch or native model
loading is claimed by this deployment milestone; M0 remains open.

The watchdog-corrected model-app-v4 passes Make and independent real freeze
admission: 148 objects, 199 commands, nine assets and the native call-chain audit.
The 827,719,680-byte package is bound in
`evidence/offline-native-llm-model-watchdog-fixed-package.json`; 16 focused suites
(145 tests) pass. Next freeze `results/source-public-llm-model-console-v2`,
independently review it, then deploy and run the installed PPSA99014 on idle .40.
The corrected watchdog and actual model loading/rejection/reuse remain untested
on console. M0 still needs per-HIP-call accounting; M1 remains complete.

Historical model recovery admission now verifies the previous frozen candidate,
accepted deployment/review bindings, exact package/nine assets, locked model and
archived audited images without requiring old source to equal today's checkout.
Strict current-source validation for new candidates is unchanged. Forty-one
focused tests, actual installed-baseline artifact validation and independent
review pass; see `evidence/offline-historical-model-baseline.json`.
The next baseline receipt is
`results/source-public-llm-model-console-v1/build-receipt.json`; preserve its
192-byte failed log and the older successful vocabulary backups. Prepared case
and freeze arguments are in `results/m0-model-console-case-v2/`; run them only
after the current model-app-v4 build and independent admission pass.


The watchdog contract correction passes independent offline review: the shared
maximum now admits the intended 120-second model budget; all existing shorter
budgets and timeout/state/join checks remain intact. The production-code test
passes 33 boundary/lifecycle cases, 30 related test methods and registered CTest.
Make runs the regression before native assembly and binds its source in receipts.
See `evidence/offline-native-llm-watchdog-fix.json`. Next build fresh
`build/llm-native-model-app-v4` via `.local/build_native_model_app_v4.py`, then
validate and freeze a new candidate. The correction is not yet native-tested.


The first installed full-model run failed before HIP/model setup: the title
requested a 120,000 ms watchdog, while the shared watchdog rejected values above
60,000 ms. Its exact `watchdog-state` quarantine followed successful startup
and runtime checks. The 150-second controller deadline led to exact-title safety
close; .40 returned idle with healthy services and both locks released. See
`evidence/console40-native-model-watchdog-failure.json`. Candidate v1 execution
is consumed; do not replay it. Correct the watchdog contract and test boundaries
offline, then build and review a fresh native candidate. No model load, graph
reuse, natural finalization or post-run mount equality was established.


The full-model candidate is now installed and independently verified on .40.
All nine mounted assets, the 827,719,680-byte package, preserved vocabulary
package/log, idle snapshots, healthy services and both released locks pass.
See `evidence/console40-native-model-deployment.json`. No app launch occurred.
Candidate `results/source-public-llm-model-console-v1` deployment is consumed;
its reviewed receipt permits one bounded native execution. Do not redeploy it.
The next gate is real model loading, two-call rejection/reuse and clean shutdown.
M0 remains open, including per-call instrumentation; inference is unimplemented.


The full-model native app now passes Make and independent real freeze admission:
148 objects, 7,005 inputs, 2,403 artifacts, all 199 commands and the 29-function
three-image call-chain audit. The package is 827,719,680 bytes. Fifteen control
and build suites (139 tests) pass. Model package controls now share a 1 GiB
ceiling accounting for UFS metadata/free space; exact bytes, hashes and all nine
assets remain mandatory. See `evidence/offline-native-llm-model-gate.json`.
Next freeze and review a fresh candidate, then validate the installed PPSA99014
full-model load and initial/reused graph rejection on idle .40. M0 remains open;
this offline milestone proves neither native model loading nor inference.

Earlier model-app-v1 failed on the missing upstream optimizer cleanup object;
model-app-v2 was deliberately stopped for a proven audit source-path bug.
Both are preserved. The corrected model-app-v3 is complete and validated.
Current checkpoint: the corrected installed PPSA99014 vocabulary app passed on
.40, including 32 exact tokenizer cases (236 tokens), HIP copy/clear, synthetic
unsupported-compute rejection and natural shutdown. The terminal controller
accepted the run with healthy services and owned-lock release. See
`evidence/console40-native-vocab-success.json`. Candidate v3 is consumed; do not
replay it. Next validate full-model loading and actual-client graph rejection,
including reuse. M0 remains open; this test performed no model tensor computation.
The failure and offline-build paragraphs below describe earlier checkpoints.

The installed native vocabulary app was tested on .40. Deployment passed,
including directory creation, all eight asset readbacks and recovery preservation.
PPSA99014 reached native startup/runtime checks, then a generic backend invariant
failed before vocabulary completion. The watchdog quarantined it; exact-title
safety close restored idle, healthy services and both released locks. This is a
failed vocabulary gate, not M0 completion. Preserve consumed candidate
`results/source-public-llm-vocab-console-v2`; diagnose the first assertion offline
before rebuilding. See `evidence/console40-native-vocab-invariant-failure.json`.
Offline inspection found post-elevation `/app0` reads inconsistent with the
helper's changed root and proven title asset paths. Phase/vocabulary/model reads
now use `/system_ex/app/PPSA99014/`, with explicit failure diagnostics; 33 focused
tests and independent review pass. See `evidence/offline-native-llm-asset-path-fix.json`.
The fresh app-v4 build now passes all 197 commands, 43 focused tests and the
24-function image audit. Independent real freeze admission verified 147 objects,
6,988 inputs and 2,395 artifacts; package size is 19,922,944 bytes. See
`evidence/offline-native-llm-asset-path-rebuild.json`. Next freeze and review a new
candidate before another installed-title test. The prior transcript alone does
not identify which assertion failed; native correction remains unproven.
Full-model changes remain staged.

M0's source-only vocabulary application build and independent real candidate
validation pass. `make llm-native-app` produced a 19,890,176-byte native package
from 147 objects; all 197 commands, 41 focused Python tests, the 24-function
image audit and final input checks passed. Independent review accepted all
6,988 input identities, 2,395 artifacts and eight package assets. See
`evidence/offline-native-llm-application.json` and `docs/LLM_NATIVE_CLIENT.md`.
Vocabulary console controls also passed 38 legacy regression tests.
The first deployment stopped before upload because a configured results-directory
ancestor was missing. The baseline package matched; no application launched,
services remained healthy and both locks were released. The controller now
creates and verifies missing configured ancestors; independent review and 52
focused tests pass. See `evidence/console40-vocab-deploy-directory-failure.json`.
Prepare a fresh candidate with the repaired controller and validate the installed
PPSA99014 title on idle .40; never replay the consumed v1 deployment.
No installed application execution is claimed. Vocabulary loading
and synthetic rejection cannot close the actual full-model/reuse gate.

`make llm-native-client-link` now rebuilds the genuine loader/tokenizer payload
from source: 71 objects, 161 factory-referenced model implementations and four
focused checks pass. The production ELF matches the earlier prototype exactly.
See `docs/LLM_NATIVE_CLIENT.md`. This is a payload-link prerequisite only:
native application startup, backend routing and inference remain unqualified.
The application package now passes offline; installed-title validation is next,
with the bounded vocabulary fixture and actual-client boundary kept distinct.

`make llm-vocab-fixture` also passes production reproduction and independent
review: 45 exact metadata fields, zero tensors, and 32 tokenizer cases match
the original model. The 725,614-byte asset is a native loader-test prerequisite,
not a model graph or inference result. See `docs/LLM_VOCAB_FIXTURE.md`.

M0's executed CPU graph inventory is now reproducible with
`make llm-executed-graph`: all 512 positions, 410,112 completed nodes and
16,384,000 finite logits pass independent review. The 14-case test suite and
Make/CMake integration pass. See `evidence/offline-llm-executed-graph.json` and
`docs/LLM_EXECUTED_GRAPH.md` for the 14-operation coverage table. This is CPU
graph evidence only; native application/backend integration still remains.

Implementation resumed after the owner's plan review.
The updated release plan tightens M0 to a reproducible native application with
loading/tokenization and enforced backend routing. Local ignored prototypes
also provide a limited native payload link and a storage-only HIP backend;
none establishes native inference. The link review
found incomplete object/tool/header provenance; source-only promotion is in
progress. M1 passed on .40 with independent outcome review. Its five stages and 30 raw buffers
passed numerical checks, with natural shutdown, healthy services and both locks
released. Do not replay its consumed deploy/execute phases. These facts
supersede the older next-action paragraphs below.

See `evidence/console40-public-weighted-qkv-success.json`: all 2,048 weighted
values, 2,304 Q8 bytes and 2,560 Q/K/V values passed frozen comparisons. Maximum
projection error was 7.50296749e-8; maximum budget fraction was 0.0415416502.
This proves one fixed-input layer-zero QKV chain, not attention or a full layer.
Next finish M0 native application/backend integration, then M2 layer operators.

The owner resumed M0 and M1 in parallel, with .40 validation authorized. The
revised [LLM-first plan](docs/LLM_V1_PLAN.md) makes native client linkage and an
executed graph inventory immediate priorities alongside the current QKV gate.
Use a narrow ggml/public-HIP integration and sequential single-token GPU prefill
initially; then qualify a complete layer, model memory, full logits, generation
and release behavior. No full ROCm or general BLAS port is a prerequisite.
The native QKV package build completed; independent package review and the
frozen console gate are next. M0 graph tracing and native client linkage proceed
in parallel without changing the candidate inputs.

## Native weighted QKV package passed offline

The 76-object native package and 21-function three-image audit pass. Independent
review reauthenticated 1,405 inputs and independently extracted all 16 package
files, including the exact 10 module/reference assets. See
`evidence/offline-native-weighted-qkv-package.json`. The candidate is ready for
freeze review and a bounded .40 cycle; no weighted QKV GPU result is claimed.

M0 proceeds in parallel: executed CPU graph inventory and a real native client
compile/link investigation. Both work in ignored staging while M1 is frozen.

## Weighted QKV package and console controls ready offline

The five-stage native harness compiles and passes independent source review.
Package integration requires three authenticated modules, seven reference assets
and a matching three-image audit. All six registered QKV test groups pass;
independent package/controller review and full-size synthetic readback checks
also pass. See `evidence/offline-weighted-qkv-package-controls.json` and
`evidence/offline-weighted-qkv-readback-verifier.json`.

The controller accepts the transport transcript only after all 30 full buffers
pass numerical and continuity checks. It owns the primary workspace lock; the
second PS5-Claude workspace lock still requires separate operator coordination.
Fresh compiler fixtures are building. Actual native package/image validation,
frozen deployment and the .40 numerical run remain pending.

## Full weighted normalization and Q/K/V references qualified offline

The source-only `make llm-weighted-qkv-reference` target now builds missing CPU
prerequisites and authenticates existing model/reference inputs before reuse.
The learned layer-zero normalization and full Q2048/K256/V256 references repeat
bitwise; five focused tests and independent numerical reconstruction pass. See
`evidence/offline-weighted-qkv-reference.json`.

These are CPU references, not GPU results. Native validation must check actual
weighted GPU output against its frozen budget, then quantize that captured
output for the chained Q8/Q4 oracle. The native five-stage harness, package and
readback verification are the next gate; a complete transformer layer remains open.

## Weighted RMSNorm source and public HIP qualification passed offline

The supported Make targets now build the selected weighted RMSNorm module from
unchanged upstream source. Its complete ABI, compiler inputs and generated ISA
are authenticated. The fresh 88-compilation sanitizer qualification and both
weighted/original public HIP packet models pass in eager and deferred modes,
including intentional packet mismatch rejection. See
`evidence/offline-public-weighted-rmsnorm.json`.

Invalid nonnull kernel buffer pointers now reject before enqueue through the
shared public launch validator. A stalled-first-dispatch regression proves this
check does not block a second asynchronous enqueue. This is point-in-time pointer
admission, not concurrent-free protection or general asynchronous error reporting;
the latter remains open.

Next qualify weighted RMSNorm -> shared Q8 -> full Q/K/V numerically on .40.
The CPU reference producer has passed independent review locally; its native
harness and build integration are still in progress. No weighted GPU result,
complete transformer layer or full-model inference is claimed.

## Halfword Q4 diagnostic passed on .40

The frozen candidate from `0dc45da` passed all 28 native stages: 592 Q4 values,
80,640 exact Q8 bytes and 112 raw buffers (9,097,984 bytes). This includes
model-width Q8 and chained Q4 consuming actual device Q8 output. Frozen budgets,
inputs, weights, guards and inactive regions passed; maximum absolute error was
7.11181201e-7 and maximum budget fraction 0.0166699232. Natural finalization,
matched idle snapshots, installed identity, healthy services and both lock
releases passed. See `evidence/console40-public-q4-halfword-success.json`.

The candidate `results/source-public-q4-halfword-console-v1` is consumed: do not
replay deployment or execution. The original failed package and logs remain
preserved. The halfword variant also changes cache flags and scheduling; this
qualifies a bounded workaround, not isolated alignment causality or full-model
inference. Next qualify learned-weight RMSNorm and full Q/K/V projections,
then the remaining layer operators, model-scale memory and complete client.

## Halfword Q4 diagnostic built and independently reviewed offline

The authenticated halfword-load variant now passes genuine public HIP host
qualification and its 76-object native package/image audit. The original Q4
public model remains passing. Independent review verified all 64 package files,
58 Q4 assets and 2,951 identities. All 56 numerical assets/budgets remain identical
to the original reference; no tolerance was changed. See
`evidence/offline-native-q4-halfword-package.json`.

Package SHA256: `bb8be0ff0eda5dfff2cd7ae6388aba95b8eeafbfd30d2047e7932c7ac5d90471`.
Build: `build/public-hip-q4-halfword-native-v1`; fixture/reference:
`results/llm-q4-halfword-fixture-v1`, `results/llm-q4-halfword-reference-v1`.
Use separate results directory `/data/ps5-hip/results-q4-halfword-v1`.
Fresh-directory controls pass 48 tests; candidate/baseline directories are
validated separately and deployment creates only a missing leaf directory.
Deployment and execution are now consumed; the outcome above supersedes the
offline next action. This variant changes cache flags and scheduling
as well as load width; success would not alone prove alignment as the cause.

## Q4 diagnostic resource admission tested offline

The core parser and packet encoder now admit exactly c3/c4 for the existing
Q4 ABI. Real original and halfword-load objects pass packet and owned native
preparation tests; all other 62 VGPR allocation fields reject. The optional
diagnostic object must be c4. Of 62 selected checks, 61 passed; the historical
reduction compiler check rejected its configured old observation-only receipt.
See `evidence/offline-q4-c4-admission.json`; no public CLR qualification or
new hardware run is claimed. Next prepare a fresh authenticated diagnostic.

## Q4 alignment hypothesis narrowed offline

The unchanged lane model reproduces the failed Q4 output bit-for-bit when
only nibble-load addresses are rounded down to four-byte alignment. A separate
compiler experiment preserves the original ABI and emits halfword payload
loads when four i16 reads are volatile, but changes register usage/cache flags.
This is not a hardware fix or a passed numerical gate. See
`evidence/offline-q4-alignment-diagnosis.json` and
`tools/probe_llm_q4_alignment.py`. Next review a fresh bounded alignment
diagnostic; original candidate and tolerances remain unchanged.

## Q4 console validation failed; analyze offline before another candidate

The frozen Q4 candidate deployed successfully on .40, including 78 verified
readbacks and preserved RMSNorm baseline. Its one execution passed the four
standalone width256 Q8 cases, then failed the first `attn-k-1` frozen-Q8 Q4
comparison: actual `0.1636260598897934` (`0x3e278d97`) versus FP64 reference
`0.21718877422972772`, budget `8.8989529262781016e-06`. This is a numerical
failure, not a passed Q4 operator gate. Model-width Q8 and chained Q4 were not
reached. Do not change the tolerance to accept the result.

The harness quarantined on the mismatch and later hit its watchdog. The
controller timed out, safety-closed the exact title and confirmed idle state,
healthy services and both owned locks released. No native fault marker was
found in captured klog. Twenty raw buffers were retained; the other 92 files
were not produced. Candidate `results/source-public-q4-console-v1` deploy and
execute phases are consumed; never replay them.

The installed package is now
`a2b781d9de42725da12c0310fc2d44226a01a287a6af1a24aef318ebabfcd5d9`.
The prior RMSNorm image/log are preserved. Independent outcome review confirmed
the failure and clean safety closure; see `evidence/console40-public-q4-failure.json`.
Next inspect raw operands and exact selected ISA/dispatch, and define a
narrow diagnostic from evidence. No automatic rerun or new console action is
authorized by a passing offline reference. Full-model inference remains open.

## Native Q8/Q4 numerical package built offline

The 76-object native Q4 title now builds and packages successfully. The package
contains 64 files, including both source-built modules and 56 authenticated
reference assets. The three-image audit passes startup, actual imports,
finalizer/watchdog ordering and preservation of 18 critical functions. Twelve
focused and existing-control test suites pass 86 tests.

Package SHA256: `a2b781d9de42725da12c0310fc2d44226a01a287a6af1a24aef318ebabfcd5d9`.
Build: `build/public-hip-q4-native-v1/link-receipt.json`; audit:
`build/public-hip-q4-native-v1/q4-native-audit.json`. Reference v4 retains the
same 56 numerical assets as v2/v3. The harness requires 28 launches, 112 full
readbacks, 592 Q4 comparisons and 80,640 exact Q8 bytes. Chained multiplication
must consume the actual preceding Q8 output without reuploading it.

Q4 freezing, pre-contact authentication, freshness, bounded raw collection and
per-stage log/raw checks are implemented. Independent package review passes,
including a separate extraction of all 64 files. See
`evidence/offline-native-public-q4-package.json`. Next freeze the committed
candidate, review its manifest and execute one bounded
idle .40 case under both owned locks. No Q4 GPU arithmetic or full-model
inference is established. The installed RMSNorm baseline remains unchanged.

## Q4 dispatch and numerical reference: offline qualification

The genuine public HIP runtime now admits the measured Q8_1/Q4_0 contracts:
256x1x1 with 328-byte hidden-geometry arguments for Q8, and 32x4x1 with
384-byte static LDS and 192-byte explicit-only arguments for Q4. The public
host model passes four dispatches, 30 negative controls, eager/deferred loading
and intentional packet-mismatch detection. The shared 87-compilation suite,
RMSNorm regression, 32 core tests and native aggregate cross-build pass.

The independent GPU-format numerical reference builder also passes: eight Q4
cases / 296 outputs, four standalone Q8 cases and 16 corruption controls.
Actual model weights and the original authenticated RMSNorm activation are
retained; width5632 activations are synthetic. These are offline references,
not GPU arithmetic or full-model inference evidence.

Next qualify the native numerical harness/package and independent raw verifier,
then freeze and validate isolated Q4 and chained Q8/Q4 execution on .40.
The draft harness uses 28 launches and retains all four buffers at every stage.
Do not deploy a draft or reuse the consumed RMSNorm candidate. No new console
run occurred; the installed RMSNorm baseline below remains current. See
`docs/HIP_QUANTIZED.md`, `docs/LLM_Q4_REFERENCE.md` and the compact offline
milestone evidence. Remaining operators, model-scale allocations, client link,
full inference and release qualification remain open.

## Native RMSNorm numerical run completed

The six-case upstream RMSNorm run on .40 completed successfully: 81,920
finite values passed the frozen CPU tolerance, with all input, guard and
inactive bytes preserved. The title naturally exited, services remained healthy,
and both owned locks were released. Independent outcome review passed. See
`evidence/console40-public-rmsnorm-success.json`.
Candidate: `results/source-public-rmsnorm-console-v1`; deploy and execute are
consumed. Do not repeat them. The installed package is now
`9daf89bf558e55c5fc0be0e8abe344162eafdaa02d50548a433ff78711e93877`,
and the application log is
`7c21709884e07e95c77edd596711329dd3899a623edc5d806d676bd407f70f2d`.
The prior aggregate image/log are preserved. This supersedes older installed
baseline and next-experiment paragraphs below.

Next: complete the client operator/call
inventory and address the first unresolved Q4_0 multiplication or native client
link blocker. Integrate one complete transformer layer before full-model
prefill/decode. Model-scale allocations, remaining operators, full inference
and release lifecycle/performance qualification remain open.

## Active priority: native LLM inference v1

Owner decision on 2026-10-15: prioritize the bounded first release in
`docs/LLM_V1_PLAN.md`. Its release checklist requires source-only builds, real
GPU prefill/decode, reference correctness, lifecycle qualification and measured
performance; full ROCm compatibility and training are not v1 prerequisites.
Start V1.1 client/model profiling, then client-required
public HIP launch/memory coverage, tensor operations and end-to-end inference.
The current upstream inventory baseline is llama.cpp master
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`. CPU reference tools now build and
pass upstream quantization and RoPE tests. The native client and model run
remain unproven. See `docs/LLM_V1_PROFILE.md` and
`evidence/offline-llm-v1-inputs.json` for this partial V1.1 checkpoint.

Optional Make targets fetch the client, build CPU tools and reproduce the
locked TinyLlama metadata/header profile. `make llm-model-fetch` now downloads
and verifies full source weights/tokenizer; see `evidence/offline-llm-model-source.json`.
F16 and pure Q4_0 conversion now pass with locked upstream tools. The Q4_0
GGUF is 619831552 bytes, with 156 Q4_0 and 45 F32 tensors; see
`evidence/offline-llm-model-conversion.json`. CPU prefill/decode references now
pass seven fixed cases twice, including 64 decode steps and the 512-token
boundary, with identical raw logits across repeats. See
`evidence/offline-llm-cpu-reference.json`. Native model memory usage and GPU
numerical tolerances remain open. Native compiler-generated embedded-image
launch and 12/16/128-byte aggregate arguments now pass bounded .40 cases below.
Next: real operator launch resources and math/library integration, using the pinned
converted weights, real compiler probe and CPU token/logit fixtures.

The bounded effective compiler probe now passes twelve host-object builds and
links backend/getrows/mmvq device modules. A controlled library-order fix now
also links norm/RoPE: OCML must precede its OCLC constants. Cooperative softmax
linkage remains unresolved, along with header-version alignment and actual
operator launch resource requirements. See `docs/LLM_V1_COMPILE_PROBE.md` and
`evidence/offline-llm-hip-compile-probe.json`. No final client link or native
operator execution is proven; diagnostic historical headers are not new locks.

Defer new retirement-ack diagnostics and general G5 optimization unless a real
client blocker requires them. Existing correctness tests remain regressions.
The full SDK roadmap stays open; v1 does not require all G5-G12 work to finish.
Historical next-step paragraphs below are superseded by this release priority.
The planning milestone itself made no console change. The subsequent native
compiler-launch checkpoint below records the current installed image/log.

## Unchanged RMSNorm module now builds from source only

`make llm-rmsnorm-fixture` builds authenticated upstream device libraries and
unchanged llama.cpp `norm.cu`, without historical `.local` outputs. The fresh
Make run and independent source-only build produce the same 60,544-byte,
16-kernel module as the original compiler probe: SHA256
`504ed4ae3e8c65f5bee7dc2f09ef1ed227235c93deab84b655e69ada654405b5`.
Full 26-explicit/14-hidden argument and descriptor checks, zero undefined
symbols, two focused tests, CTest registration and independent review pass.
See `evidence/offline-rmsnorm-source-fixture.json` and `docs/LLM_V1_PROFILE.md`.
Root Make session 42388 terminated successfully. Use a fresh output per run.

## Native RMSNorm numerical package qualified offline

The 76-object native numerical title now builds with the unchanged module and
twelve authenticated CPU input/reference tensors. Independent package extraction
matches all 19 files, including the 13 RMSNorm assets. Startup, actual imports,
finalizer registration and 15 critical functions pass the three-image audit.
Nine focused CTests, 31 existing control tests and independent reviews pass.
See `evidence/offline-native-public-rmsnorm-package.json` and `docs/HIP_REDUCTION.md`.

The candidate checks six cases/81,920 values and retains twelve full raw buffers.
Freeze controls require the image audit, frozen CPU oracle and fresh output names;
acceptance also requires independent raw numerical/guard checks and matching
transcript metrics. Synthetic raw-verifier tests do not establish GPU execution.

Native build session 59484 terminated with exit 0. Build receipt and image audit
are in `build/public-hip-rmsnorm-native-v1/`. Package SHA256:
`9daf89bf558e55c5fc0be0e8abe344162eafdaa02d50548a433ff78711e93877`.
Next: commit/freeze this candidate, obtain independent manifest review, then run
one bounded .40 experiment under both owned locks. No console execution is
claimed in this checkpoint; the aggregate package remains the installed baseline.

## Public CLR RMSNorm resources qualified offline

The genuine public HIP runtime now loads the unchanged mixed norm module and
admits its supported RMSNorm kernel. Eager/deferred host runs verify rows 1/2/17,
1024 threads, 128 dynamic LDS bytes, all 488 kernarg bytes, optional null pointers,
caller mutation capture, exact code backing/entry and clean ownership. Invalid
geometry/shared sizes reject synchronously; the deliberate packet mismatch fails.
See `evidence/offline-public-clr-rmsnorm-resources.json` and `docs/HIP_REDUCTION.md`.

The full 87-compilation CLR sanitizer suite, three focused CTests, independent
reviews and 77-object native aggregate cross-build pass. Root sessions 95483,
5779 and 36938 all terminated with exit 0. Local receipts are respectively
`results/source-public-rmsnorm-v1/receipt.json`,
`results/public-rmsnorm-model-v1/receipt.json` and
`build/public-hip-rmsnorm-runtime-native-v1/link-receipt.json`.

Next: package the native numerical RMSNorm harness with authenticated module and
CPU fixtures, retain full readbacks, audit and freeze the candidate, then validate
on idle .40. The harness draft is not yet packaged or qualified. No native
RMSNorm math or end-to-end inference is claimed; the installed console baseline
remains the aggregate checkpoint.

## Real RMSNorm resource preparation qualified offline

The unchanged 16-kernel norm module now passes the bounded ReductionLds core
route for its unfused 2048-column RMSNorm kernel: 1024 threads, 26 explicit
arguments, 488 kernarg bytes, hidden dynamic size at offset 352, and 128 requested
LDS bytes rounded to 1024 allocated bytes. Independent packet checks verify
rsrc1/rsrc2 0x60af00c3/0x1038c and resource limits 0x00400000 for rows 1/2/17.
Host-mapped native preparation verifies real pointer capture, aggregate ownership,
null optional pointers, temporary cleanup and rejection before allocation.

The 32 core sanitizer regressions, three actual-object CTests, direct four-kernel
profile and fresh PS5 core-library cross-build pass. Source review passes with
an explicit generic-HIP caller contract: dynamic size is supplied by the caller;
metadata cannot prove the algorithmic minimum. Legacy reduction still requests
4096 bytes; only the exact measured invocations have corresponding evidence.
See `evidence/offline-rmsnorm-resource-preparation.json` and `docs/HIP_REDUCTION.md`.
Root processes 99096 (core tests) and 19744 (native cross-build) terminated with 0.

Source-only fixture generation now passes above. Next: qualify public CLR
mixed-module admission and nonzero shared-memory routing before a genuine
numerical GPU candidate. Preserve failed generator v1 evidence; corrected v2
and the fresh Make run passed the complete hidden-argument/descriptor checks.
No console change; native RMSNorm numerical accuracy and full inference remain open.

## Nullable kernel pointers qualified offline

Optional null global-pointer values now survive public HIP capture and native
argument preparation as zero values with zero extent, without a fake allocation
or retained buffer. Nonnull live-extent checks, argument-storage validation and
the strict vector path remain enforced. This is required for unused RMSNorm
multiply/add inputs, not permission for kernels to dereference null.

The full genuine CLR host qualification passes 87 compilations with ASan/UBSan,
including a zero-count nullable launch followed by the original live-output
launch and negative controls. The 32 configured core regressions, direct HIP
profile checks and independent source review pass. See
`evidence/offline-public-nullable-arguments.json`. Root qualification session
76756 terminated successfully; `results/source-public-nullable-v1/receipt.json`
is its local receipt. No new console run or native numerical result.

ReductionLds preparation now passes above. Continue with public module admission
and shared-memory routing after reproducible fixture generation.
Use the frozen CPU oracle below; preserve existing rejection and ownership tests.

## CPU RMSNorm numerical oracle qualified offline

`make llm-rmsnorm-reference` now reproduces the genuine upstream CPU operator
reference. Six width-2048 cases cover stress inputs and real Q4_0 embedding rows
at 1/2/17 rows. All 81,920 outputs match bit-for-bit across two fresh processes;
maximum CPU-versus-FP64 absolute error is approximately 2.23e-6. A fresh Make run
matches the independently reviewed reference. The three oracle tests, CTest
registration, CPU prerequisite checks and artifact/input rehashes pass.
See `evidence/offline-llm-rmsnorm-reference.json` and `docs/LLM_V1_PROFILE.md`.

The proposed GPU tolerance is frozen, not hardware-qualified. Nullable optional-pointer qualification now passes above. Next: actual RMSNorm launch resources
(1024 threads, 128 dynamic LDS bytes), then validate the unchanged upstream
operator against these tensors on .40. No console change in this milestone;
aggregate-launch evidence remains the installed baseline.

## Native aggregate arguments passed on .40

Latest native checkpoint: two genuine compiler-stub GPU launches preserve
320 member bytes and 64 guard bytes with 12/16/128-byte arguments. One allocation
and queue retire, application finalization completes, and PID 249 exits naturally.
Installed identities, matching idle snapshots, service health and both lock
releases passed independent review. See `evidence/console40-public-aggregate-success.json`.

Current package: `54d880eeb81097f7a9caaacad926a424673a1103b6fb33250d46f7b318c494f7`.
Current aggregate log: `4090033df580cdd6e15da34b52ba34a4030327222120a41863124d4158b8c3b1`.
Both phases of `results/source-public-aggregate-console-v1` are consumed. The
previous embedded package/log are preserved. Future cases need fresh controls.

The native aggregate package now builds with 76 ordinary objects plus the
genuine compiler object. Image/layout/teardown audits, package verification
and 41 focused control tests pass. See
`evidence/offline-public-aggregate-native-package.json`. The subsequent console
checkpoint above validates this exact package.

Genuine HIP compiler stubs now pass 12/16/128-byte by-value arguments through
CLR in eager and deferred host runs. Caller mutation, compiler offsets, zero
argument gaps, guards, unsupported-call recovery and clean modeled ownership
pass; no-op and corrupt transports fail the output oracle. The full 87-unit
CLR sanitizer qualification and 32 configured core regressions pass.
See `docs/PUBLIC_AGGREGATE_ARGUMENTS.md` and
`evidence/offline-public-aggregate-arguments.json` for exact scope and provenance.
Nested pointers remain opaque and require caller-managed lifetime. Next: genuine
2048-column RMSNorm, including 1024-thread geometry, 128-byte dynamic LDS,
metadata-selected hidden fields and unused null optional pointers. Freeze a CPU
numerical oracle and tolerances before native validation; see
`evidence/llm-next-runtime-gate.json`. No real LLM operator/model run is proven.

## Static module failure handling qualified offline

Previous native checkpoint: genuine CLR compiler registration/stubs passed on .40
with two GPU launches, 146 values, 16 guards, one allocation/free and one retired
queue. Finalization, natural exit, installed readbacks, idle snapshots, service
health and both lock releases passed independent review. See
`evidence/console40-public-embedded-success.json`. This closes the bounded native
compiler-launch prerequisite, not V1.2 or LLM inference.

Previous package (preserved): `e3d38022543668343a029d14cc55c1506bd5e1bc026719d3a1bd39c599093359`.
Previous embedded log (preserved): `45c411ac7cc7203a726853146995cb5e88b463de20e5c467c803cb0f2c57d978`.
Both phases of `results/source-public-embedded-console-v1` are consumed. The
previous package/copy_gpu log are preserved; older installed-baseline paragraphs
below are historical. Future console cases require a fresh candidate.

The actual native compiler-launch candidate now builds and passes the
raw/converted/extracted image audit and packaging. See
`evidence/offline-public-embedded-native-package.json`: 76 ordinary objects plus
one genuine HIP compiler object; ten audited functions preserve their bytes,
with native execute-only text and verified registration/teardown ordering.
All 75 runtime translation units match the previously qualified bootstrap.
The subsequent native checkpoint above validates this candidate. This package
receipt itself remains offline evidence; native LLM inference remains open.

Superseded next step: bounded compiler embedded-image registration is now
qualified offline through genuine CLR, including eager/deferred, two-module,
reversed-order and missing-kernel-name controls. The full 87-compilation host
suite and 76-object native bootstrap package pass. See
`evidence/offline-public-embedded-images.json`. Next build and audit the native
compiler-launch title; no new console image or native embedded launch is proven.

The genuine CLR static loader now publishes only successfully extracted modules.
Failed extraction no longer becomes false success on retry. Function/variable
lookups reject null failed-module slots, and late registration logs extraction
failure instead of asserting. The focused sanitizer regression reproduces the
original bug and covers errors, allocation failure, retry and cleanup; the full
85-compilation public HIP sanitizer qualification passes on the final inputs.
See `evidence/offline-static-module-failure.json`.

This earlier failure-handling checkpoint alone did not enable compiler-generated
embedded images. The subsequent bounded host qualification above supersedes
that next step. No console run or new installed image.

## CLR command lifetime accounting qualified offline

The public source build now accounts both Command base constructors and actual
base destruction. Retained references and refused destruction remain live;
derived-constructor failures balance. Both constructors, two failure paths,
retained owners, four concurrent threads/256 objects and final zero ownership
passed the fresh 85-unit sanitizer qualification. The 74-object native package
and exact patch preparation passed independent review.
See `evidence/offline-clr-command-accounting.json`.

Deferred by the inference-v1 priority: diagnostic post-retirement acknowledgement
for warm replay command counts and detailed CPU preparation/submit/wait profiling.
Stream synchronization can return before the worker releases its final barrier
reference, so immediate count equality is not a valid plateau check. Do not
replace that boundary with sleeps or hold device locks across callbacks.
The counter destructor hook precedes Event-base cleanup; these counters alone
prove neither full resource retirement nor heap stability. G5 remains open.

No console was contacted. The installed backing-test image/log below remain
the baseline; use a fresh candidate for the next meaningful native validation.

## Native replay backing samples passed

The .40 run matched all 32 synchronized samples: 32768 live bytes in two
allocations after warmup, 128 allocations/releases in each execution path,
98304 accounted peak bytes and zero final live backing. The combined title
completed 84 GPU dispatches, verified 5840 additional values and 640 guards,
and retired all seven queues. Natural exit, installed files, idle mounts,
service health and both lock releases passed independent outcome review.
See `evidence/console40-replay-backing-success.json`.

Installed image: `639cc88438b720df630c90f10479ae191d98984c3bba14761e8b42989b305b3b`.
Current copy_gpu log: `1ed01138f133404876e234d8b26bf7ac1cb249816e4f170f9dfa1a1d5032596a`.
Both phases of `results/source-public-replay-backing-console-v1` are consumed;
the previous two-kernel image/log remain preserved.

Next: CLR command lifetime accounting and CPU preparation/submit/wait
measurements. These bounded samples cover NativeMemory backing only; setup,
parameter updates and resets are excluded from execution churn. Host heap,
command plateau, sustained reliability, true GPU time, GPU-dependent retirement
and measured graph efficiency remain open. G5 is incomplete.

## Replay backing samples qualified offline

Deployment checkpoint: the frozen backing-measurement package was installed and
read back on idle .40 after the full settling period. The previous two-kernel
image/log are preserved; services are healthy and both locks released. See
`evidence/console40-replay-backing-deployment.json`. No launch has occurred.
Next: independent receipt-bound execution review, then one bounded run of
`results/source-public-replay-backing-console-v1`. Native sampling is still pending.

The shared two-kernel fixture now checks 32 synchronized samples against one
post-warmup baseline. Full 84-unit sanitizer qualification measured two live
allocations/32768 bytes, 128 allocations/releases in each execution path,
98304 accounted peak bytes and zero final backing. Deliberately retained backing
fails the shared sample check. A fresh 74-object native package, independent
artifact audit and 27 transcript controls passed.
See `evidence/offline-replay-backing.json`.

Next: freeze/review `build/public-hip-replay-backing-v1`, then validate once on
idle .40 using the installed two-kernel image/log below as baseline. The live
backing check covers NativeMemory only. Churn excludes setup, parameter updates
and buffer resets; neither command/process-heap plateau nor speed improvement
is established. Native sampling and the broader G5 contract remain open.

## Native backing accounting qualified offline

`native_memory_stats()` now provides coherent cumulative allocation/release
counts, live page-rounded bytes and accounted peak bytes. Failed constructor
cleanup, unmap/release failures and unresolved GPU ownership remain counted.
Four concurrent owners over 128 operations, registry quarantine and submission
temporary cleanup passed sanitizer tests. All 63 host tests and the PS5 native
core build passed; a stale copy-transcript negative test was also corrected.
See `evidence/offline-native-backing-accounting.json`.

Next: sample these counters after synchronized warm replay, with a retained
backing negative control, then validate on .40. Command lifetime and host-overhead
measurements remain separate work. No console was contacted for this change;
the installed two-kernel image/log below remains the next deployment baseline.
These counters do not measure process RSS, host heap or instantaneous physical
high-water usage during concurrent kernel calls. G5 remains incomplete.

## Native two-kernel replay passed

The joined two-kernel workload passed on .40 over four warmup and sixteen
verification rounds, alternating eager/captured order. Its 5840 added values
and 640 guards matched; the combined title completed 84 GPU dispatches and
28 callbacks, freed four allocations and retired all seven queues. Natural
exit, installed files, idle mounts, service health and both lock releases were
verified. See `evidence/console40-multi-kernel-success.json`.

Installed image: `309b66fedf667fcac958b4d2821a1159d01eb22cb1014fe8ee6ae7841daf610c`.
Current copy_gpu log: `f70d127289c1c0f7d23d59177d758ce045f234c7dd68c5512deff2904a063bbf`.
Both phases of `results/source-public-multi-kernel-console-v1` are consumed.
The previous CPU lifetime image and log remain preserved. The two kernels write
independent regions; ordering is verified, not computed-data dependence or
concurrent GPU execution. These bounded replays do not establish a memory plateau.

Next: instrument successful native physical allocations/releases and command
lifetimes, then collect warm replay resource and host-overhead baselines before
optimization. GPU-dependent retirement, sustained reliability, measured replay
efficiency and the full G5 contract remain open.

## Two-kernel replay qualified offline

The shared fixture now captures two kernels across joined streams, checks its
exact six-node topology, alternates eager/captured order and verifies changing
inputs over four warmup and sixteen verification rounds. Each kernel writes a
separate guarded region; both independent omission controls reject. The combined
model verifies 84 dispatches, four allocations/frees and seven retired queues.
Full 84-unit sanitizer qualification, 26 controls and a fresh 74-object native
package passed. See `evidence/offline-two-kernel-replay.json`.

The package deployed and verified on idle .40; the CPU image/log are preserved,
services healthy and both locks released. See
`evidence/console40-multi-kernel-deployment.json` (relative to repository root).
Installed image: `309b66fedf667fcac958b4d2821a1159d01eb22cb1014fe8ee6ae7841daf610c`.
Next: receipt-bound review, then one bounded execution. Acceptance requires
the existing joined oracle plus 5840 added values, 640 guards and clean aggregate
retirement. The 30-second title watchdog and 40-second controller limit remain.
These are repeated correctness checks; native validation, memory plateau,
measured replay efficiency and the full G5 contract remain open.

## Native allocation-lifetime cases passed

The CPU-only copy title passed queued free with a live executable and parameter
migration followed by freeing old allocations before replay on .40. All 384
lifetime comparison bytes and 7168 rectangular-copy bytes matched; eight
allocations/frees and four retired queues were verified. The title exited
naturally, installed files and idle mounts matched, services remained healthy,
and both locks released. See `evidence/console40-copy-lifetime-success.json`.

Installed image: `dd5f92c5c117518958ec088da0ced54980904b61eec17302527344bd099227f7`.
Current copy log: `b0596a28d2ae370b309ce4be2b683c543d699e82d065206b9f8044524fbeef58`.
Both phases of `results/source-public-copy-lifetime-console-v1` are consumed.
The previous joined GPU image and log remain preserved. This CPU-worker test
does not prove GPU-completion-dependent retirement; its 100 ms observation also
cannot prove entry into the freeing API before gate release.

Next: qualify changing-input two-kernel replay, then measure actual native
allocation/command lifetime and submission overhead. HIP allocation counters
alone miss the per-dispatch command/marker allocations. Establish eager/captured
baselines before optimizing; sustained replay, memory plateau, performance and
the full G5 contract remain open.

## Shared native allocation-lifetime candidate

The CPU-only copy title now uses the same lifetime helper as the sanitizer model:
queued free while the executable remains alive, and parameter migration followed
by freeing old allocations before replay. Static callback storage, bounded waits,
checked pthread creation/join and exception quarantine preserve uncertain work.
Both injected pthread failures are rejected at the shared check. Full 84-unit
ASan/UBSan qualification, 26 controls and a fresh 74-object native package passed;
independent artifact review confirms six package files and no GPU imports.
See `evidence/offline-native-lifetime-fixture.json`.

The lifetime package deployed and verified on idle .40; both locks released.
See `evidence/console40-copy-lifetime-deployment.json` (relative to repo root).
Next: receipt-bound review, then one bounded execution.
The joined image/log baseline below is preserved. Installed image is
`dd5f92c5c117518958ec088da0ced54980904b61eec17302527344bd099227f7`. Native
acceptance requires 7168 existing comparison bytes plus 384 lifetime bytes,
eight allocations/frees, four retired queues, no GPU submissions and natural
exit. The 100 ms gate observation cannot prove entry into the freeing call.
Native lifetime validation, sustained replay, memory plateau and G5 remain open.

## Native joined capture passed

The two-stream producer/kernel-to-consumer-copy capture passed on .40, including
event rejoin, event reuse in eager execution, four dispatches, 292 exact values,
32 guards, six disabled-copy replays and eight callbacks. All four queues and
both allocations retired; the title exited naturally. Post-run installed files
matched, idle mounts matched, services stayed healthy and both locks released.
This proves the bounded producer/consumer topology, not concurrent GPU dispatch.
See `evidence/console40-copy-joined-success.json`.

Installed image: `6949263ffac68b8ea42df743c80fb38f5ee3f68299ead22809766cbcd83b4fd5`.
Current log: `cb4542048a934f644c064fda2eb252c2771f71dc2d535245f3965813fb49ec79`.
Both phases of `results/source-public-copy-joined-console-v1` are consumed.
Next: share the host allocation-lifetime cases with the CPU-only copy title,
qualify offline, then validate native allocation/worker retirement. Keep static
callback state/buffers for quarantine safety, checked thread lifecycle and bounded
waits. Expected added coverage is six allocations/frees and three exact 128-byte
comparisons; integrated queue counts must be measured before native acceptance.
Native lifetime, sustained replay, memory plateau, performance and G5 remain open.

## Previous offline joined capture and lifetime checkpoint

Full source ASan/UBSan qualification passed 84 compilations, including two-stream
GPU-producer/copy-consumer capture with event rejoin/reuse, exact dependencies,
missing-rejoin rejection and the existing output mutants. The shared oracle
retains four launches, 292 values and 32 guards, with four queues retired.
Separate lifetime cases passed queued free with a live executable and parameter
migration followed by freeing old allocations before replay; all six allocations
were released. Their 100 ms scheduling observation does not prove runtime entry.
A fresh 74-object native joined package and 26 control tests passed offline.
See `evidence/offline-joined-capture-lifetime.json`.

The joined package deployed and verified on .40; its prior image/log are
preserved, services healthy and both locks released. Installed image:
`6949263ffac68b8ea42df743c80fb38f5ee3f68299ead22809766cbcd83b4fd5`.
See `evidence/console40-copy-joined-deployment.json`.
Next: receipt-bound review, then one bounded execution. Native allocation-lifetime
validation, sustained replay and performance remain open.
Neither host modeling nor this one producer/consumer topology completes G5.

## Native copy recovery passed

The startup-instrumented package passed one bounded .40 run: 27 capture-recovery
cases, four rejected setters, two invalid update handles and compatible generic
update, followed by four GPU dispatches matching 292 values and 32 guards.
Six disabled-copy replays and eight callbacks passed. Both allocations were
freed, all three queues retired, and the title exited naturally. Mounted files
were reverified, idle mounts matched, services stayed healthy and both locks
were released. All eight startup markers appeared; elevation returned success
through the helper. The prior timeout's root cause remains unproven.

Installed image: `378b823a72930bea754938189b0d74e902d5489bac73d322dc02ae5a6739f8a4`.
Current application log: `10e600412f16acfd9fc2dc17cc4320457af9bec63341778efb661926c5cc296a`.
Both phases of `results/source-public-copy-startup-console-v1` are consumed.
See `evidence/console40-copy-recovery-success.json`.
Next: continue G5 allocation-lifetime and cross-stream capture work offline;
general update transactionality, sustained replay and performance remain open.
Native nested/subtype update rejection and startup reliability are not proven
by this bounded result. G5 and the broader SDK remain incomplete.

## Previous generic-update and recovery candidate

Generic executable updates now validate handles and preflight exact node types
and nested child counts before mutation. Full ASan/UBSan source qualification
passed late/nested mismatch rejection, preserved-output checks and compatible
copy/child/driver replay. A shared native fixture now checks 27 copy-capture
recoveries, four rejected setters and two invalid update handles before its
existing four-dispatch GPU oracle. The 74-object package, Make revalidation,
26 controls and independent extracted-artifact/ISA audit passed.
See `evidence/offline-graph-update-recovery.json`.

The recovery package deployed successfully, but its single .40 execution failed
at the 40-second observation deadline before producing an application log.
The loader reached PPSA99014 PID179; no HIP dispatch or recovery assertion is
proven. The controller safety-closed that exact title, confirmed idle/healthy
services and released both locks. There is no explicit native/GPU fault in the
scoped kernel log; absence does not rule one out.

Both phases in `results/source-public-copy-recovery-console-v1` are consumed;
never replay them. Image verified immediately before launch:
`dcfe7293c0034c21b6e5d36fb0e7b405eb763bd5a9c3041a730fc8cfd4858c8b`.
It was not reread after safety-close. The old image/log were preserved and
verified at deployment; the new application log was absent at final capture.
Next: investigate startup offline, including elevation before log redirection.
Separate PID180 debug-subsystem wait messages are a clue, not a root cause.
Any further hardware test requires a fresh frozen, reviewed candidate.
See `evidence/console40-copy-recovery-outcome.json` and
`evidence/console40-copy-recovery-deployment.json`.
Other parameter/dependency failures can still partly mutate an executable;
nested compatibility and direct child-setter safety remain separate gaps.
This does not close G5 or the SDK roadmap.

## Previous offline 1D copy-setter checkpoint

The two 1D memcpy-node setters now reject invalid executable handles and wrong
node representations before dereferencing or casting them. UBSan reproduced the
previous empty-node type confusion. Full source ASan/UBSan qualification passed
public wrong-node, internal 3D/symbol subtype, foreign/stale handle and subsequent
valid copy-update/replay checks. A 74-object native package build and cached
revalidation passed. See `evidence/offline-copy-setter-safety.json`.

This is offline validation only. The native package uses the existing copy_gpu
fixture, not a targeted rejection case; the installed .40 image is unchanged.
Next: fix generic hipGraphExecUpdate subtype handling, then prepare and validate
a shared failure/recovery fixture on .40. Allocation lifetime, cross-stream GPU
capture, sustained replay and performance remain open.

## Previous offline copy-capture checkpoint

Rejected asynchronous 1D copies now invalidate the PS5 capture session instead
of leaving a partial graph replayable. Full source ASan/UBSan qualification
passed 27 distinct failure cases across three capture modes and four copy APIs,
with cleanup/reuse and later replay. Valid 1D graph/executable parameter updates
passed source-isolation, offset/length, exact guards and rejected-update checks.
The 63 existing host regressions passed. See `evidence/offline-copy-capture-invalidation.json`.

This is offline evidence only; the installed .40 image below is unchanged.
Next: validate executable handles and exact node subtype in memcpy setters,
then package the new failure/recovery cases for .40. Joined-stream failures,
external capture-graph ownership, allocation lifetime, sustained replay and
performance remain open. Do not infer general memory-node safety from valid
1D-node updates.

## Current native public HIP checkpoint

Captured H2D/kernel/D2D/D2H graph replay passed its bounded .40 test through
WSL at `ac93df8`. Two changing-input rounds matched eager execution over four
GPU dispatches, 292 exact values and 32 guards. Six disabled-copy replays,
argument ownership/update and replay after source-graph destruction passed.
Both buffers freed, all three queues retired and PID 169 exited naturally.
The controller accepted 26 complete reads, healthy services and owned-lock
release. Independent audit verified identity, stable snapshots and scoped klog
without a native fault. See `evidence/console40-public-copy-kernel-graph-success.json`.

Current installed image:
`25cf1041ac4ac9b5a18e54728913816ce450a889bb9f76ac0096f79119ed9503`.
Current log: `/data/ps5-hip/results/ps5-hip-public-copy_gpu.log`, SHA
`0ad6a71f8f0810547b5a852134d16abeeff3af98298b21bfd76050eafd4062dd`
(not yet SHA-preserved). The prior kernel-capture image/log are preserved.
Both phases in `results/source-public-copy-gpu-console-v1` are consumed;
never replay them.

The shared oracle poisons both device buffers after eager execution. Offline
negative controls reject omitted H2D/D2D/D2H edges and missing/corrupt GPU writes.
Full source sanitizers, native Make and cached validation, 62 host regressions
and 26 controls passed. See `evidence/offline-native-copy-kernel-graph.json`.
The disabled-copy command retirement fix is separately recorded in
`evidence/offline-public-copy-graph.json`.

Next: copy-capture invalidation, memory-node parameter updates and lifetime,
GPU cross-stream capture, then sustained replay/plateaus and measured efficiency.
This bounded single-stream 1D copy/kernel result does not close G5 or the broader
roadmap.

## Historical source-build checkpoints

The mounted-path package deployment passed through WSL: all 21 reads and
seven mounted files verified, old image/log preserved, services healthy and lock
released. See `evidence/console40-mounted-path-package-deploy.json`. Independent
review permits one bounded execute tied to the exact receipt; deploy is consumed.

The mounted-path correction has a fresh 74-object Make package, cached
validation and independent artifact/deployment-only review at `7207a39`. Frozen
candidate: `results/source-public-launch-mounted-path-console-v1`; see
`evidence/offline-native-mounted-path-package.json`. Its unchanged kernel and
original GPU acceptance remain pending native validation.

Both native module fixtures now use the verified system-visible title mount
for code objects after filesystem elevation. The two changed title units compile
and independent review passed; see `evidence/offline-native-module-mounted-path.json`.
This corrects the source-supported namespace mismatch; a fresh package and native
run are still required to prove the file-open failure is resolved.

The diagnostic run identified `hipModuleLoad` status 301 (`hipErrorFileNotFound`)
and a null module: file open failed before parsing, lookup, allocation or GPU work.
PID 119 stopped after the title-specific safety close; services, idle state and
lock release passed. See `evidence/console40-module-file-open-failure.json`. Both
phases of `results/source-public-launch-diagnostic-console-v2` are consumed.

Current image: `1b844cd679fd688f93a43c45c6d11b953cd9b627db93840cdbf0ad93e28f17fa`.
Current launch log: `6baab2fea70f01c8069415594a0e0302249fe029138a92c52287efd2d8c35dbe`
(not yet preserved). The prior image and failure log were SHA-preserved. These
identities supersede all older baseline notes below.

The diagnostic v2 deployment passed: 21 complete FTP reads, exact seven mounted
files, preservation, idle state, healthy services and lock release. Its longest
image read took 33.921 seconds, validating the longer image-only deadline. See
`evidence/console40-diagnostic-package-deploy.json`. Deploy is consumed; independent
review permits one execute against its exact receipt. No new GPU result yet.

The unchanged diagnostic package is revalidated and frozen with corrected
controls at `ad02169` in `results/source-public-launch-diagnostic-console-v2`.
Independent deployment-only review permits one retry of the pre-upload transport
failure; see `evidence/offline-diagnostic-transport-retry.json`. Execution still
requires separate review of a successful deployment receipt.

The diagnostic deployment stopped before upload: the baseline image download
hit its 30-second deadline after 6,922,236 bytes. Services stayed healthy and the
owned lock was released; image and log were not renamed, and nothing launched.
See `evidence/console40-diagnostic-deploy-transport-failure.json`. The partial hash
does not verify the installed image. The consumed diagnostic v1 deploy is not reusable.

Image transfers now allow 90 seconds while logs retain 30 seconds and socket
inactivity stays at 8 seconds. All 22 control tests and independent review passed;
byte bounds and complete-content checks are unchanged. See
`evidence/offline-image-transfer-deadline.json`. Freeze and review a new candidate
with these controls before the one permitted retry of this transport-only failure.

The module diagnostic now has a fresh 74-object Make package, cached validation,
independent package audit and deployment-only approval at `6603bbc`. Candidate:
`results/source-public-launch-diagnostic-console-v1`; see
`evidence/offline-native-public-launch-diagnostic.json`. Offline stdio/import/parser
review found no concrete defect. The distinct case records the failing public API
and status; unchanged success criteria still require both GPU launches and teardown.

Module failure reporting now separates `hipModuleLoad` from
`hipModuleGetFunction` and records the numeric HIP status and handle presence.
The changed native title compiles; success acceptance is unchanged. See
`evidence/offline-public-launch-module-diagnostics.json`. This is diagnostic
coverage only, not a loader fix or a new packaged/console-qualified candidate.

The first native public HIP GPU-launch attempt failed at the combined module
load/function-lookup check, before allocation or GPU submission. The watchdog
fired; the controller closed PID 108 and verified idle state, healthy services
and lock release. No natural teardown or GPU pass is claimed. See
`evidence/console40-public-launch-failure.json`; both phases of
`results/source-public-launch-current-console-v1` are consumed. Next: separate
module-load and function-lookup diagnostics offline before another reviewed case.

Current installed image: `998e6b60805b8a9a92686629c93896d05e3fff9904f4d77a94dae5dc12d85459`.
Current launch failure log: `c74c92be3ff8c48d89420312b4bae766b94288f3fb1783d06957b771229ca8ae`
(not yet SHA-renamed). The previous graph image and log are now preserved.
Use this current pair for any future reviewed deployment; older baseline notes
below are historical.

A fresh 74-object public HIP GPU-launch package at `4acfab2` passed Make,
cached identity validation and independent package/kernel ISA review. It is frozen
in `results/source-public-launch-current-console-v1` with deployment-only approval;
see `evidence/offline-native-public-launch-current.json`. The pending native test
requires two launches, 146 exact outputs, 16 guards, two rejected geometries and
one fully retired allocation/queue. No GPU result is claimed before execution.

Deployment subsequently passed through WSL: all seven mounted files matched,
the graph image/log were preserved, and services and lock release passed. See
`evidence/console40-public-launch-deploy.json`. The exact deployment receipt has
independent approval for one bounded execution; the deploy phase is consumed.

The source-built public host-graph fixture passed on .40 via WSL at `cf9fafe`:
32 changing-input replays, 64 callbacks, six single-origin capture invalidations,
three caller-owned graphs and all five queues retired. The title stopped naturally;
ordered lifecycle, mount snapshots, services and owned-lock release passed. See
`evidence/console40-public-host-graph.json` and the separate deployment evidence.
Both phases under `results/source-graph-rejoin-console-v1` are consumed; never
replay them. This is native host-node graph acceptance, not GPU execution or G5
completion; cross-stream/rejoin matrices remain host-model evidence.

Next deployment baseline: installed image
`85bc98387b524e7898f15bcefd546cba7298bbe68be059cc585a6f8a3f7301c4`,
graph log `a233b4d1611c3d3b501b4632b88e3a8d457d3c4f6239a1b6531ea00dad1e489e`
(current, not yet SHA-renamed). Preserve this pair in the next reviewed cycle;
the older memory image and SHA-renamed memory log remain preserved. Recheck
services, idleness and baseline bytes under the owned lock before further work.

Descendant event rejoins now preserve the capture parent tree, fixing a reproduced
EndCapture stack overflow. Fresh source qualification covers 24 canceled rejoins,
invalidated/invalid-flag wait admission, three rejected cross-session merges and
eight exact changing-input replays of a successfully rejoined five-node graph.
See `evidence/offline-public-capture-rejoin.json`. The full 74-object native Make
rebuild and cached validation passed. The replacement candidate is frozen at
`cf9fafe` in `results/source-graph-rejoin-console-v1`; independent deployment-only
review passed. See `evidence/offline-native-graph-rejoin-runtime.json`. The later
deployment and execution passed as recorded above.

The owner confirmed .40 recovery and WSL-only access. After two failed service
checks and fresh owner confirmation, a read-only WSL preflight verified all three
services, no active application titles, the full installed image and preserved
memory log. See `evidence/console40-recovery-preflight.json`. Deployment still
requires a committed, frozen, independently reviewed candidate and fresh locked
preflight; the later graph cycle passed as recorded above.

Cross-stream capture invalidation now propagates across the tested capture tree,
fixing the reproduced false-success result from ending an invalidated child's
origin capture. Fresh qualification passes 24 cases across origin, child, sibling
and grandchild positions, all three modes and both graph ownership forms. All
four streams reset and can capture again; both events are reusable, caller-owned
graphs survive, and all four queues retire. The old runtime fails the same status
oracle. See `evidence/offline-public-capture-session.json` for source qualification,
native compilation and review; the failing baseline remains in
`evidence/offline-child-capture-diagnostic.json`. Concurrent mutation, complex
event rejoin topologies and further operation admission after invalidation remain
unqualified. The replacement native graph package described next includes this
runtime fix; its title still tests single-origin capture recovery only.

A fresh 74-object Make graph package at `c57f014` passed cached validation and is
frozen in `results/source-graph-session-console-v1`. See
`evidence/offline-native-graph-session-runtime.json` for package and frozen-candidate
audit identities. The unchanged title checks 32 replays, 64 callbacks and six
single-origin invalidations; the 24-case cross-stream matrix remains host-only.
Recovery confirmation, fresh baseline verification and independent operational
review are required before deployment. No console contact occurred.

A dedicated native host-graph fixture now shares its oracle with the sanitizer
model: explicit graph replay followed by capture recovery checks 32 changed-input
replays, 64 callbacks, six invalidations and three caller-owned graphs. It requires
five retired queues, no device allocations and no GPU work. See
`evidence/offline-native-graph-fixture.json` for fresh shared-oracle qualification,
74-object Make packaging, strict transcript checks and independent review.
This prepares native graph testing; it does not establish a console pass.

Adding the graph transcript and freezer support changed bound control inputs.
Earlier frozen candidates, including `results/source-copy-capture-console-v1`,
remain preserved but must not deploy through this changed checkout without a new
freeze and review. The earlier graph package was frozen at `454c8fd` in
`results/source-graph-console-v1`. Independent offline review verified all 28
bound files, six package files and 99 committed build inputs. See
`evidence/offline-frozen-native-graph.json` for identities and review provenance.
Recovery confirmation, fresh full-image and preserved-log baseline verification,
and independent operational review are still required before .40 deployment.
No console contact or native graph execution is claimed.

A replacement native copy package includes the capture cleanup at `50a2361`:
`build/public-hip-copy-capture-v1` passed the full 74-object Make build and cached
verification. Its frozen candidate is `results/source-copy-capture-console-v1`;
see `evidence/offline-native-copy-capture-runtime.json` for artifact review and
identities. The unchanged copy title checks 7,168 bytes and 16 rejections; it does
not exercise capture cleanup. Recovery confirmation, fresh baseline verification
and operational review remain required before .40 deployment. No console contact
occurred; older candidates and diagnostics are preserved.

Invalidated capture cleanup now resets the stream, releases internally owned
graphs and preserves caller-owned graphs. A reproduced baseline left status
invalidated after end-capture. Six origin-stream regression cases cover all three
capture modes and both ownership forms, then reuse the same stream for successful
capture and replay. See `evidence/offline-public-capture-recovery.json` for fresh
qualification, source-preparation drift checks, native compilation and independent
review. Child/event recovery and other rejected API invalidation remain open.
The previously frozen rectangular-copy package does not contain this runtime fix;
the replacement packages above now include it. The capture-specific native
fixture is built and its bounded console acceptance now passes as recorded above.

Host-node graph construction and thread-local capture now pass fresh source
qualification: each replays 16 changing inputs, updates callback functions and
user-data pointers, and survives source-graph destruction with exact 32 callbacks
and three retired queues. Capture performs no callbacks before launch, and a
no-op producer fails the unchanged output oracle. See
`evidence/offline-public-host-graph.json` for the 79-compilation/ten-link result
and independent review. Device-memory/kernel graph nodes, capture invalidation,
concurrent updates and performance remain unqualified; bounded native host-node
execution now passes as recorded above.

Legacy default-stream ordering now passes four fresh sanitizer-model cases:
ordinary streams order implicitly with the null stream in both directions, while
nonblocking streams allow the held producer to remain pending as the consumer
finishes. No explicit event wait supplies that dependency. Exact two-queue
retirement and successful post-synchronization queries are required. See
`evidence/offline-public-default-stream.json` for independent review and the
78-compilation/nine-link qualification. This is a bounded host-model observation;
native default-stream behavior, per-thread defaults and GPU concurrency remain
unqualified. The frozen rectangular-copy candidate is unchanged.

Historical rectangular-copy candidate: frozen at `27ebfd4` in
`results/source-rect-copy-console-v1`. Independent offline review verified all
28 bound files, six package files and 94 committed build inputs, plus the exact
7,168-comparison/16-rejection oracle. Both deployment and execution fail before
network access without operational approval. See
`evidence/offline-frozen-rectangular-copy.json`. No console contact occurred;
owner recovery confirmation, fresh baseline verification and operational review
remain required. Older frozen candidates are historical and may have stale
bound controls; do not substitute them for this candidate.

The native copy fixture now covers contiguous and rectangular 2D/3D transfers:
7,168 comparisons and 16 rejections over two changed rounds, with exact cleanup
and no GPU imports. Fresh 74-object Make packaging, cached verification, 62 host
tests, 19 control tests and independent artifact/oracle review pass. See
`evidence/offline-native-rect-copy-fixture.json`. No console run is claimed;
The candidate is now frozen as described above; recovery, baseline verification
and operational review remain required.

Historical runtime-only package: the native copy package includes the rectangular-copy runtime at
`61a1797`: Make compiled 74 objects, packaged and passed cached verification and
independent artifact audit. See `evidence/offline-native-copy-rect-runtime.json`.
Its title tests contiguous copies only; the newer rectangular fixture supersedes
that test scope. Console acceptance remains pending. No console contact occurred.

Public rectangular D2D copying now passes fresh qualification: 78 compilations,
nine links, 4,096 compared bytes over two changed rounds and 31 rejection checks.
Checked origin/span admission, adapter dispatch and synchronous failure propagation
fix the reproduced false-success path. Host/array rectangles and capture/graph
routes remain unqualified and reject on tested entry paths. See
`evidence/offline-public-rect-copy.json`; 62 host tests, 18 control tests and three
changed PS5 translation units also pass. That checkpoint did not include a native package or console run; the newer
package checkpoint above covers packaging only.

The native memory owner now implements rectangular D2D copying with checked
spans, padding preservation and exact row-overlap rejection. All 62 host tests,
PS5 native-core compilation and independent code/overlap review pass. See
`evidence/offline-native-rect-copy-core.json`. The public integration is now qualified in the newer checkpoint above; this
core-only evidence remains scoped to the primitive.

**Historical bug, now fixed in the qualified D2D path above:** public `hipMemcpy2D` could return success without
copying any requested bytes. Independent sanitizer reproduction observed 0/32
bytes copied for an 8-by-4 region with 16-byte pitches. The adapter rejects the
rectangular command while upstream forces asynchronous enqueue. See
`evidence/public-rect-copy-false-success.json`. The failing baseline is preserved; the newer rectangular-copy checkpoint
records the fix and its bounded acceptance.

Public asynchronous D2D dataflow now passes event-ordering and stale-data bypass
cases in the fresh 77-compilation/eight-link qualifier. Both verify 128 bytes,
three single frees and three retired queues. Independent review passed; see
`evidence/offline-public-async-copy-order.json`. This is bounded host-model
coverage, not native asynchronous transfer or GPU concurrency acceptance.

The retirement race now has a durable fresh-source regression in the public HIP
qualifier: 77 compilations and eight strict links pass independent audit, including
the fixed barrier and a test-only admission bypass with exact one-queue cleanup.
See `evidence/offline-retirement-regression.json`. The 500 ms observation is
scheduler bounded; this adds host coverage, not console acceptance.

Make now accepts `PUBLIC_BUILD_DIR`, `MEMORY_BUILD_DIR` and `COPY_BUILD_DIR`
to preserve earlier artifacts. A full default Make run with a fresh bootstrap
directory passed 62 host tests, native core and 74-object packaging, followed by
cached verification and independent audit. See `evidence/offline-make-output-overrides.json`.
No console execution is claimed.

The pitched-fill package is now frozen at `919ef03` and independently verified:
28 bound files, six package files and 91 committed build inputs. Missing operational
approval blocks deployment before network access. See
`evidence/offline-frozen-pitched-fill.json`. Recovery and baseline verification
remain pending; no console contact occurred.

The native fill fixture now includes pitched 2D/3D execution: two changed rounds
compare 4,096 bytes including guards, row padding and slice gaps, and check 16
invalid requests without mutation. Fresh Make packaging and offline review are
recorded in `evidence/offline-native-pitched-fill-fixture.json`. This is a prepared
console test, not a console pass; recovery confirmation is still pending.

The replacement native fill package now passes a fresh 74-object Make build,
repeated Make verification and independent package audit at `be2f0b1`. It includes
the retirement fix and checked pitched-fill runtime. Its existing console fixture
still tests contiguous fills only. See `evidence/offline-native-fill-retirement.json`.
No console run or operational candidate freeze is claimed.

Pitched 2D/3D fills now pass fresh host qualification with exact row/slice-padding
checks across two changed rounds (2,048 byte comparisons). Shared checked-layout
arithmetic validates the complete touched span before writing. Reserve/command
allocation failures return out-of-memory without mutation or leaked ownership.
A discovered queue-retirement race is fixed: synchronization now waits until
prior batch references are released, allowing one successful free. All 74 fresh
compilations, existing modes, 62 host tests, 18 control tests and three PS5
compiles pass. See `evidence/offline-public-pitched-fill.json`.

**Console candidates are held for rebuilding:** earlier frozen launch/fill
packages predate the retirement fix. Preserve their evidence, but do not deploy
them as current candidates. The replacement package above is built; no console execution is claimed.

The native fill candidate is frozen at `fff2451` and passed independent offline
identity review: 28 bound files and six package files. See
`evidence/offline-frozen-public-fill.json`. Deployment and execution remain
blocked by absent operational approval; no console contact occurred. Recovery
confirmation and complete baseline verification are still pending.

A dedicated native fill fixture now builds with `make public-hip-fill-package`:
74 fresh native objects and a six-file package, with GPU imports excluded.
Its expected console contract checks two changed-pattern rounds, 512 compared
bytes including guards, eight rejected requests, and exact cleanup without GPU
submission. All 62 host tests and 18 console-control tests pass. See
`evidence/offline-source-public-fill.json`. No console fill execution is claimed.

The updated fill runtime now builds through plain `make`: all 62 host tests,
native core libraries and a fresh 74-object bootstrap package pass. A separate
74-object launch package and repeated Make verification also pass. See
`evidence/offline-source-make-fill-runtime.json`. Both packages are offline only;
the launch fixture is unchanged and does not exercise fill calls on console.

Contiguous public HIP buffer fills are implemented through the serialized native
memory owner. Fresh 72-unit sanitizer qualification checks 1/2/4-byte patterns,
offsets, guards, rejected alignment/overflow/pitched/capture/graph inputs and
cleanup. Injected rejection propagates as `hipErrorUnknown` without changing
bytes; 62 host tests and two PS5-targeted compiles pass. See
`evidence/offline-public-fill.json`. No fill console execution is claimed. New packages are recorded above; the
previously frozen launch candidate still uses the prior runtime.

Logical cross-stream event ordering now passes the fresh 71-compilation sanitizer
qualifier: a held producer blocks the dependent consumer, while omitting the wait
allows early consumer progress; both cases retire two queues cleanly. See
`evidence/offline-public-stream-order.json`. This is bounded, scheduler-dependent
host-model evidence, not GPU concurrency or native event acceptance.

Timing-disabled public HIP events now pass a fresh host-model query/lifetime
case: NotReady during held dispatch, success after release, event destruction
and clean ownership. See `evidence/offline-public-event-query.json`. Independent
blocking event synchronization, timestamps and native event behavior
remain unqualified; the full streams/events gate stays open.

The launch candidate is frozen at `6f5167e` and independently verified offline:
29 bound files, seven package files and 86 build inputs match. See
`evidence/offline-frozen-public-launch.json`. No operational approval or console
contact occurred. Owner recovery status, remote baseline verification and a
separate deployment review remain required.

`make console-controls` now builds the read-only teardown helper as well as the
launch/close controllers. Repeated invocations verify the successful receipt,
artifact and input identities instead of accepting a failed build's receipt.
See `evidence/offline-source-console-controls.json`. All work is offline; the
helper does not itself certify console state.

The native public HIP launch fixture now builds with
`make public-hip-launch-package RESULTS_DIR=/data/ps5-hip/results`: 74 fresh
objects, genuine public module/launch/copy/synchronization calls and a checked
seven-file package including `thread_words.hsaco`. Its expected console contract
covers two changed-input dispatches,146 active values,16 tail guards, two rejected
unsupported blocks, and exact cleanup. All 62 host tests and17 console-control test
methods pass. See `evidence/offline-source-public-launch.json`. No console run
has occurred; the .40 recovery hold remains in force.

Public HIP launch admission now passes the fresh source sanitizer model: all twelve
base modes plus a separate launch harness, 73 exact modeled outputs, eight guards,
argument capture after caller mutation, and clean ownership. Missing/corrupt
computation negative controls fail the output oracle. Unsupported geometry, shared
memory, packed arguments and kernel capture/graphs fail before submission.
All 62 host tests and two PS5-targeted native translation-unit compiles passed.
See `evidence/offline-public-launch-admission.json`. This uses modeled providers
and transport; no public GPU launch on .40 or new native package is claimed.

The native public HIP module fixture now builds with `make public-hip-module-package`:
74 fresh native objects, a seven-file package with the compiled AOT canary,
validated imports and FSELF/package roundtrips. All 62 host tests pass.
Its loader-only backend rejects GPU creation; the expected console contract
checks simultaneous module isolation and clean teardown without allocations or
queues. See `evidence/offline-source-public-module.json`. Console execution
remains pending the existing .40 recovery hold; this is offline acceptance only.

Public file-backed AOT module loading now passes the genuine HIP host model:
load, function lookup and unload, copied-file ownership after unlink, malformed
image rejection, and both runtime-unbundler settings. The fresh 69-unit v2
passed all twelve ASan/UBSan modes, including real gfx1030 rejection,
simultaneous-module isolation and effective flag checks.
All 61 host tests and the native loader translation-unit compile passed.
See `evidence/offline-public-module-aot.json`. Pointer-only module images remain
explicitly unsupported; no public
module GPU execution or console module run is claimed. The native package is
qualified separately above.

Full public HIP host-runtime qualification now builds directly from pinned source
with `make public-hip-host-test`: 69 translation units; the original ten ASan/UBSan cases
passed, including exact cleanup/retention outcomes. No historical workspace
receipts are needed. See `evidence/offline-public-hip-source-model.json`.
This is a host model with simulated providers/transport, not console acceptance.

Public function discovery now exposes the validated AOT kernel table after load,
with atomic output publication and explicit invalid-state failures. Sixty host
tests, native object compilation and independent review passed; see
`evidence/offline-public-module-symbols.json`. Public module load/lookup/unload is now host-qualified above; public launch and
console qualification remain open. .40 connectivity has not been restored.

Public module-loading prerequisite: extraction failures now propagate instead
of dereferencing uninitialized device entries. Malformed result lists are rejected
before indexing. Sanitizer regression, unpatched negative control and native
translation-unit compilation passed; see `evidence/offline-public-module-errors.json`.
This does not enable or qualify public module loading/GPU execution.

Latest offline G5 progress: `7880705` builds the public synchronous copy fixture
with 58 host tests and 74 fresh native objects; independent package/ISA review
passed. Synchronous completion failures now propagate through `hipMemcpy`.
The .40 deployment failed during FTP verification; the copy title was never
launched. A subsequent bounded diagnostic timed out and its health probe returned
`No route to host`. Current installed-image integrity is unverified. The prior
memory log was preserved, and all owned locks were released. Do not replay the
consumed deployment or execute this candidate. See
`evidence/offline-source-public-copy.json` and the resume instructions in `HANDOFF.md`.

Latest G5 progress: source-built public HIP allocation/free passed on .40 at
`be6e521`. Two distinct 257-byte and 4097-byte allocations, interior-pointer
rejection without retirement, two frees, double-free rejection, watchdog join
and clean title shutdown passed. The prior bootstrap image/log were preserved;
services were healthy at that acceptance and the owned lock released. See
`evidence/native-source-public-memory-success.json`. This qualifies bounded
allocation ownership, not copied data or public HIP GPU dispatch. G5 stays open.

Source-only `make RESULTS_DIR=/data/ps5-hip/results` passed at `7342726`:
56 host tests, native core, 74 fresh public HIP objects and complete package
validation. A distinct reviewed package then passed the bounded bootstrap on
.40: public `hipInit=0`, device count `1`, full finalization, shell-exit return
`0`, natural title stop, matching mount observations and healthy services.
The owned lock was released. See `evidence/native-source-make-bootstrap-success.json`.
That bootstrap case claimed no allocation or GPU execution; the subsequent
memory case above qualifies allocation/free only. G5 remains open.

The preceding candidate `910f716` failed shell-exit with SIGSEGV because Make
omitted the native converter import-library patch. Its image, log and receipts
remain preserved, and its failure is not reclassified. See
`evidence/native-source-make-bootstrap-failure.json`. The repair applies the
patch, corrects import hash metadata and verifies serialized bindings before
signing. Historical D/E controls remain consumed or inert; original-workspace
recovery is no longer needed for fresh source builds.

## Actual status

- **Source-only public HIP package:** the root Makefile fetches verified
  public inputs, builds the SDK, zlib, UFS2Tool, exact-title Lapy helper and
  launch/close controllers, runs the host suite, cross-compiles the native
  core, and builds a new 74-object public HIP bootstrap with checked FSELF
  roundtrip and six-file UFS extraction. ProsperoEden and the protocol are
  locked snapshots of public `main`. Each Make output is an offline artifact
  until separately qualified; the corrected v2 is now qualified above. A fresh GitHub-only clone at `7ef291c`
  passed default `make`: 54 host tests, native core, 74 objects, strict link,
  package validation and cached revalidation. See
  `evidence/offline-main-source-build.json` and `docs/SOURCE_ONLY_REBUILD.md`.
  That earlier offline result is now followed by the fresh console bootstrap
  qualification above.

- **Historical public HIP bootstrap registration (superseded by the source-Make case above):**
  the independently reviewed, clean 74-object native bootstrap image
  (`836c10c6...`) was installed and registered as PPSA99014 by a one-use D
  control. Independent D review found the mounted EBOOT/libc bytes matched the
  package, the previous GPU image/log and 34 historical backups were preserved,
  and the title remained idle with services healthy and owned locks released.
  The D receipt and review are bound in
  `evidence/native-public-hip-bootstrap-deployment.json`. The public HIP E
  control remains inert and unbound: there is **no** public HIP startup, API,
  allocation, or GPU result from that historical package. Do not replay consumed D phases.

- **Historical native candidates:** a strict native public HIP
  allocation/free title passed 74 fresh-object build, package, and independent
  review on `development/g5-native-public-memory` (`1f969f6`); a ROCclr FP32
  GEMM title passed corrected GeometryLds admission, 39-unit sanitizer model,
  46-unit native link/package, and independent control review on
  `development/gemm-native-bridge` (`a0af56e`). Neither was launched. Their
  current controls must be checked against the now-registered memory image
  before any console use. See `HANDOFF.md`.

- **Genuine GPU-backed ROCclr dispatch passed on .40:** frozen `e79bd63`
  (integrated at `08ba1ad`) completed one A-write/B-copy/A-kernel/B-read
  upstream ROCclr cycle through the PS5 AgcBackend. PID 407/app `0xa018`
  reported 73 exact active values, 81 readback words with eight preserved
  copied guards, two checked queue joins, two buffer frees, watchdog join,
  application finalization and natural title stop. Independent reviews bind
  source/native/package bytes and deployment/execution raw evidence; pre/post
  matching mounts, installed bytes, services and lock release passed. The
  CPU-passing image and log plus 34 historical artifacts were preserved.
  Active outputs use scalar/index arithmetic; the guards check copy
  propagation. This establishes a bounded native ROCclr GPU path, **not**
  public HIP GPU support, concurrency, performance, full G4 or G5. See
  `evidence/native-clr-gpu-dispatch-success.json`.

- **Genuine public HIP allocation bridge qualified offline:** upstream
  `hipMalloc`/`hipFree` now run through the ROCclr SVM and PS5 modeled native
  registry in a 69-unit linked closure. All ten ASan/UBSan and ten TSan model
  modes passed, including interior/double-free rejection and the strict
  intentional failed-join leak checks. Independent review cleared this bounded
  model at `results/public-memory-bridge-full-review-v1/review.json`; the
  integrated patch stack is at `209664d` through `debf465`. Native public HIP
  GPU launch remains unproven; bounded native allocation/free and startup are qualified above. See
  `evidence/offline-hip-public-memory-bridge.json`.

- **Public HIP allocation prerequisite integrated offline:** isolated `70b2c19`
  adds bounded upstream `SvmBuffer` allocation/free through the PS5 device and
  native registry, retaining ownership on live-command or failed-unmap paths and
  rejecting false-success free. It is imported at `937a946`. An independent
  review cleared 18 ASan/UBSan and 18 TSan modes on 39 actual units, a genuine
  upstream SVM queue model, native Device/memory/HIP-memory compilation, and
  exact patch reproduction. Full public HIP bootstrap/link, real public API
  execution and GPU transport remain open; see
  `evidence/offline-clr-public-memory.json`.

- **Native upstream ROCclr CPU workload passed on .40:** frozen executable-path
  repair `3f4b885` was packaged, independently reviewed, registered and launched
  once as PPSA99014. PID 398/app `0x8018` completed eight computed outputs,
  eight unchanged failed-dependency sentinels, four memory and queue retirements,
  watchdog join, application finalization and a natural title stop. Independent
  deployment and execution reviews verify exact installed image and mounted bytes,
  pre/post matching mount rows, absent prior CPU log, preserved old image plus 33
  historical artifacts, healthy services and owned lock release. This clears the
  bounded **CPU** ROCclr startup/command/lifecycle prerequisite; it makes no GPU
  or public HIP claim. The exact affinity and path patches are integrated at
  `88d497c` with a passing pinned-upstream roundtrip and focused sanitizer test.
  See `evidence/native-clr-cpu-path-success.json`. The follow-on GPU candidate
  has since passed the bounded dispatch above.

- **Native ROCclr startup advanced, then trapped in `AppProfile` on .40:**
  independently reviewed affinity-repaired image `f63a8f3` registered with the
  diagnostic image and all 32 earlier artifacts preserved. Its one-use CPU
  launch reached PID 389/app `0x4018`, passed the earlier `Os::init` affinity
  boundary, then hit `SYSTEM_ILLEGAL_FUNCTION_CALL` at a direct Linux
  `readlink` syscall from the global `AppProfile` constructor before `main`.
  A later read-only check confirms the installed image is unchanged, .40 is
  idle and no CPU result log exists. The workload did not run and cannot be
  accepted or replayed. A new isolated candidate must repair PS5 executable
  path discovery and audit reachable pre-main calls. See
  `evidence/native-clr-app-profile-readlink-failure.json`.

- **FP32 NN GEMM subset integrated offline:** a reusable checked API with
  row-major strides and alpha/beta was imported from isolated `7feebc1` and
  rebuilt in this checkout. The root proof ran 17 sanitizer/model modes, 14
  numerical cases and genuine Linux/PS5 compiler paths with consumed inputs
  bound. This is host-model and compiler evidence only: no GPU numerical run,
  native framework integration or G6 acceptance. See
  `evidence/offline-gemm-subset.json`.

- **ROCclr pre-main failure localized on .40:** a distinct, independently
  reviewed diagnostic image reported page size 16,384 and 16 processors, then
  `pthread_getaffinity_np` returned 34 (`ERANGE`). PID 384 aborted before title
  `main`; mounts, idle state, services and both lock releases were verified.
  This consumed run is diagnostic only, with CPU/GPU acceptance false. Offline
  inspection of a native `libkernel.sprx` suggests the port's 32-byte CPU mask
  argument differs from a 16-byte console wrapper, but the dump's firmware is
  unproven. A new candidate must repair and validate the actual affinity ABI.
  See `evidence/native-clr-os-init-affinity-failure.json`.

- **Repaired CPU title failed native startup on .40:** frozen `6ed79f5` was
  registered and launched once; PID 381 aborted in ROCclr's `Os::init()`
  constructor before title `main` or the queue workload. Independent audit proves
  the fatal path but not which OS prerequisite failed. The FTP history transfer
  timed out, so post-run mounts and the result log remain unknown. Services stayed
  reachable and owned locks were released. This consumed case is not accepted or
  replayable. See `evidence/native-clr-cpu-startup-failure.json`; the next candidate
  must diagnose the first failing `Os::init()` call without bypassing it.

- **Repaired native CPU package independently reviewed:** frozen `6ed79f5`
  rebuilds all 34 native units and passes strict linking, 174-import/startup audits,
  FSELF conversion and independent six-file package extraction. The missing-tool
  packaging interruption and verified resume are recorded explicitly in
  `evidence/offline-clr-cpu-lifetime-package.json`. Fresh console-control binding
  and console validation remain pending; the old deployed image stays disabled.

- **Reviewed ROCclr PS5 adapter imported:** `9289710` imports all 21 files from
  frozen `169a118` exactly, including the device/AOT bridge, portable lifetime
  repairs, sanitizer models and saved-evidence checker. Independent review clears
  the final LF sources, 19 checker mutations and optimized-Python rejection.
  See `evidence/offline-clr-adapter-integration.json`. Full native adapter linking,
  console GPU execution through ROCclr and public HIP initialization remain open.

- **Repaired shared lifetime passes final CPU sanitizer review:** both 31-unit,
  seven-mode CPU proofs pass ASan/UBSan and TSan with independent source/dependency
  verification. Successful actual queue termination gates manual owner deletion;
  failed joins retain those owners. The genuine adapter and two upstream AOT
  bridge units also compile with the PS5 SDK without model hooks. See
  `evidence/offline-clr-lifetime-final-sanitizers.json`. Full native CPU rebuild
  and validation remain pending; the old deployed candidate remains disabled.

- **Lifetime repair has partial offline evidence:** six affected upstream units
  compile with the actual PS5 SDK at the isolated v5 checkpoint; see
  `evidence/offline-clr-lifetime-native-compile.json`. The replacement CPU client
  also passed seven fully instrumented ASan/UBSan modes with independent review,
  including failed-join owner retention. Subsequent TSan findings require newer
  event-publication and final-batch fixes, so neither result clears native launch.
  The old execution case is mechanically disabled and its reviewed bytes preserved.

- **ROCclr CPU launch held after new shared-thread findings:** the CPU package
  was deployed and registered on .40 with independently audited preservation,
  mounted bytes, idle state, service health and lock release; it was not launched.
  Full adapter ThreadSanitizer instrumentation then reported upstream thread-state
  and queue-teardown races in the same termination path used by this CPU title.
  Previous offline passes do not resolve this finding. Repair and qualify actual
  thread termination before rebuilding a new CPU candidate. Raw diagnostic:
  `.local/g5-clr-ps5-adapter/results/tsan-v1/normal.stderr`; deployment review:
  `results/clr-cpu-deployment-v1/deployment-20261013T201506.283228Z-independent-review.json`.
  Compact source-bound findings: `evidence/offline-clr-thread-lifetime-hold.json`.

- **Native ROCclr CPU title packaged and independently reviewed offline:** frozen
  isolated `5db9751` passes the repaired six-mode client proof, focused watchdog
  checks, actual SDK compile/link/conversion and startup/provider audits. Bound
  source bytes match Git, and independent UFS extraction verifies all six files.
  Console startup, command execution and natural teardown remain unvalidated.
  See `evidence/offline-clr-cpu-native-package.json`. The prospective console case
  checks eight computed values plus eight preserved failed-dependency sentinels;
  it makes no GPU execution or public HIP initialization claim.

- **Multi-module/query and reduction runtime integrated:** the exact qualified
  isolated implementation is imported at `de93ce1`, preserving the upstream CLR
  port. All 57 configured CMake tests, 24 multi-module runtime cases, 42 threaded
  ASan cases, 22 TSan cases, client/lifecycle suites and both native title builds
  passed offline. Fresh root compiler authority preserves both combined objects;
  independent review verifies the exact import and rejects corrupted/wrong-root
  authorities. Optional CMake proofs remain separately scoped. See
  `evidence/offline-multi-reduction-sdk-integration.json`. This adds no hardware
  execution or broader G4 acceptance.

- **Multi-module/query computation observed; full cycle incomplete:** frozen
  `b253c62` produced all 219 exact A/B/A outputs, 522 whole-allocation comparisons,
  84 guard checks, 32 successful allocation queries and eight invalid/stale
  rejections. Independent audit binds the fresh application log and mounted bytes
  to PID372/application `0x0000e018`, both unregister callbacks, State destruction
  and natural stop. The collector reached its 120-second deadline before sending
  the post-run snapshot, so original acceptance remains false. Preserve this
  consumed case. A separate read-only follow-up now passed independent audit:
  eventual mount rows match, installed bytes/log remain unchanged, no intervening
  title lifecycle or faults appeared, and services/lock cleanup passed. This does
  not retroactively accept the original cycle. See
  `evidence/native-multi-query-incomplete.json` and
  `evidence/native-multi-query-delayed-post-state.json`.

- **Upstream ROCclr executable-memory boundary qualified offline:** the current
  `ports/clr` patch rejects RWX reserve, commit and protect requests before any
  mapping mutation. Focused syscall checks preserve existing bytes and writable
  mappings; ordinary NONE/READ/RW operations and both real CPU command workloads
  pass. The native OS object and strict native workload link pass, and the patch
  reproduces all 14 adapted files exactly. This closes an unsafe path to an absent
  SDK hook; it does not provide executable memory or native application startup.
  See `evidence/offline-clr-rwx-guard.json`.

- **Experimental upstream ROCclr port now compiles and links natively:** all
  27 upstream units plus pthread TLS link into an ordinary shared ELF against
  the actual SDK. Independent review verifies unwind bounds and 144 strong
  imports; six weak SDK hooks still require application startup handling.
  The combined source passes the Linux two-queue CPU command workload, and the
  same workload also links natively. `ports/clr` preserves the 14-file patch and
  TLS helper, with exact patch-application verification. No native execution,
  public HIP initialization or GPU backend is claimed. See
  `evidence/offline-upstream-clr-native-port.json`.

- **Upstream ROCclr platform checkpoint passed offline:** actual two-queue
  write/copy/kernel/read commands preserve dependencies, memory lifetime and
  failure propagation on a synthetic CPU backend. The compute profile rebuilds
  27 upstream units plus pthread TLS, rejects unsupported graphics sharing, and
  compiles its three adapted graphics units with the PS5 SDK. A separate BSD OS
  adaptation compiles, passes affinity-mask sanitizer tests, and links its selected
  OS API calls against the real SDK. Combined native linking, native OS behavior
  and a GPU backend remain open. The validated narrow HIP runtime remains the
  current implementation; see `evidence/offline-upstream-clr-platform.json`.

- **Upstream block reduction passed on .40:** frozen `7b41b24` ran three
  dispatches with 16 groups of 1,024 threads and 4,096 bytes of dynamic LDS.
  All 48 signed outputs and 72 guard checks matched across changed inputs and
  a partial logical tail; input/mask readback stayed unchanged. Nine allocations
  and frees, genuine compiler/runtime cleanup, and natural PID363/application
  `0x0000c018` shutdown passed independent audit. This qualifies the bounded
  workload; general reduction libraries and performance remain open. Combined
  SDK integration is now qualified above; see `evidence/native-hip-reduction.json`.

- **Tracked synchronous device copies passed on .40:** frozen `9628fa6`
  completed 24 producer/consumer launches around 12 D2D copies. All 2,648 active
  outputs and 5,926 whole-buffer comparisons, including 630 guards, matched.
  Distinct and disjoint same-allocation ranges, changed second rounds, source
  preservation and all 21 allocations/frees passed. PID354/application `0x0000a018`
  stopped naturally after audited cleanup. This proves the bounded workload;
  DMA, bandwidth and general cache coherence remain unproven. The runtime and
  title are now integrated with 64 passing offline tests; see
  `evidence/native-hip-device-copy.json` and `evidence/offline-device-copy-sdk-integration.json`.

- **Tracked fills and device discovery passed on .40:** frozen `9bb191b`
  checked discovery before backend initialization, then consumed 24 byte fills
  in six GPU launches. All 662 active values, 746 input and 746 output words
  matched, including guards and changed second-round inputs on reused storage.
  PID345/application `0x00008018` stopped naturally after complete cleanup;
  independent lifecycle, service and lock audit passed. This proves the bounded
  workload, not DMA, performance or general cache visibility. The title and APIs are now integrated offline with 62 passing CMake tests;
  see `evidence/native-hip-memory-fill.json` and
  `evidence/offline-fill-sdk-integration.json`. The native run remains bound to
  the original frozen source and image; broader G4 remains open.

- **Bounded contending HIP callers passed on .40:** frozen `70a604a` observed
  overlapping A-launch/B-push API lifetimes and B requesting the same runtime
  gate while A owned it. Three main/A/B dispatches produced 331 exact values,
  813 readback and 813 template comparisons; seven AGC events passed and both
  workers joined. PID336/application `0x6018` stopped naturally after complete
  cleanup, with independent lifecycle, service and lock audit. Kernel-level
  blocked waiting, performance and GPU overlap remain unproven. The experiment
  is not yet integrated; see `evidence/native-hip-contention.json`.

- **Compiler/model dependency split integrated offline:** compiler-only proofs
  bind their actual sources, headers, tools and recipes; HIPIFY runtime models
  consume a separate immutable compiler proof. Native builds also rehash imported
  compiler helpers. Three compiler objects/device payloads remain unchanged.
  All 59 CMake tests, 34 threaded ASan modes, 16 CPU-only TSAN modes, refreshed
  main/init worker models and HIPIFY lifecycle checks passed independent review.
  The previously native-tested worker-init title is included in this line.
  See `evidence/offline-worker-sdk-integration.json`. This milestone adds no
  hardware execution; broader G3/G4 remain open.

- **Worker-side queue initialization passed on .40:** frozen `a7e6da7`
  observed no backend after the 17 host checks, then actual AGC initialization
  on worker A. Four A/B/A/main dispatches produced 660 exact values with 1,084
  device-readback and 1,084 template comparisons; all nine AGC events passed.
  Both workers joined and PID327/application `0x4018` stopped naturally after
  complete cleanup. Independent audit verified freshness, mounts, services and
  locks. See `evidence/native-hip-workers-init.json`. The runtime is already
  integrated, including this title variant. A bounded overlapping-caller case has now
  passed above; general concurrency remains unqualified.

- **Validated threaded runtime integrated:** `a5b89aa` brings the native-tested
  main-initialized worker implementation into this development line. Fresh
  source-bound compiler authorities preserve the exact example objects and
  shaders. All 55 configured Clang/ASan CMake tests, 34 threaded ASan modes,
  16 CPU-only TSAN modes and 29 HIPIFY title cases passed. See
  `evidence/offline-threaded-runtime-integration.json`. Worker-initialization
  candidate `a7e6da7` subsequently passed the separate native cycle above.
  G3/G4 and the larger platform plan remain open.

- **Joined HIP worker callers passed on .40 and independent audit:** frozen
  `2aa3c530` initialized the queue on main, then submitted from main/A/B/A/main.
  Both workers joined; five dispatches produced 661 exact values with 1,355
  device-readback and 1,355 separate host-template comparisons. All 17 phase
  checks and 11 directly observed AGC events passed. PID318/application `0x2018`
  completed unregister, State destruction and natural shutdown; mounted bytes,
  visible mounts, services and locks passed audit. This finite serialized case
  does not qualify worker-side initialization or simultaneous callers. The
  runtime is now integrated above; see
  `evidence/native-hip-workers-main.json`.

- **Threaded runtime passed its native single-thread control:** frozen `30f3c9f`
  ran the unchanged HIPIFY client on .40: six launches, 662 exact unsigned sums
  and 2,238 whole-buffer comparisons. Actual PID309/application `0x18` completed
  unregister, State destruction and natural shutdown. Independent audit verified
  log freshness, installed/mounted bytes, healthy services and released locks.
  The subsequent joined-worker case passed above. See
  `evidence/native-threaded-hipify-control.json`.

- **Threaded HIP runtime qualified offline:** isolated frozen `30f3c9f` on
  `development/hip-host-threads` passes 34 ASan/UBSan modes, 16 CPU-only TSAN
  modes, all five title lifecycle suites (145 cases/83 transcript mutations),
  and fresh legacy client regressions. Independent review binds all 30 changed
  files and covers checked locking, per-thread errors/configurations, failure
  retention and shutdown. This source is now integrated into this root above.
  Native HIPIFY control and main-initialized joined-worker qualification passed above;
  see `evidence/offline-threaded-hip-runtime.json`. G3/G4 remain open.

- **CPU threading passed on .40 and independent audit:** frozen `147c3b2`
  ran two workers with explicit 512 KiB stacks, 256 ordered updates, isolated
  emulated TLS values and exactly one key destructor per worker. Both joined;
  synchronization cleanup and natural PID300/application `0xe018` shutdown
  passed. Visible mounts matched, services stayed healthy and locks released.
  This CPU-only proof does not qualify concurrent HIP calls or worker-thread
  GPU affinity. See `evidence/native-host-threads.json`. Checked per-thread HIP
  state and serialized ownership now have the isolated offline proof above.

- **HIPIFY title/build integration qualified offline:** frozen `6e35911` is
  integrated with existing matmul and corrected math. Nine focused proof,
  profile, staging and log checks passed with explicit optional native-artifact
  skips. Independent review verified the exact 13-file change and unchanged
  core. The build selects the actual `ps5/translated.o` compiler artifact;
  no native rebuild or broader hardware claim follows. See
  `evidence/offline-hipify-title-integration.json`.

- **Corrected math fixture passed on .40 and independent audit:** frozen
  `1a399323`, integrated at `9b9c606`, passed all 1,542 results for 257 inputs
  under the unchanged numerical contract. Inputs, guards and 378 inactive words
  stayed intact; allocations closed and PID257/application `0x4018` stopped
  naturally. Visible mounts matched, all services stayed healthy and locks
  released. The correction changes two GPU cutoff immediates; all host binaries
  match the failed candidate. All 27 focused integration sanitizer tests passed.
  Its original failure remains preserved. This
  qualifies the finite fixture, not all-domain math or full G3/G4; see
  `evidence/native-math-exp-corrected.json`.

- **Tiled HIP matrix client passed on .40 and independent audit:** frozen
  `0d671174`, integrated at `b61af98`, completed eight ordinary HIP launches over
  four matrix shapes and two changed-input rounds. All 982 integer-valued FP32
  results were exact, including partial tiles; inputs and guards stayed intact.
  Compiler unregister, State destruction and allocation closure completed before
  PID266/application `0x6018` stopped naturally. Visible mounts matched, services
  stayed healthy and locks released. This proves the finite client using combined
  2D geometry and static LDS, not arbitrary GEMM accuracy or performance. See
  `evidence/native-hip-matmul.json` and `evidence/offline-hip-matmul-title.json`.

- **Unedited HIPIFY client passed on .40 and independent audit:** frozen title
  `1b9479c` completed six ordinary launches over lengths 1/73/257 with changed
  inputs. All 662 wrapping unsigned sums and 2,238 whole-buffer comparisons
  passed, including inputs and guards. Genuine compiler unregister, State
  destruction and allocation closure preceded PID275/application `0x8018`'s
  natural stop. Visible mounts matched, services stayed healthy and locks released.
  Pinned HIPIFY output and device bytes are unchanged; the host entry is renamed
  by compiler flag and called through its actual no-argument ABI. This qualifies
  this translated client, not general CUDA compatibility. See
  `evidence/native-hipify-client.json`.
  Compiler/runtime checkpoints `b1fd26c`/`c49c1d9` and root integration `0a6856d`
  retain their independent offline proofs; the root integration passed fresh
  three-object/19-model qualification while preserving matmul and corrected math.
  See `evidence/offline-hipify-client.json` and `evidence/offline-hipify-runtime.json`.

- **Reusable HIP client passed on .40 and independent audit:** frozen `ed82c331`,
  integrated at `5807a97`, completed eleven blocking submissions in sequences
  0/1/3/7 through one genuine two-kernel module. Exact results, changed inputs,
  guards, compiler unregister, State destruction and allocation closure passed.
  PID248/application `0x2018` stopped naturally; visible mounts matched, services
  remained healthy and locks released. All 76 offline integration checks passed.
  The zero sequence establishes prebackend state within this process. Serialized
  single-module limits and broader G4 remain open; see `evidence/native-hip-generic.json`.

- **Math validation found a real numerical failure:** the original `722a59a`
  run lost FTP observation within its 20-second bound and remains inconclusive.
  Read-only recovery retrieved the failure at output 645: `expf` of input bits
  `0xc2cff1b4` returned zero, while the unchanged oracle requires a nonzero
  subnormal. Offline reproduction identifies LLVM's explicit underflow cutoff
  in the emitted code. The exact title was subsequently cleaned up separately;
  this does not establish natural teardown or successful math qualification.
  See `evidence/native-math-current-failure.json`. The isolated compiler
  correction is now independently qualified offline in `5e856b4`: no-op builds
  reproduce original bytes, while corrected objects change only two cutoff
  immediates. The oracle is unchanged. The corrected fixture subsequently passed
  the bounded native cycle above; see `evidence/offline-exp-cutoff-correction.json`
  and `evidence/native-math-exp-corrected.json`.

- **Combined geometry/LDS route qualified offline:** isolated `89bf865` admits
  the measured 8x8 block with static LDS512 and complete hidden geometry through
  shared planners/ownership. Forty sanitizer checks and two final test-only
  replays passed, with independent review. The tiled client subsequently passed
  the native cycle above; see `evidence/offline-geometry-lds-runtime.json`.

- **Reusable HIP runtime qualified offline:** frozen `c0cc2bb`, integrated at
  `860b8f6`, selects supported launch classes from metadata rather than kernel
  names or fixed diagnostic counts. Six real compiler objects and 24 new models
  passed, together with existing client regressions. The integrated run passed
  73/74 checks; the remaining strict receipt check passed after supplying its
  complete native evidence, with no source change. Single module, serialized
  thread and blocking launches remain explicit limits. No new hardware claim;
  see `evidence/offline-bounded-hip-runtime.json`.

- **Fixed relaxed atomic suite passed on .40 and independent audit:** frozen
  `90df59d` completed six dispatches covering contended unsigned 32/64-bit add,
  CAS and indirect add. All 98,694 semantic values, wrapping tickets, guards and
  unchanged offsets passed; allocations closed. PID231/application `0xe018`
  stopped naturally, visible mounts matched, services stayed healthy and locks
  released. See `evidence/native-atomic-suite.json`. This does not qualify other
  atomic memory orders/scopes or general HIP atomic support; broader G3 remains open.

- **Shared genuine four-kernel HIP client passed on .40 and independent audit:**
  frozen title `67072dd`, integrated at `1fb5ada`, completed eight blocking launches
  in two changed-input sequences through one runtime: integer, static LDS,
  buffer scratch and 2D geometry. All 393,216 whole-page host word comparisons
  passed; this count includes inputs and guards, not only active shader outputs.
  Four compiler registrations, genuine unregister, complete State destruction and
  allocation closure preceded shell exit. PID222/application `0xc018` stopped
  naturally; visible mount identities matched, services stayed healthy and locks
  released. See `evidence/native-hip-routes.json`. Broader G3/G4 remain open.

- **Shared HIP runtime and genuine four-kernel client qualified offline:** frozen
  `de82c60`, integrated at `0381b05`. All 70 integration tests, 26 vector models,
  22 vector title cases and 19 four-route models passed. Linux/PS5 compiler
  objects, PS5 link/conversion and independent evidence review passed; 70 native
  imports and no external HIP imports. Integer, LDS, compiler scratch and geometry
  share one runtime across eight launches and two changed-input sequences.
  See `evidence/offline-shared-hip-runtime.json`. The descendant title passed
  71 integrated checks, 29 title lifecycle models, 19 ordinary route models and
  the unchanged 22 vector title cases before the fixed native cycle above. Its
  linked title has 117 imports and no external HIP imports; broad G3/G4 remain open.

- **Genuine HIP vector client passed on .40 and independent audit:** two blocking
  73-element launches produced 146 exact results, unchanged inputs and intact
  guards. Compiler unregister, runtime State destruction and allocation closure
  preceded successful shell exit. PID213/application `0xa018` stopped naturally;
  matching visible mounts, healthy services and released locks complete the
  bounded cycle. All 145 inputs and 27 raw files were audited. See
  `evidence/native-hip-vector.json`. This fixed client does not close broad G4.

- Shared owned native preparation is frozen at `7aed16b`: initial, LDS,
  buffer-scratch and geometry routes share checked allocation and explicit
  temporary cleanup. All 65 sanitizer tests, 26 genuine vector host-model cases
  and independent review passed. Existing HIP admission remains unchanged; the
  helper now has bounded hardware evidence through the fixed four-route client
  above, without generalizing its supported shapes or arguments. See
  `evidence/offline-owned-native-integration.json`.
- The reviewed planners and owned helper are now integrated into
  `development/offline-bringup` at `469ada5`. All 184 implementation, build and
  test paths match the qualified `7aed16b` tree exactly; only current documentation
  and evidence differ. See `evidence/offline-development-integration.json`.

- Geometry joins the shared pure planners in frozen commit `296222d`. All 63
  saved sanitizer tests and independent review passed. The 14 allocation,
  submission and HIP control files remain unchanged; this is offline integration,
  not new runtime admission or a combined hardware qualification. See
  `evidence/offline-geometry-core-integration.json`.

- **3D geometry fixture passed on .40 and independent audit:** 1,536 exact records
  (24,576 words), 819 logical points and 717 inactive tail positions, unchanged
  inputs/guards and closed allocations. PID204/application `0x8018` finalized
  and stopped naturally; matching visible mounts, healthy services and released
  locks complete the bounded cycle. All 149 inputs and 25 raw files were audited.
  See `evidence/native-geometry-3d.json`; arbitrary shapes and full G3 remain open.

- **2D geometry fixture passed on .40:** 384 exact records (6,144 words), 273
  logical points and 111 inactive tail positions. Inputs/guards stayed unchanged,
  allocations closed, and PID195/application `0x6018` finalized and stopped
  naturally. Visible mount snapshots matched; services stayed healthy and locks
  released. Independent review reproduced acceptance and verified 144 inputs and
  25 raw files. See `evidence/native-geometry-2d.json`.
- Checked scratch/LDS/private-call planners are integrated in isolated development
  commit `ce24c01`. All 58 sanitizer tests and independent source review passed;
  current allocation/submission and HIP controls are unchanged. This is pure
  planner integration, with no new HIP admission or combined hardware claim. See
  `evidence/offline-core-planner-integration.json`.

- **Partial-wave32 fixture passed on .40 and independent review:** lanes 4-19 in
  each wave produced 128 exact active words; 128 inactive words retained poison.
  Guards, allocation closure and finalization passed. PID186/application `0x4018`
  stopped naturally; visible mount snapshots matched, services stayed healthy and
  locks released. All 130 bindings and 25 raw files were audited. See
  `evidence/native-wave-partial.json`. Arbitrary masks and full G3 remain open.

- **Full-wave32 fixture passed on .40:** 256 exact words for broadcast, shuffle,
  ballot and reduction across two wave32 waves. Guards and allocation closure
  passed; PID177/application `0x2018` finalized and stopped naturally. The bounded
  cycle has matching visible mount snapshots, healthy services and released locks.
  Independent review rehashed 137 bindings and 25 raw files and reproduced acceptance.
  See `evidence/native-wave-full.json`; full G3 remains open.

- 2D/3D geometry candidates are frozen at `7bdcea0`. All 69 host checks and
  both native package validations passed; independent source/host and package
  reviews are clear. The measured 64-thread shapes use full HIP builtins and
  checked hidden arguments. Both exact O2 fixtures subsequently passed hardware
  cycles above; broader shapes and O1 hardware remain open. See
  `evidence/offline-geometry-readiness.json`.

- Prospective lifecycle parsing now tolerates the bounded, known resource-manager
  records observed between spawn context and matching PID. Eight offline tests and
  independent review passed; foreign/duplicate identities and faults still reject.
  Earlier raw logs and inconclusive acceptance remain unchanged. See
  `evidence/offline-bounded-lifecycle-parser.json`. Wave and geometry controls bind it.
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
  older candidates; both refreshed images now have exact hardware passes above. All 38
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

Advance G5 from the bounded native public HIP GPU pass to explicit-stream/event
ordering and resource-lifetime qualification with exact changed-input readback.
Preserve the current image/log in the checkpoint above, freeze a new candidate,
and independently review each deployment/execution. Historical bootstrap and
diagnostic phases are consumed; do not reactivate them. Use only WSL and .40.

Earlier G3/G4 work remains qualified only to its declared bounds:
Continue unified-runtime integration after the fixed genuine HIP vector client pass.
Both wave32 and both measured geometry fixtures have passed.
Preserve the installed HIP image/result, retained 3D image/result and earlier evidence, including
the original inconclusive private-call and visibility cycles. Atomic, numerical,
addressing and broader visibility qualification remain open; isolated fixture
passes do not close G3.

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

- 2026-10-15 | G5 | 7207a39 | .40 | pass: bounded native public HIP GPU | evidence/console40-native-public-hip-gpu-success.json | next: stream/event GPU ordering

- 2026-10-15 | G5 | main | fresh public event pending-query/completion/lifetime model | evidence/offline-public-event-query.json | next: isolate cross-stream event ordering and recover .40

- 2026-10-15 | G5 | main | frozen public launch identity/preservation review, no deployment approval | evidence/offline-frozen-public-launch.json | next: .40 recovery and operational review

- 2026-10-15 | G5 | main | source-only Make console helpers with failed-build/tamper rejection | evidence/offline-source-console-controls.json | next: freeze reviewed launch candidate and recover .40

- 2026-10-15 | G5 | main | source-built public HIP launch package and strict result/lifecycle controls | evidence/offline-source-public-launch.json | next: .40 recovery, frozen candidate and native validation

- 2026-10-15 | G5 | main | public launch admission and fresh sanitizer launch/output/ownership qualification | evidence/offline-public-launch-admission.json | next: native public launch package and .40 recovery

- 2026-10-15 | G5 | main | source-built 74-object module package, 62 host tests, loader-only import controls | evidence/offline-source-public-module.json | next: .40 recovery and public launch implementation

- 2026-10-15 | G5 | main | offline AOT load/lookup/unload and failure ownership | evidence/offline-public-module-aot.json | next: native package, .40 recovery

- 2026-10-14 | G5 memory | be6e521 | .40 | pass: 2 public allocations/frees, invalid-free rejection, clean teardown | evidence/native-source-public-memory-success.json | public copies/launch; G5 open

- 2026-10-14 | G5 startup | 7342726 | .40 | pass: public init/count and clean teardown | evidence/native-source-make-bootstrap-success.json | allocation then GPU; G5 open

- 2026-10-14 | G5 | 910f716 | .40 | failed: public init/count observed, shell-exit SIGSEGV | evidence/native-source-make-bootstrap-failure.json | fix converter; preserve v1

- 2026-10-14 | public HIP bootstrap D | `3ac8385` | .40: exact candidate registered, never launched; installed/mounted bytes, old image/log, 34 backups, idle/services/locks independently reviewed | `evidence/native-public-hip-bootstrap-deployment.json` | bind and review one-use E; G5 open
- 2026-10-14 | native public memory and ROCclr GEMM | `1f969f6` / `a0af56e` | offline native packages and independent reviews passed; no console runs | `HANDOFF.md` | refresh controls after bootstrap registration

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

- 2026-10-13 | G3 full wave32 | c3972cb | .40:256 exact words, clean bounded cycle; independent audit clear | evidence/native-wave-full.json | partial waves next
- 2026-10-13 | G3 geometry offline | 7bdcea0 | 69 host checks and both native packages independently reviewed | evidence/offline-geometry-readiness.json | hardware pending

- 2026-10-13 | G3 partial wave32 | c3972cb | .40:128 active +128 inactive words, clean bounded cycle; independent audit clear | evidence/native-wave-partial.json | 2D/3D geometry next

- 2026-10-13 | G3 geometry | 7bdcea0 | .40:2D384 and3D1536 exact records, bounded cycles independently audited | evidence/native-geometry-3d.json | genuine HIP client next
- 2026-10-13 | G3 core integration | ce24c01 | scratch/LDS/private planners combined;58 sanitizer tests and independent review passed | evidence/offline-core-planner-integration.json | add geometry
- 2026-10-13 | G3 geometry integration | 296222d | 63 sanitizer checks and independent review passed | evidence/offline-geometry-core-integration.json | shared owned preparation
- 2026-10-13 | G4 shared preparation | 7aed16b | 65 sanitizer tests,26 genuine host-model cases and review passed | evidence/offline-owned-native-integration.json | shared HIP runtime
- 2026-10-13 | G4 native client | 038fdb6 | .40:2 genuine HIP launches,146 exact results,clean unregister/destruction and bounded cycle | evidence/native-hip-vector.json | shared multi-kernel runtime

- 2026-10-13 | G4 shared native client | 67072dd / 1fb5ada | .40:8 launches,2 changed-input sequences,393216 whole-page comparisons,clean unregister/destruction and bounded cycle independently audited | evidence/native-hip-routes.json | broader G3/G4 client qualification

- 2026-10-13 | G3 atomics | 90df59d | .40:6 dispatches,98694 values,clean bounded cycle audited | evidence/native-atomic-suite.json | numerical and broader compute qualification

- 2026-10-13 | G4 reusable runtime | c0cc2bb/860b8f6 | offline qualification audited | evidence/offline-bounded-hip-runtime.json | dedicated native client validation

- 2026-10-13 | G3 math | 722a59a | failed numerical contract recovered; separate cleanup | evidence/native-math-current-failure.json | correct compiler underflow lowering

- 2026-10-13 | G4 reusable client | ed82c331/5807a97 | .40 | pass: 11 submissions and natural teardown | evidence/native-hip-generic.json | tiled matrix client

- 2026-10-13 | G3 exp cutoff | 5e856b4 | offline correction audited | evidence/offline-exp-cutoff-correction.json | new unchanged-oracle native candidate
- 2026-10-13 | G4 combined geometry/LDS | 89bf865 | offline route audited | evidence/offline-geometry-lds-runtime.json | genuine tiled-matrix client

- 2026-10-13 | G3 corrected math | 1a399323/9b9c606 | .40 | pass: 1542 unchanged-oracle values and natural teardown | evidence/native-math-exp-corrected.json | genuine tiled HIP client

- 2026-10-13 | G4 tiled HIP matmul | 0d671174/b61af98 | .40 | pass: 8 launches, 982 exact outputs and natural teardown | evidence/native-hip-matmul.json | HIPIFY runtime/title

- 2026-10-13 | G4 HIPIFY | 1b9479c | .40 | pass: 6 launches, 662 exact sums, natural teardown | evidence/native-hipify-client.json | CPU threading proof

- 2026-10-13 | G4 CPU threads | 147c3b2 | .40 | pass: 2 workers, 256 updates, TLS and natural teardown | evidence/native-host-threads.json | checked threaded HIP runtime
- 2026-10-13 | G4 HIPIFY integration | 6e35911 | offline | pass: 9 focused checks, exact reviewed source | evidence/offline-hipify-title-integration.json | threaded runtime
- 2026-10-13 | G4 threaded runtime | 30f3c9f | offline | pass: checked threading and lifecycle review | evidence/offline-threaded-hip-runtime.json | native control, then worker GPU submissions
- 2026-10-13 | G4 threaded control | 30f3c9f | .40 | pass: six HIPIFY launches and natural teardown | evidence/native-threaded-hipify-control.json | joined worker GPU submissions
- 2026-10-13 | G4 joined HIP workers | 2aa3c530 | .40 | pass: main-initialized queue, two joined workers, five dispatches and natural teardown | evidence/native-hip-workers-main.json | worker-side initialization, root integration
- 2026-10-13 | G4 integration | a5b89aa | offline | pass: threaded runtime, refreshed compiler authorities, 55 CMake tests | evidence/offline-threaded-runtime-integration.json | worker initialization
- 2026-10-13 | G4 worker initialization | a7e6da7 | .40 | pass: init on A, four dispatches, natural teardown | evidence/native-hip-workers-init.json | overlapping host callers
- 2026-10-13 | G4 contention | 70a604a | .40 | pass: overlapping HIP calls, 331 exact values, natural teardown | evidence/native-hip-contention.json | native memory fill
- 2026-10-13 | G4 fills | 9bb191b | .40 | pass: 24 fills, 662 GPU values, discovery and natural teardown | evidence/native-hip-memory-fill.json | SDK integration, D2D
- 2026-10-13 | G4 SDK | 9bb191b source | offline | pass: fill/discovery integration, 62 CMake tests | evidence/offline-fill-sdk-integration.json | native D2D

- 2026-10-13 | G4 D2D | 9628fa6 | .40 | pass: 12 copies, 2648 GPU outputs, natural teardown | evidence/native-hip-device-copy.json | SDK integration

- 2026-10-13 | G4 SDK | 9628fa6 source | offline | pass: D2D integration, 64 tests | evidence/offline-device-copy-sdk-integration.json | reduction and multi-module

- 2026-10-13 | G3/G4 reduction | 7b41b24 | .40 | pass: 3 dispatches,48 outputs,72 guards,natural teardown | evidence/native-hip-reduction.json | combined SDK integration

- 2026-10-13 | CLR | pinned65d174c | offline | partial-pass: real CPU commands,PS5 platform objects | evidence/offline-upstream-clr-platform.json | combined native closure

- 2026-10-13 | CLR port | pinned65d174c | offline | pass:28 native objects,strict link,exact patch | evidence/offline-upstream-clr-native-port.json | native startup

- 2026-10-13 | CLR memory boundary | reviewed v7 | offline | pass: early RWX rejection, ordinary memory and two command modes, native link, exact patch | evidence/offline-clr-rwx-guard.json | native CPU title startup

- 2026-10-13 | G4 multi/query | b253c62 | .40 | inconclusive cycle: exact computation and natural stop, post-snapshot missing | evidence/native-multi-query-incomplete.json | separate read-only observation

- 2026-10-13 | G4 multi/query | b253c62 | .40 | delayed post-state pass, original cycle remains incomplete | evidence/native-multi-query-delayed-post-state.json | native CLR startup
- 2026-10-13 | G4 SDK | de93ce1 | offline | exact multi/reduction import and root compiler authority validated | evidence/offline-multi-reduction-sdk-integration.json | native CLR startup
- 2026-10-13 | CLR | 5db9751 | offline | CPU package and exact source independently verified | evidence/offline-clr-cpu-native-package.json | native startup
- 2026-10-13 | CLR | 5db9751 | offline | launch held after shared thread-lifetime findings | evidence/offline-clr-thread-lifetime-hold.json | repair and requalify
- 2026-10-13 | CLR | 347fa8c | .40 | failed: AppProfile readlink syscall58 before main; delayed idle/image/log checked | evidence/native-clr-app-profile-readlink-failure.json | repair path query

- 2026-10-14 | Public HIP source build | 7ef291c | fresh GitHub-only clone, offline | pass: default make, 54 host tests, native core, 74-object title, FSELF/UFS audits | evidence/offline-main-source-build.json | independent candidate review before any console run

- 2026-10-14 | G5 public copy | 7880705 | offline pass: 58 host tests, 74 objects, independent review; .40 FTP deployment failed, no execution | evidence/offline-source-public-copy.json | connectivity and full image verification before new reviewed deployment

- 2026-10-14 | G5 recovery controls | offline pass: preserved-log verification and FTP diagnostics; 15 unit cases | evidence/offline-source-copy-recovery.json | restore connectivity

- 2026-10-14 | public modules | offline pass: extraction error guard, sanitizers and native object | evidence/offline-public-module-errors.json | module loading and execution

- 2026-10-14 | public modules | offline pass: loaded kernel discovery, 60 host tests, native object | evidence/offline-public-module-symbols.json | qualify real module loading

- 2026-10-14 | parser portability | offline pass: Clang18/libstdc++15 recursive metadata type, 60 host tests, PS5 object | evidence/offline-parser-portability.json | full-runtime sanitizer build

- 2026-10-15 | public HIP | offline pass: source-only 69-unit runtime, ten sanitizer cases | evidence/offline-public-hip-source-model.json | real module-loading qualification
