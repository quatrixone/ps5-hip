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
76-object public HIP bootstrap package. The
ProsperoEden and protocol locks refer to their public `main` snapshots; old
local-only commits are no longer build prerequisites. The protocol launch
controller now returns a status without printing a result, so the historical
one-use E control remains inert pending revised lifecycle checks and review.

## First-release priority

Follow [LLM_V1_PLAN.md](docs/LLM_V1_PLAN.md): standalone TinyLlama 1.1B Q4_0,
batch one and 512-token context. Resolve Q4 arithmetic, qualify required
operators and one complete layer, then full-model inference and release checks.
Training and broad SDK coverage are later milestones, not prerequisites.

## Plan review and resumption

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
`0.21718877422972582`, budget `8.8989529262781016e-06`. This is a numerical
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

## Resume here: native LLM inference v1

The owner prioritized a useful inference first release on 2026-10-15. Read
`docs/LLM_V1_PLAN.md` before choosing more general SDK work. Continue V1.1/V1.2
from the evidence below: model conversion and CPU references already pass, as do
bounded native compiler launches and upstream RMSNorm. The next gates are
Q8_1/Q4_0 dispatch and numerical validation, then the remaining client-required
operators and complete native inference.
Do not restart completed model preparation or broaden SDK coverage without a
concrete inference blocker. The release-progress table in the v1 plan separates
completed prerequisites from the still-open end-to-end release gates.
Use the release-contract checklist as the shipping gate: reproducible Make
build, actual GPU prefill/decode, CPU-reference correctness, qualified lifecycle
and measured performance. Full SDK coverage and training remain later work.
The inspected upstream master snapshot is
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`; local sparse source is ignored at
`.deps/sources/llama-client` (now a full checkout). Optional Make targets
`llm-deps-fetch`, `llm-cpu-reference` and `llm-model-profile` reproduce the
client inputs, host tools and locked model metadata. Read `docs/LLM_V1_PROFILE.md`
and `evidence/offline-llm-v1-inputs.json` before continuing. CPU quantization and
RoPE tests passed. Full source weights/tokenizer are downloaded and hash-verified
through `make llm-model-fetch`; see `evidence/offline-llm-model-source.json`.
F16 and pure Q4_0 conversion are verified through `make llm-model-convert`;
see `evidence/offline-llm-model-conversion.json`. The Q4_0 file is 619831552
bytes and retains 45 F32 tensors alongside 156 Q4_0 tensors. CPU reference
logits now pass seven cases twice, including 64 decode steps and a 512-token
boundary: `make llm-reference-fixtures` reproduces them from frozen token inputs.
See `evidence/offline-llm-cpu-reference.json`. Native client execution and GPU
comparison tolerances remain unqualified.
V1.1 remains partial. The native compiler-generated embedded-image launch now
passes its bounded .40 case. Continue with the demonstrated client ABI/resource
and math/library blockers using the pinned model and CPU reference fixtures.

That bounded probe is now recorded in `docs/LLM_V1_COMPILE_PROBE.md` and
`evidence/offline-llm-hip-compile-probe.json`: twelve host objects compile;
backend/getrows/mmvq device modules link. The follow-up also links norm/RoPE
using the same verified bitcodes with OCML before OCLC constants; a reversed-order
negative control confirms the cause. Resolve the remaining header/cooperative-library
and ABI/resource blockers before claiming a runnable client. Historical matching
hipBLAS headers were diagnostic only, not a selected dependency update.

Retirement-ack work was stopped before edits or experiments. Do not resume it
as a v1 prerequisite. Current public launch routes, model-scale memory and
required math operations are higher priority. The original SDK goals remain;
this changes release sequencing, not their acceptance criteria.
All older next-step paragraphs below are historical and subordinate to this
priority. The latest native compiler-launch checkpoint below records the current
installed image/log; the earlier backing-test image/log are preserved backups.

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

## Resume: aggregate native gate passed; real operator resources next

Latest native evidence: `evidence/console40-public-aggregate-success.json`.
Candidate source `55a6197934e6879f65710f982b0ddbcfbdb5b82e` passed on .40:
two genuine compiler-stub launches, 320 member bytes, 64 guard bytes, one
allocation/free and one retired queue. PID 249 exited naturally; finalization,
installed identities, idle snapshots, health and both lock releases passed review.

Installed package: `54d880eeb81097f7a9caaacad926a424673a1103b6fb33250d46f7b318c494f7`.
Installed FSELF: `6602ba5998b9245ac6a3923569b0fd82bf62c7f42d6c259e0572d0a877028dfc`.
Aggregate log: `4090033df580cdd6e15da34b52ba34a4030327222120a41863124d4158b8c3b1`.
Both phases in `results/source-public-aggregate-console-v1` are consumed;
root deploy12561 and execute28974 terminated successfully. Never reuse controls.
Previous embedded image/log backups are preserved on the console.

Genuine HIP compiler stubs now pass 12/16/128-byte by-value arguments through
CLR in eager and deferred host runs. Caller mutation, compiler offsets, zero
argument gaps, guards, unsupported-call recovery and clean modeled ownership
pass; no-op and corrupt transports fail the output oracle. The full 87-unit
CLR sanitizer qualification and 32 configured core regressions pass.
See `docs/PUBLIC_AGGREGATE_ARGUMENTS.md` and
`evidence/offline-public-aggregate-arguments.json` for exact scope and provenance.
Nested pointers remain opaque and require caller-managed lifetime. The next
concrete client gate is unchanged upstream 2048-column RMSNorm: 1024 threads,
128-byte dynamic LDS, 26 arguments, metadata-selected hidden LDS offset 352,
and unused null optional mul/add pointers. The canary-specific reduction route
does not implement this contract. Read `evidence/llm-next-runtime-gate.json`;
freeze actual ggml CPU outputs and tolerances before native numerical acceptance.

The native aggregate title now builds and passes image/package audits:
`build/public-hip-aggregate-native-v1/link-receipt.json`, package SHA
`54d880eeb81097f7a9caaacad926a424673a1103b6fb33250d46f7b318c494f7`.
See `evidence/offline-public-aggregate-native-package.json`. The subsequent
console checkpoint above validates this package. Full v1 inference remains open.

## Static module failure handling checkpoint

Previous native checkpoint: `evidence/console40-public-embedded-success.json`.
Genuine CLR compiler registration/stubs executed two GPU launches, 146 exact
values and 16 guards, then balanced device allocation/free and queue retirement.
PID 239 exited naturally; installed files, idle mounts, health and both locks
passed review. Its preserved previous package is
`e3d38022543668343a029d14cc55c1506bd5e1bc026719d3a1bd39c599093359`;
embedded log is `45c411ac7cc7203a726853146995cb5e88b463de20e5c467c803cb0f2c57d978`.
Both phases of `results/source-public-embedded-console-v1` are consumed. Preserve
its receipts; do not reuse its controls. Previous image/log backups remain on
the console. Earlier baseline/next-run instructions below are historical.

Latest offline checkpoint: `build/public-hip-embedded-native-v2` contains the
actual 77-object compiler-launch package and passing layout/startup audit.
Read `evidence/offline-public-embedded-native-package.json`; parent review
verified artifact hashes and unchanged sources for all 75 runtime units.
The earlier v1 audit failed only because it expected RX native text; the
converter emits execute-only text. The corrected audit keeps exact byte checks
and rejects incorrect permissions. Failed artifacts remain preserved. Deployment
and execution were completed by the native checkpoint above. The package receipt remains
offline evidence, while the separate console receipt proves the bounded run.

Earlier offline prerequisite: genuine CLR embedded registration passed the full
87-compilation host suite and actual compiler-stub tests (single/two modules,
reversed order, eager/deferred loading and repeated missing-name rejection).
Read `evidence/offline-public-embedded-images.json` and
`docs/PUBLIC_EMBEDDED_IMAGES.md`. The 76-object native bootstrap builds, but
does not contain the compiler-launch candidate. Next integrate/audit that title
and test it on .40; that bounded case has since passed above. Device backing cleanup is checked; complete static registry
retirement accounting remains unproven.

The publication-on-failure bug and its null-slot consumers are fixed and passed
focused plus full genuine CLR host sanitizer qualification. Read
`evidence/offline-static-module-failure.json` and `docs/PUBLIC_EMBEDDED_IMAGES.md`.
The first qualification directory `results/source-public-static-module-v1` is
superseded: it rejected changed inputs after caller fixes were added. The final
successful qualification is `results/source-public-static-module-v2`.

Next: compiler-generated bounded wrapper/bundle admission through genuine CLR,
generic per-object linker spans and registration failure lifecycle. Pointer-only
images remain unsupported; eager registration failure and broader lifecycle
handling remain open. No native package or console execution was qualified for
this change; freeze a fresh candidate before the actual registration/launch test.

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

The accounting API described in the current checkpoint is now implemented.
Use snapshot deltas after synchronization; do not reset counters. Submission
tests reclaim mock quarantines outside `NativeMemory`, which intentionally
does not change production accounting. Command lifetime and CPU preparation,
submission and wait measurements remain separate work.

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
Current log: `cb4542048a934f644c064fda2eb252c2581f71dc2d535245f3965813fb49ec79`.
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

- 2026-10-15 | G5 | ac93df8 | .40 | pass: bounded captured copy/kernel graph | evidence/console40-public-copy-kernel-graph-success.json | next: memory-node lifetime

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

## Historical pitched fill and retirement checkpoints

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

- 2026-10-15 | G5 | 4acfab2 | .40 | failed: public module before GPU | evidence/console40-public-launch-failure.json | next: loader diagnostics

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

- 2026-10-15 | G5 | cf9fafe | .40 | pass: native host graph and teardown | evidence/console40-public-host-graph.json | next: native public GPU qualification

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

Historical frozen fill candidate: `results/source-pitched-fill-console-v1`, source
`919ef03`, manifest `1f9214d0fc5d79af9219eb764c9777f24a7b81e1cad7b1d70f39e7c7ed31785b`.
Offline identity review passed; `review.json` operational approval remains absent.
See `evidence/offline-frozen-pitched-fill.json`. Do not deploy before recovery,
full baseline verification and independent operational review.

The earlier fill fixture is `build/public-hip-pitched-console-v1`: contiguous plus
pitched 2D/3D fills, 4,096 comparisons and 16 rejections over two changed rounds.
The transcript validator and candidate hypothesis require that expanded contract.
See `evidence/offline-native-pitched-fill-fixture.json`. Its bound control inputs have since changed; preserve this candidate and refreeze
before any future fill deployment. Earlier candidates must not substitute for it. No console run is claimed.

Replacement package `build/public-hip-fill-retirement-v1` now passes fresh Make
build and cached verification at `be2f0b1`, with independent offline audit. See
`evidence/offline-native-fill-retirement.json` for hashes. It retains the contiguous
fixture contract and is not yet frozen or approved for deployment. Historical
frozen candidates remain held; recovery and baseline verification are pending.

`results/public-pitched-fill-source-v1` passed the fresh source qualifier with
74 compilations: 69 base, three client harnesses and two injected allocation-failure
memory units. The pitched case checks 2D row padding, 3D slice gaps, equal pitch
and width with extra slice rows, changed rounds, overflow/bounds/shape rejection,
and zero-width no-op. Both reserve/new failures preserve bytes and clean ownership.
All base/launch/event/stream/contiguous cases, 62 host tests, 18 control tests and
three changed PS5 translation units pass. See `evidence/offline-public-pitched-fill.json`.

Initial diagnostics v1/v2 exposed a pre-existing retirement race: command completion
could precede worker batch-reference release, making a later free fail. The adapter
now requests a fresh finish marker and defers internal marker completion until FIFO
flush has retired preceding references. Independent held-worker comparison shows
old synchronization returns early while the corrected path waits and frees once.
Keep failed raw probes; do not hide the failure with a free retry or early release.

The earlier frozen launch and contiguous-fill candidates below contain the old
retirement behavior and are now held for rebuilding. Their manifests/packages stay
preserved; checkpoint files record the hold. A new native package and independent
review are required before console preparation. Owner recovery and full remote
baseline verification remain pending. Graph/capture fill remains unsupported.

## Frozen fill candidate

`results/source-fill-console-v1` in the WSL worktree is frozen at `fff2451`.
Independent offline identity review verifies 28 bound files, six package files,
build inputs, helpers and preservation requirements. See
`evidence/offline-frozen-public-fill.json`; `offline-review.json` explicitly has
`approved:false` and `deployment_authorized:false`. Operational `review.json`
is absent. An offline approval-gate check confirms both deploy and execute stop
before network access or phase directory creation.

This is preparation only. Await owner recovery confirmation, then verify the full
installed image and preserved memory log against the manifest, console idleness,
services and owned lock before a separate operational review. Baseline hashes
are requirements, not current remote observations. No previous consumed phase
may be replayed. Keep the earlier launch candidate and raw evidence preserved.

## Native fill fixture checkpoint

`make public-hip-fill-package FILL_BUILD_DIR=build/public-hip-fill-v1
RESULTS_DIR=/data/ps5-hip/results` builds the dedicated contiguous-fill fixture:
74 native objects and six package files, with no GPU-driver imports or code object.
The client rejects GPU backend creation, checks changed byte/16-bit/32-bit patterns
and low-byte truncation across two rounds, compares all 128 bytes before and after
four rejected requests per round (512 comparisons total), and verifies one
allocation/free and one retired queue. Host storage is static across quarantine.
See `evidence/offline-source-public-fill.json` for exact evidence.

The strict fill transcript and freezer support are included; all 62 host tests
and 18 console-control test methods pass. This package has not been deployed or
executed. Console recovery status, complete remote baseline verification and
independent operational review are still required. Preserve historical candidates
and consumed controls. The broad GPU/stream/graph roadmap remains open.

## Updated runtime Make checkpoint

At committed runtime `ea0c1ca`, plain `make RESULTS_DIR=/data/ps5-hip/results`
completed in the WSL worktree: 62 host tests, native core libraries, and a fresh
74-object bootstrap package at `build/public-hip`. The second builder invocation
verified the cached package. The separate launch package at
`build/public-hip-launch-fill-v1` also built 74 fresh objects and passed repeated
Make verification. See `evidence/offline-source-make-fill-runtime.json`.

These packages contain the fill implementation but retain the existing bootstrap
and launch clients. They provide no native fill execution evidence. Neither has
been frozen into a new operational console candidate or deployed. Preserve the
historical frozen candidate below; .40 recovery and remote baseline verification
remain pending. The native fill client and transcript are now built above; its operational
console controls still require recovery evidence and independent review.

## Contiguous fill implementation checkpoint

`results/public-fill-source-v1` passed the fresh 72-compilation qualifier: all
base, launch, stream and fill cases. `submitFillMemory` now writes captured
1/2/4-byte patterns (and validated internal power-of-two patterns up to 128 bytes)
through a fully resolved logical allocation under the device gate. Public
synchronous calls wait and propagate command failure; asynchronous calls remain
queued. Alignment and count overflow fail before enqueue. Pitched/multidimensional
fills and fill graphs/capture explicitly remain unsupported.

See `evidence/offline-public-fill.json` for exact evidence identities. The injected
unsupported-command probe returns `hipErrorUnknown`, preserves all bytes and
cleans up. All 62 host tests and the two changed PS5 runtime compiles pass.
Preserve diagnostic v2's failed byte comparison: it caught an unbraced upstream
`HIP_RETURN` macro and is superseded by corrected v3/fresh acceptance.

The source builder now applies the fill wrapper (including memory/copy prerequisites)
to every package fixture. The frozen launch candidate below remains a historical
prior-runtime candidate; current build-input hashes differ. Build and independently
review a distinct package/candidate for the new runtime before console deployment.
The .40 owner recovery question remains pending; no traffic occurred in this work.

## Offline event checkpoint

Fresh event qualification in `results/public-event-source-v1` completed successfully
(session65028 finished). It verifies NotReady while modeled dispatch is held,
successful event synchronization/query after release, event destruction and clean
ownership. See `evidence/offline-public-event-query.json`. This narrow claim does
not independently prove blocking synchronization; a later device sync still
protects readback. All twelve base modes and launch oracle controls also passed.

Logical stream ordering is now qualified separately in
`results/public-stream-source-v1`: 69 base units plus launch and stream harnesses,
all twelve base cases, four launch cases and two stream cases pass ASan/UBSan.
See `evidence/offline-public-stream-order.json`. The producer host callback is
held outside the adapter gate. The normal case observes no dependent consumer
progress within one second; the omitted-wait case detects early consumer progress
within the same deadline. Both release the producer and retire both queues.
This bounded observation is scheduler-dependent and does not prove GPU concurrency,
native event behavior, timestamps, re-recording or graph semantics.

Earlier `public-stream-dependency-probe-v1/v2` remain diagnostic only: GPU-held
adapter serialization confounded their missing-wait control. The new logical
probe avoids that confound without changing production runtime code.

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
  Integrated commits `209664d`Ã¢â‚¬â€œ`debf465`; see
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
  package/control review. It targets a 9Ãƒâ€”11Ãƒâ€”13 FP32 result with 99 active values
  and 45 padding guards. No native GEMM numerical result exists.

Overall progress is roughly **20Ã¢â‚¬â€œ22% of the twelve-goal ambition**, a judgment
about the full roadmap rather than a formal gate metric. G5Ã¢â‚¬â€œG12 are open.

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

## Scope update Ã¢â‚¬â€ 2026-10-14

The owner removed ProsperoAI integration. G7 now targets a standalone ggml/llama
inference client and SDK preview; application-specific ABI, UI and model coverage
are no longer acceptance requirements. Historical source/evidence records remain
provenance. No implementation gate is closed by this scope change.

## Fresh-checkout continuation Ã¯Â¿Â½ 2026-10-14

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

At the historical memory checkpoint, the installed PPSA99014 package on .40 was
`37decef81b2d32dace7662f0c003322a09da234275a92191628524a027a1e1b5`.
That checkpoint's memory log hashes to
`1425fabbd3a010b3be3bf3dc656c5c6fc1b39e4c4c8d9e4fe7491b82bf06791b`.
Both phases under `results/source-memory-console-v1`, and both prior bootstrap
cases, are consumed and must not be replayed. Previous image/log pairs remain
preserved. Services were healthy and the owned lock released at completion.
Use the current dedicated-controller environment in `.local/ENVIRONMENT.md`;
the sole lock for this session is `C:/Users/Denis/Documents/PS5/lock.txt`.

## Next work, in order

1. Keep the supported source pipeline on `main`; preserve both consumed fresh
   cases and their independent reviews. Do not reactivate historical D/E phases.
2. Implement and qualify native public HIP explicit-stream/event GPU ordering.
   Use the successful mounted-path GPU image/log baseline recorded above.
   The two-dispatch result does not establish broader stream or graph semantics.
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
observation is the native public host-graph case recorded at the top of this file.
