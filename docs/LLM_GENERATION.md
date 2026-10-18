# Greedy generation qualification

Native short-generation qualification now passes on .40: actual live-logit feedback
produces `, there was a` twice after reset, 16 commits traverse all 22 layers and
all 384 captured float outputs match the CPU oracle bitwise. See
`../evidence/console40-native-short-generation-success.json`. The 4.4-second
per-decode diagnostic timings include readbacks and file writes. Wider prompts,
long context and full M4/M5 acceptance remain open.

The native full-decoder smoke already qualifies three forced-token requests,
all 22 layers and full logits. M4 adds actual sampling feedback: the next input
must come from the preceding live logits, rather than a reference token list.
The first profile uses `Once upon a time`, the model's tokenizer with special
tokens enabled and special-text parsing disabled, and at most four new tokens.
It repeats after clearing the actual KV state.

`ports/ggml/generation.h` is the shared CPU scheduling/sampling driver. Tensor
computation belongs to its decoder callback. It checks all 32,000 logits for
finite values, selects the lowest token ID on an equal maximum, and stops on
the vocabulary's end-of-generation tokens or the requested output limit.
The final selected token is emitted without a subsequent decoder call; EOS is
also emitted without being fed back. Submitted positions therefore number
`prompt_tokens + generated_tokens - 1`. The worst-case **combined prompt/output**
budget must still fit 512 tokens, including the final selected token, before the
first submission; invalid requests submit nothing.
The caller owns decoder reset and error recovery.

On Linux/WSL, the source-only Make targets fetch locked dependencies and use
the existing model/CPU-reference prerequisites:

```sh
make llm-generation-reference
make llm-generation-numeric-tests
```

Existing authenticated inputs can be selected without rebuilding libraries:

```sh
make llm-generation-reference \
  LLM_CPU_BUILD_DIR=build/llm-cpu-reference-v3 \
  LLM_FULL_GRAPH_MODEL=/tmp/ps5-hip-tinyllama-gguf-v3/tinyllama-q4_0.gguf \
  LLM_GENERATION_REFERENCE_DIR=results/m4-generation-reference-repro
make llm-generation-numeric-tests \
  LLM_CPU_BUILD_DIR=build/llm-cpu-reference-v3 \
  LLM_FULL_GRAPH_MODEL=/tmp/ps5-hip-tinyllama-gguf-v3/tinyllama-q4_0.gguf \
  LLM_GENERATION_REFERENCE_DIR=results/m4-generation-reference-repro \
  LLM_GENERATION_NUMERIC_TEST_DIR=results/m4-generation-numeric-repro
```

Builders require fresh output directories. Receipts bind source, compiler,
consumed headers, runtime libraries, model and raw outputs. References contain
every committed input/position, all 22 layer outputs, final normalization and
full logits, generated IDs and vocabulary pieces. Python independently checks
argmax, feedback, stopping and exact reset. The HIP interpreter compares every
captured boundary to CPU under the existing numerical budgets. It uses the actual graph scheduler and production arithmetic
in a host interpreter; it does not execute a GPU.

The short generation profile has passed packaging, image audit, frozen
reference/readback contracts, timing and .40 validation. Preserve the accepted
native generation and smoke candidates; never replay either. This short profile
does not replace five-prompt/16-step, 64-step, 512-token, EOS and repeated-request
release gates in `LLM_V1_PLAN.md`, nor original-format loading requirements.

The corpus implementation uses a fixed source-bound five-case table shared by the
CPU oracle, numerical harness and production helper. Each case requests 16 new
tokens and repeats after KV/backend reset; preserve the short case as regression
coverage. Bind case/run/position to all boundary files, token histories and
transcripts. Keep one model/context resident across cases. Derive bounded app
and capture deadlines from the corpus commit count and measured diagnostic
cost; the short profile's 150-second limit is insufficient. Validate the complete
corpus offline before building and running one native image. A naturally early
EOS cannot silently satisfy the required 16-token case. Later use the same fixed
case mechanism for 64 tokens and combined prompt/output length 512; that combined
limit ends at decoder position 510 because the final emitted token is not decoded.

The fixed table is now `ports/ggml/generation_cases.def`. Capture its CPU oracle
with `make llm-generation-corpus-reference`; select a fresh output using
`LLM_GENERATION_CORPUS_REFERENCE_DIR`. Existing `LLM_CPU_BUILD_DIR` and
`LLM_FULL_GRAPH_MODEL` inputs are reused, with no CPU library rebuild when their
receipts exist. The short target and its output directory remain the default.
The corpus contains story, science, arithmetic, structured text and non-ASCII
continuations, each with 16 generated tokens and two reset rounds. Output is
`raw/<case-id>/` with a source-bound `corpus.json` index, and the total reference
artifact size is capped at 128 MiB.

The production helper exposes corpus profile 3 with one resident model/context,
case-specific output paths and logs, and strict early-EOS rejection. Run
`make llm-native-generation-corpus-tests` to compare the production helper with
the oracle under host HIP interpretation; select fresh output with
`LLM_NATIVE_GENERATION_CORPUS_TEST_DIR`. The builder authenticates the corpus
profile, uses its strict transcript parser and retains unchanged numerical gates.
It rejects corpus execution through the older non-production short harness.
Native corpus integration now includes title ABI 3, authenticated aggregate assets,
generation image-audit contracts, frozen nested references and controller readbacks.
The exact corpus is 272 requests, 217,056 nodes, 199,376 launches, 6,528 float files
and 11 metadata files. All six modules and the external model are reused.

Corpus budgets are fixed at 1,800 seconds watchdog, 1,830 seconds observation,
600 seconds raw capture and 4,200 seconds whole phase. The freezer binds these
values and admits a 128 MiB reference only for corpus; short limits remain intact.
The controller captures five fixed subdirectories, rejects stale/extra outputs,
uses exclusive local writes and checks all captured hashes against comparison.
Corpus application logs allow 2 MiB; baseline preservation also supports that size
so a subsequent deployment can preserve a completed corpus log.

Next build one fresh corpus package and inspect its actual raw, converted and
extracted image audit. Then freeze the exact candidate and validate on idle .40.
The implemented audit contracts and offline tests do not establish native GPU
execution of this corpus. See `evidence/offline-native-corpus-controller.json`.

The first accepted CPU corpus is `results/m4-generation-corpus-reference-v2`:
272 actual decoder commits, 6,528 float files (86,065,152 bytes), exact reset
equality in all five cases, and 11 metadata files. The first attempt's science
instruction produced early EOS and was rejected; its evidence remains in v1.
Continuation prompts were selected for the 16-token gate without weakening EOS
or numerical checks. `results/m4-corpus-native-helper-v1` passes mocked transport
and ownership checks with input-dependent feedback under ASan/UBSan. See
`../evidence/offline-five-prompt-generation-corpus.json`.

`results/m4-corpus-native-generation-numeric-v1` subsequently passes the actual
production helper with real model tensors and HIP kernel arithmetic interpreted
on the host. All 6,528 float files match the CPU oracle byte-for-byte, with exact
metadata and reset equality, 16 balanced allocations/frees, one backend instance
and 285 validated timing intervals. The final provenance/artifact receipt passes.
Host execution took 542.509 seconds; this is diagnostic timing, not PS5 throughput.
See `../evidence/offline-production-helper-corpus-numerics.json`. Native console
acceptance and the remaining M4/M5 gates are still open.

## Historical first offline outcome

The CPU oracle passes: five prompt tokens and three feedback decodes per run,
two runs, 16 full-logit files and four selected tokens `, there was a`.
The host HIP interpreter matches these selected tokens and resets all 384 raw
boundaries exactly, but fails all 16 numerical logit comparisons under unchanged
budgets. The maximum absolute error is 0.19304943 and relative L2 0.00918597.
See `evidence/offline-greedy-generation-reference-and-regression.json`.

Callback scheduling and external fusion settings were ruled out experimentally.
CPU SiLU substitution restores exact logits for positions 0/1 only; adding CPU
softmax still leaves 12 failures. Both are local diagnostic substitutions, not
production GPU work. Next inspect the first remaining position-2/layer-14
intermediate and required F32 contraction/rounding. Native generation remains
unqualified; the earlier forced-token native smoke is preserved independently.

## Current result: canonical GPU-source arithmetic

`evidence/offline-greedy-canonical-arithmetic.json` records the passing follow-up:
all 16 full-logit files are bitwise CPU-equal under unchanged limits, with actual
feedback/reset and balanced cleanup. All 384 captured boundary files reset exactly;
the independent CPU numerical reference here covers logits. The original
three-request/72-output smoke regression also remains bitwise CPU-identical.

Full mode now selects separate SiLU/softmax/RoPE symbols, retaining legacy
selected-layer behavior. The nonlinear routines translate pinned AVX2 polynomial
and reduction arithmetic into ordinary GPU scalar operations; RoPE uses explicit
FMAs. No CPU callbacks or captured activation data implement these kernels.
`kernels/llm_cpu_math.h` retains upstream MIT attribution. Standalone comparisons
against actual CPU library functions pass 33 SiLU and 134 softmax cases in each of
two compiler modes, including tails, masks, finite extremes and exact aliasing.
The fresh 16-symbol fixture passes Initial admission without private memory,
LDS, spills or undefined calls. Native arithmetic and performance remain open.

The initial arithmetic proof used `results/llm-layer-fixture-v8`, CPU oracle v1
and canonical numeric v1. These are historical generation receipts after the
combined-context fix. The production-helper result below used oracle v2; the
current all-boundary result uses v3. Old failures/diagnostics are preserved; their CPU
substitutions are never native proof. Next finish native profile/audit/reference
contracts and timings, then freeze and test a fresh package on .40.

## Production native helper qualification

The generation branch of `run_native_full` shares model/context creation and
cleanup with the existing smoke/sequence profiles. It consumes the actual model
vocabulary and live output logits, captures all 24 boundaries per decode, and
writes `generation.json` and `steps.jsonl` through the same exclusive writer as
raw tensors. It checks that the captured logits are the sampled logits. Both
rounds clear actual KV memory and backend sequence state; repeated selected/input
tokens and stop reasons must match. The final selected token is never decoded.

`make llm-native-generation-tests` executes this exact helper with the numerical
HIP interpreter, using `LLM_NATIVE_GENERATION_TEST_DIR` as a fresh output directory
and the existing model/reference overrides above. It does not deploy a PS5 ELF
or native package. With current local inputs, use CPU oracle
`results/m4-generation-reference-v2` and a fresh numerical output; the validated
helper result is `results/m4-native-generation-numeric-v4`.

`tools/source_llm_generation_log.py` checks every ordered raw/commit record,
metadata sizes, stop/text result, ownership counts and monotonic interval algebra
against authenticated CPU policy. Missing/duplicated stages, wrong tokens or
positions, oversized readback durations, bad timing sums and trailing data fail.
Its scope is the helper fragment; native startup, memory snapshots, watchdog join
and application finalization still require the future full-profile controller.

Timings use checked `CLOCK_MONOTONIC` nanoseconds. Per-decode intervals include
synchronous callback reads, file writes and callback log sync; readback time is a
subset and must not be added again. Run time includes all decode intervals;
remaining time combines scheduling, sampling and per-request logging. Load and
context intervals include surrounding initialization/setup. Cleanup covers
context/model/backend release. These offline timings describe a host interpreter,
not console latency or tokens per second.

The smoke and 514-request mock sequence remain covered. Forced-EOS transport tests
verify prompt completion and stopping before EOS feedback, with normal cleanup;
real numerical/native EOS qualification remains a separate release gate. The
combined 512-token limit now rejects a 512-token prompt plus output before submitting
a decoder request. The accepted context capacity and long-sequence requirements
were not reduced.

The native title currently selects smoke. Generation still needs an authenticated
build selection, profile-specific reference/assets, image audit, full transcript,
freezer/controller integration and .40 validation. Do not treat the host helper
as an accepted native application.

## All-boundary oracle and generation assets

The current oracle is `results/m4-generation-reference-v3`, and the production
helper comparison is `results/m4-native-generation-numeric-v5`. All 384 float
outputs are bitwise CPU-equal across 16 actual committed decodes. The CPU callback
captures each completed tensor before reuse and checks that captured logits are
exactly those consumed by sampling. The prompt and generated text are unchanged.
Old v2 oracle/v4 numerical receipts remain historical evidence.

References record the original producer output directory. Authentication checks
its exact compile/run command identities even after copying the entire reference
into a candidate, while hashing the actual copied artifacts. A copied reference
with a modified intermediate is rejected. This does not make toolchain paths
portable to another computer: source-only builds create fresh references there.

`build_public_hip_full_assets.py --profile generation` authenticates the same six
modules and binds all 384 float files plus `generation.json` and `steps.jsonl`.
The default remains `smoke`. This asset profile alone cannot deploy generation;
the native title, image audit, freezer and application controller must all agree
on that profile before console use. No native SDK/library rebuild was needed for
this oracle/host-comparison milestone.

## Generation reference and controller contract

The explicit generation profile now survives the original/copy authentication
boundaries in the reference freezer. It binds all 386 expected output files.
Package admission additionally requires matching `full_profile=generation` and
`actual_generation_call_chain_audited=true` from the native image audit. A smoke
image audit is rejected. The actual native builder/title/audit integration is
still pending, so this does not authorize a console deployment.

The readback verifier accepts `profile='generation'`, checks metadata bytes against
the authenticated oracle, independently validates greedy/reset controls, and
compares all 384 float tensors under unchanged budgets. It hashes captured bytes
before and after comparison and rechecks the complete reference closure. It does
not accept the interpreter's synthetic `backend.json` as a native readback.
The controller uses the frozen inventory, rejects stale token metadata, and checks
that numerical verification used precisely its captured bytes.

Application transcript validation adds the unchanged startup/finalization frame,
four ordered allocator snapshots with cleanup recovery, ownership, watchdog join
and generation timings. Transcript success still proves neither GPU origin nor
process teardown; those require the fresh image audit and actual controller run.
`make llm-native-generation-tests` runs these offline tests and the numerical
helper. Current helper output is `results/m4-native-generation-numeric-v6`, with all
384 tensors still bitwise CPU-equal. No native package rebuild was needed here.

## Native generation package selection

Build through Make with `LLM_NATIVE_APP_PHASE=model-full LLM_FULL_PROFILE=generation`.
The full-assets target selects `LLM_GENERATION_REFERENCE_DIR` and defaults to a
separate `results/llm-generation-assets` receipt; smoke retains its existing
default directory. Reused assets must have the requested profile. The app builder
authenticates that receipt, checks the pinned metadata sizes and derives the title
macro; there is no runtime profile file that can turn a smoke package into
generation. Weights stay external at `/data/hip/models/tinyllama-q4_0.gguf`.

The audit CLI takes `--full-profile generation` and checks the matching seventh
helper argument, generation tokenizer/reset/sampler/process/decoder/logit/EOS
call edges and both prompt/feedback process callsites. It retains the common HIP,
wrapper, loader and cleanup contracts. This is selected static call-chain proof;
indirect feedback/callback dataflow is additionally source/host/native evidence.
The native object preflight compiles and exposes the expected symbols/profile2
operand. Actual linked-image and console validation still follow this contract
milestone. Current native client v7 remains authenticated and can be reused.

## Frozen artifacts during deployment

For a new full-model smoke, generation or corpus candidate, add
`--frozen-artifact-admission` to `tools/freeze_source_candidate.py` with the
usual build, output, snapshot and baseline arguments. This authenticates the
complete producer closure before issuing `artifact-admission.json`; mutations
after authentication are rejected. Existing candidates retain their old checks.

The new controller path checks the exact frozen package, controllers, CPU
reference and current validator code, including the prompt policy, before and
after use. It does not reopen original producer paths during deployment or
numerical comparison. Independent review of the manifest remains required.
The copied provenance receipts are historical evidence, not executable paths.

Keep the candidate's repository-relative directory when relocating a workspace.
If the external model is absent on the console, supply `--model-file` to
`tools/source_console.py`; existing remote model verification stays mandatory.
The local two-lock wrapper must forward this option when uploading a missing
model. Source-only tests are `python3 tests/frozen_full_admission_test.py`.
This offline integration is not a real issuer qualification or measured speedup.
