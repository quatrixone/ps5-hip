# LLM-first implementation plan

Reviewed: 2026-10-18. This is the authoritative first-release execution plan.
M0, M1, the selected real Q4 first-layer sequence and native full-decoder smoke
pass. Short native greedy generation now passes: two reset rounds, four new
tokens per round, 16 actual commits through all 22 layers, and all 384 float
boundaries bitwise CPU-oracle equal with clean lifecycle. This closes the bounded
short-generation checkpoint, not full M4/M5. Host verification improvements are
committed; the five-prompt / 16-token corpus now passes production-helper HIP
numerical interpretation with all 6,528 float files bytewise CPU equal and clean
ownership. Native corpus packaging, controller bounds and the linked-image audit now pass.
The historical FTP dot-entry rejection is fixed. The subsequent corpus v2
attempt failed before launch during existing-model FTP readback, followed by
`No route to host`; no corpus inference was observed. The four-token native
proof remains the latest console generation result. Original-safetensors host
inference and the linked native package now pass their separate offline gates;
original-format console execution remains pending. Extended scratch host numerical qualification passes: 1,176 commits and all
2,096 float outputs bitwise CPU-equal. The implementation is integrated after rebasing to `167b22d`
with unchanged numerical code and 229 combined tests passing. The supported main extended GGUF CPU oracle now passes fresh authentication
and matches all 2,096 scratch baseline outputs. Native extended execution remains
unproven. Retain M2 dense/extreme cases and
the original source-format loading requirement. Use idle .40 for meaningful
validation and push each validated milestone to main.

## Outcome and scope

Deliver a standalone native PS5 example that loads the locked TinyLlama 1.1B
Chat Q4_0 GGUF and generates tokens using ps5-hip for all model tensor computation.
This is a useful inference release, not completion of the wider ROCm SDK.

Validate the installed PPSA99014 application through its normal title lifecycle.
Raw ELF linkage is an intermediate build check; ELF-loader diagnostics and
deployment helpers cannot satisfy application startup or inference acceptance.

The fixed initial configuration is one request, one GPU, 22 layers, hidden width
2048, feed-forward width 5632, 32 query heads, four KV heads, head width 64,
vocabulary 32000, and a maximum combined prompt/output context of 512 tokens.
Weights are Q4_0 except the model's F32 normalization weights; KV storage is F16.
Model, tokenizer, conversion and client source identities remain pinned in the
existing input files. No maintained upstream fork is required.

CPU work is limited to loading/parsing, tokenization, scheduling, position/mask
input preparation, sampling and output. Embedding lookup, normalization,
projections, attention, feed-forward computation and logits run on the GPU.
Diagnostic CPU fallback must be explicit and cannot satisfy release acceptance.

Initially process prompt tokens sequentially with the same single-token path
used for decode. This is GPU prefill, with lower throughput than batched prefill;
it avoids making general GEMM or optimized multi-token kernels prerequisites.
Keep token-by-token numerical references consistent with this execution order.
A later prefill optimization must preserve those results and causal semantics.

The release must build from GitHub source through Make, generate reproducible
model/reference inputs, run repeated requests correctly, and publish measured
memory use and initialization/prefill/decode speed. There is no invented speed
or completion-percentage target. A correctness preview need not beat the CPU.

## Review conclusion

The compiler/runtime direction is justified by native arithmetic evidence. The
main planning risk is completing more isolated kernels without integrating a
client that can execute a whole model. Client linkage, the executed operator
inventory, model-memory integration and bounded native generation now pass.
The immediate inference work is wider-prompt and long-context qualification,
alongside original-format native loading. Do not expand the operator catalogue
without a concrete model or release requirement.

Use the existing llama.cpp loader, tokenizer, graph construction and scheduling.
Provide the smallest ggml backend adaptation that routes the required operations
through genuine public HIP allocation, module loading, dispatch and synchronization.
Reuse selected upstream kernels and existing adapter code where applicable.
Do not replace the SDK with a private direct-submission inference runtime or
require the entire upstream HIP backend to compile all
unused operator specializations. The client integration gate below must settle
the exact source selection and backend boundary before further expansion.

Full hipBLAS/rocBLAS, TheRock distribution packaging, Python/PyTorch, graph
optimization, VMM, managed memory, RCCL and on-console compilation are not v1
requirements. If an actual required operation needs a library function, either
provide its real bounded implementation or adapt that operation explicitly.
Never remove dependency checks and substitute success stubs.

## Native source-format requirement (2026-10-16)

The owner requires the installed PS5 app to read original `.safetensors` and
PyTorch `.bin`, `.pt`, `.pth` weights directly. Desktop conversion does not
satisfy this requirement. Keep the existing GGUF qualification baseline while
implementing this additional loading path; do not call extension recognition
or host import native support. A PyTorch runtime port is not a prerequisite.

Use the pinned client's `llama_model_init_from_user` interface to supply model
metadata in memory and fill allocated tensors from original files. This avoids
a second inference engine and an intermediate GGUF weight file. The adapter must
parse original configuration/tokenizer sidecars, validate tensor names/shapes,
and perform the HF Q/K RoPE row permutation. A packaged vocabulary GGUF is only
an intermediate, profile-bound test fixture, not complete source-format support.

Implement in these reviewable stages alongside the existing M0-M5 work:

1. Bounded native safetensors reader: validate header size, unique names, dtypes,
   checked shape products, exact byte ranges and shard ownership. Read tensors
   in bounded chunks without retaining a second complete model copy.
2. TinyLlama/Llama metadata, tokenizer and tensor adapter. Compare original
   checkpoint values, transformed Q/K rows, tokenization and resulting logits
   against the locked host oracle. Handle optional tensors and tied embeddings
   explicitly: the current callback API can create absent optional tensors and
   has no boolean failure return. Reject incomplete loads before exposing a model.
3. Restricted native PyTorch archive/state-dictionary reader for all three
   extensions. Validate ZIP/storage ranges, strides, aliases and tensor types;
   interpret only the supported tensor reconstruction records. Never execute
   Python code or arbitrary pickle globals. Unsupported object checkpoints and
   legacy serialization variants must produce explicit errors. Enumerate the
   supported serialization variants in the implementation and tests.
4. Integrate original bundles into persistent `/data/hip/models` storage,
   authenticated deployment, native package lifecycle and strict receipts.
   Qualify each format on .40: equivalent tensors/tokenization, bounded memory,
   clean repeated loading/cleanup and, when the inference gate is ready,
   equivalent generated results. No desktop-prepared weight file may be needed.

Original F16/BF16 weights must either have a qualified computation path or be
quantized on the console into the existing supported Q4_0 representation;
parsing alone does not establish inference. Any optional cache must be keyed to
all input identities and conversion settings, written atomically and invalidated
on changes. The accepted architecture/profile remains explicit: supporting a
serialization format does not support every model distributed in that format.

Status: bounded safetensors/index readers, exact TinyLlama tensor mapping,
original tokenizer metadata, row conversion and the tensor callback are
implemented. The supported host Make path loads all 201 original tensors and
matches all 384 short-generation boundaries bitwise over 16 commits/two resets.
The runtime reads original files; GGUF is only the independent numerical oracle.
See `evidence/offline-supported-original-model-generation.json`.
The native package and 435-function image audit pass offline; no original-format
native execution is qualified. See `evidence/offline-native-original-linked-package.json`.
PyTorch `.bin/.pt/.pth`, sharded-model integration and native format qualification
remain open. M0 and M1 are complete; M2 dense/extreme cases and full M4/M5
acceptance remain open. The removed desktop-conversion prototype is not an
implementation of this requirement.

## Evidence at this checkpoint

| Area | Proven | Still missing |
| --- | --- | --- |
| Inputs and CPU oracle | Locked model/tokenizer/conversion; native smoke matches all 72 CPU outputs bitwise | Full generated-history and long-context native comparisons |
| Public HIP foundation | Native full-client smoke routes 2199 GPU launches with model residency and balanced cleanup | Broader resources, failure and long-context qualification |
| Native arithmetic | Unweighted RMSNorm, Q8/Q4 and the learned weighted-normalization/full-QKV chain passed on .40 | Dense floating-point whole-layer and extreme-input coverage |
| Weighted QKV | M1 passed: 2048 weighted values, 2304 Q8 bytes, 2560 projections, all 30 buffers and clean lifecycle; now integrated in the bounded actual-model prefix | General input/sequence coverage beyond the four qualified position-zero tokens |
| Client | Full native decoder commits positions 0/1/reset0; all 22 layers and logits pass with clean ownership | Long full-model cache progression and remaining M2 fixtures |
| Full inference | Native live greedy feedback produces four tokens per reset round; all 384 layer/norm/logit outputs match CPU bitwise | Wider prompts, long context, performance and release qualification |

See STATE.md and the referenced compact evidence for individual outcomes.
A running package build is not a passed build. Historical compile-probe limits
are not current capability declarations. Native forward execution and bounded four-token generation
are established; full M4/M5 and release acceptance are not yet claimed.

## Next implementation checkpoints

1. Source-only Make reproduction of the native client, bounded vocabulary fixture
   and native application package now passes with independent freeze admission.
   Preserve that path without dependencies on historical build objects or
   local-only assets. Offline package success is not PS5 startup evidence.
2. Native loading, tokenization, real HIP storage and bounded call counters
   now pass on .40, including actual graph rejection and graph reuse.
   A vocabulary-only asset and a rejected synthetic ADD are useful preliminary
   tests, but cannot alone close M0's model graph routing requirement.
3. The selected GPU first layer now passes its 512-position/reset sequence.
   Complete the dense floating-point whole-layer fixture and extreme-input
   cases; retain the same backend implementations and numerical gates.
4. Full native smoke now passes model-scale residency, all 22 layers, final
   normalization/logits and real positions 0/1/reset0. Preserve its immutable
   receipts; long full-model context remains unqualified.
5. Preserve the qualified short sampling/incremental-decode path; qualify wider
   prompts, long context, EOS, repeated requests, errors, source-only rebuilds
   and measured performance under M4/M5.

The corrected source-only native vocabulary package is 19,922,944 bytes. Its reviewed
32 MiB transfer ceiling is specific to this fixture; legacy fixture limits
remain unchanged. Source/input provenance and real freeze admission pass in
`evidence/offline-native-llm-application.json`. Neither packaging nor tokenizer
success establishes inference. Installed vocabulary-title validation passed;
see `evidence/console40-native-vocab-success.json`. Actual-model loading/context/rejection/reuse also passed scheduler-v10; model
tensor computation and short native generation have since passed the separate
M3/M4 evidence below; full release qualification remains open.

## Required computation, in model order

The shapes below define the intended single-token path. Confirm exact strides,
views, dtypes and operation parameters from an executed graph trace; source
inspection alone cannot close the inventory.

| Operation | Required shape/behavior | Implementation target and acceptance |
| --- | --- | --- |
| Token embedding | Select one Q4_0 row from 32000 x 2048 | GPU gather/dequantization to F32; exact row identity and numerical comparison |
| Learned RMSNorm | 2048 F32 activations and 2048 F32 weights | Existing weighted kernel; attention norm, FFN norm and final norm |
| Q/K/V projections | 2048 input to Q2048, K256, V256 | Q8 activation then Q4 matrix-vector products; reuse actual GPU Q8 |
| RoPE | Q[32,64], K[4,64], actual token position | Locked model's exact rotation convention and parameters; positions 0, 1, intermediate and 511 |
| KV update | Per-layer K/V[512,4,64], F16 | F32-to-F16 conversion, correct writes/strides and session reset; no overwrite of other positions |
| Attention scores | 32 query heads over up to 512 cached positions | Correct grouped-query mapping to four KV heads, F16/F32 arithmetic and 1/sqrt(64) scale |
| Mask and softmax | Valid causal prefix for each query head | Stable max/sum/exp reduction, finite outputs and correct masks; future/unwritten cache must not contribute |
| Attention values | Probability-weighted V, combined 2048 output | Correct cache strides and head mapping; GPU float matvec/reduction subset, not a general BLAS port |
| Attention output and residual | Q4 projection 2048 -> 2048, add input | Existing quantized route extended to complete model shape plus F32 addition |
| FFN | Norm2048; gate/up 2048 -> 5632; SiLU(gate)*up; down 5632 -> 2048 | Real Q4 projections, activation/multiply, residual addition; frozen intermediate comparisons |
| Output | Final norm2048; Q4 projection 2048 -> 32000 | Full finite logits, compared to matched CPU reference; CPU sampling consumes those logits |
| Layout and execution glue | Views, reshape/permute, copies and conversions actually emitted | Metadata-only views when valid; real GPU conversion/copy where needed; reject unsupported strides/ops |

Do not infer full projection support from a one-row matrix test. Do not compare
quantized intermediate bytes to a frozen CPU intermediate if acceptable upstream
rounding can change quantization. First validate the actual upstream GPU output,
then derive the next-stage oracle from those captured bytes, while retaining
whole-layer and full-logit comparisons to detect accumulated error.

## Milestones and exit criteria

### M0 ? Establish the actual client boundary (V1.1 / V1.2)

Start immediately when implementation resumes, in parallel with M1.

CPU inventory is complete for the fixed configuration: `make llm-executed-graph`
and [the operator coverage table](LLM_EXECUTED_GRAPH.md) reproduce all 512
positions and the 14 observed operations. This closes only the CPU trace part
of M0; native loading/rejection/reuse and bounded storage calls have since
passed their separately recorded .40 gates. Model GPU computation remains M2.

The source-only payload-link prerequisite and weight-free vocabulary fixture
also pass Make reproduction. See [native linkage](LLM_NATIVE_CLIENT.md) and
[vocabulary fixture](LLM_VOCAB_FIXTURE.md). The installed vocabulary app now
passes on PS5, but the fixture cannot construct a real model graph. The v7
full-model test now demonstrates loading, context creation and actual-client
rejection/reuse. That older run was rejected for a context heap warning. Scheduler v10 now passes
strict clean-run/post-execution acceptance with 213 balanced owned allocations;
bounded per-HIP-call qualification now passes. See
`evidence/console40-native-hip-call-success.json` and
`evidence/console40-native-scheduler-model-success.json`.

- Link a native minimal client using the existing GGUF loader, tokenizer and
  graph builder with the chosen narrow ggml/public-HIP backend integration.
- Capture the executed CPU graph for the exact target settings: op, dtype,
  shape, stride, views and operation parameters. Trace prompt processing and
  incremental decode, including cache growth. Separately map and instrument
  GPU kernel specializations, public HIP calls and library calls in the native
  backend; CPU callbacks cannot establish that coverage.
- Maintain one coverage table with states: observed, implemented, host-tested,
  native-tested, integrated. Every missing item must point to its actual caller.
- Prove native loading/tokenization and backend selection with bounded assets;
  reject unsupported graph operations instead of silently falling back.
- Resolve coherent production headers/source selection. Diagnostic old hipBLAS
  headers and partial translation-unit compilation do not close client linkage.
- Promote the local graph-trace and link prototypes into reproducible Make
  targets. Bind every linked object, compiler, SDK and generated/header input
  into build receipts; qualify application imports, packaging and startup.
- Enforce the no-fallback rule in the actual llama execution path after graph
  placement and before computation. A standalone rejection test is insufficient.
  The genuine CPU backend may remain for upstream initialization, but no model
  tensor operation may execute there in the release configuration.

Exit: the native application builds reproducibly, loads/tokenizes on PS5,
selects the backend and rejects unsupported computation before execution;
every required operation/call has an implementation task. This is not inference.
If full upstream HIP linkage adds unused libraries or kernels, select only the
needed sources/backend operations rather than expanding the SDK to satisfy them.

### M1 ? Finish the current weighted QKV chain (V1.3)

Completed on .40 on 2026-10-15. See
`evidence/console40-public-weighted-qkv-success.json`. The candidate's deployment
and execution phases are consumed; the procedure below records acceptance,
not an instruction to replay it.

Use the already prepared candidate workflow. Audit the actual native package,
freeze its source and bytes, then run once on idle .40 under the console protocol.
Require all 2048 weighted outputs, exact Q8 derived from that output, and full
Q2048/K256/V256 projections; all 30 raw buffers, guards, unchanged inputs,
shared intermediates and clean lifecycle must pass. Keep the original failed Q4
case and qualified halfword workaround evidence. Do not relax tolerances.

Exit: real GPU weighted normalization and full projections compose correctly.
This closes one part of a layer, not attention or model inference.

### M2 ? Complete one transformer layer in the client (V1.2 / V1.3 / V1.4)

The selected Q4 first layer passed on .40 at source `f251366`: all 37 nodes,
512 positions plus reset/two repeats, 514 requests, 16,962 launches and 151
numerically validated readbacks. All 21 sampled stages include the final
residual; reset is exact and 528 allocations/frees balance with restored
memory. See `evidence/console40-model-layer-success.json`. The original full
graph remains rejected; this does not establish inference or decoder commits.

The selected real-model milestone is complete. M2 remains open for the small
dense floating-point whole-layer fixture and representative extreme inputs.
Keep activations on the GPU and reuse the integrated implementations.

Exit: one real layer matches CPU intermediate and final outputs for a token
sequence, with cache reuse at positions 0/1 and nontrivial lengths up to 512.
Include a small dense floating-point fixture to isolate indexing/numerics,
then real Q4 weights. Test grouped-query mapping, causal masking, cache strides,
reset and representative extreme inputs. Position zero alone cannot qualify RoPE
or attention. Do not introduce optional fusions or throughput work at this gate.

The local CPU trace observes padded attention extents of 256 and then 512,
strided F16 K/V layouts and 32-to-4 grouped-query mapping. Include positions
255/256 as well as 0/1/511, and ensure padded/unwritten entries are masked.
Port the traced SET_ROWS, gather, conversion and layout semantics explicitly;
simple contiguous matvec tests cannot qualify these paths. Independently review
and reproduce the trace when promoting its ignored prototype into the build.

### M3 ? Model-scale memory and full forward pass (V1.2 / V1.4)

Native real-model smoke passes on .40: three real decoder commits/reset, all
22 layer boundaries, final weighted normalization and 32,000 logits per request;
72 outputs are bitwise CPU-equal. Model storage and host/GPU cleanup are measured
and balanced. See `evidence/console40-full-decoder-smoke-success.json`. This
establishes the bounded real-model forward milestone; dense fixture, long
full-model context and failure coverage below remain open.

Begin memory integration alongside M2; do not wait until all operators are done.

- Advertise the real 64 MiB buffer maximum to the backend and use upstream
  tensor-boundary chunking. All current tensors fit individually; the largest
  is 35.15625 MiB. Do not raise limits to accommodate grouped allocations.
- Qualify cumulative residency: 590.414 MiB weight payload plus 11 MiB F16 KV,
  and measured activations, workspace, staging, host memory and runtime backing.
  The 84.516 MiB CPU workspace observation is not a GPU workspace measurement.
- Upload weights once and retain them across requests; use bounded scratch
  reuse and explicit KV ownership. Verify largest transfers and representative
  readbacks, failure cleanup and unload. Record actual allocation traces.
- Run the small dense fixture end to end, then all 22 real layers, final norm
  and all 32000 logits through the native client.

Exit: real-model GPU forward pass with matched layer boundaries and full logits,
within a measured memory budget. If fit fails, diagnose placement/workspace and
record evidence before changing the model or release scope; no silent downgrade.

### M4 ? Generate tokens with prefill and incremental decode (V1.4)

The short CPU greedy oracle now passes and is reproducible with
`make llm-generation-reference`. The corresponding actual-feedback HIP
interpreter now passes all 16 full-logit gates bitwise after full-mode nonlinear
and RoPE arithmetic fixes. The device fixture passes ISA/Initial admission. See
`evidence/offline-greedy-generation-reference-and-regression.json` and
`LLM_GENERATION.md` and `evidence/offline-greedy-canonical-arithmetic.json`.
The bound native short-generation profile has since passed on .40; matching
text alone is insufficient. Diagnostic CPU substitutions are never acceptance.

The exact production helper now passes the same short numerical gate, including
actual tokenizer/feedback, metadata, KV reset and checked phase/request timing.
See `evidence/offline-native-generation-helper.json`. Combined prompt/output
context is enforced before submission, including the final selected token.
Native title/assets/audit/freezer/controller binding and the four-token proof
now pass: two reset rounds, 16 commits and 384 bitwise-equal float boundaries.

The five-prompt corpus passes host production-helper numerical qualification,
but its native v2 attempt stopped before launch on a transport failure; see
`evidence/console40-corpus-model-read-transport-failure.json`. The extended
64-step, combined-512 and normal-EOS scratch host numerical proof now passes
1,176 commits and all 2,096 float outputs bitwise CPU-equal. Its implementation is integrated after rebasing
to `167b22d` with unchanged numerical code and 229 combined tests passing;
the supported main extended GGUF CPU oracle now passes fresh authentication
and matches every scratch baseline output. See
`evidence/offline-supported-extended-cpu-reference.json`. Native extended
execution remains open. Preserve full logits,
generated-history comparisons, reset and monotonic phase timings. The short
proof does not replace the full M4 gates below, and the 17-second whole-app
smoke observation cannot budget the longer profiles.

Connect CPU sampling to GPU logits, feed tokens back, advance position and reuse
KV. Process the prompt through the same GPU single-token path. Compare fixed
teacher-forced token sequences first, then deterministic greedy generation;
this separates numerical drift from divergent sampled histories.

Exit: at least five fixed prompts with 16 decode steps, a 64-step case and a
512-token boundary case pass predeclared intermediate/logit/error checks and
greedy-token checks. Investigate argmax differences; plausible text is not proof.
Validate normal end-of-generation stopping separately from fixed-step fixtures.
Reject context overflow before submitting work; reset must remove prior state.

### M5 ? Qualify the first release (V1.5)

Run at least 20 sequential requests and two unload/reload cycles. Test different
prompt lengths, safe-boundary cancellation, allocation failure, malformed and
unsupported models and explicit error propagation. No stale output, uncertain
GPU force-free, hidden CPU tensor fallback or unexplained ownership growth.
General asynchronous error propagation remains a known gap: the client must
receive failure for required paths; successful synchronization cannot hide a
failed dispatch. Pointer preflight alone does not solve this.

Exit: clean source-only SDK/client/model preparation on a second environment;
fixed-fixture inference and lifecycle evidence; measured initialization,
sequential-prefill and decode rates, peak memory, and exact capability limits.
Push each validated milestone to private main with state/handoff updates.
Publishing a public release or changing visibility still requires owner direction.

## Work selection and reporting

The critical path is M0 + M1 -> M2 -> M3 -> M4 -> M5. Memory work overlaps M2.
Full 22-layer forward execution and short native greedy generation now pass;
the next product-level proofs are the remaining M4 cases and original-format
native loading/generation.
Report completed exit criteria and remaining blockers, not API counts or a
single speculative percentage. Keep compiler, host-model, native-operator,
client-integration and release evidence distinct.

M0, M1 and the selected native Q4-layer milestone are complete. Finish the
remaining M2 numerical fixtures while qualifying wider full-model generation
and original-format native execution.
Every additional SDK capability must name the model
operation or release failure test it unblocks. Avoid new infrastructure unless
existing tools cannot establish that specific acceptance criterion.

Training, LoRA, other models/quantizations, batched prefill optimization, Flash
Attention, general BLAS, framework ports and broad ROCm coverage are deferred.
They retain their wider SDK goals; this inference release does not complete them.
