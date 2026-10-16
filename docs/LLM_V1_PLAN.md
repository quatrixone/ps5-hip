# LLM-first implementation plan

Reviewed: 2026-10-15. This is the authoritative first-release execution plan.
M1 is complete; M0 native application/backend integration is the immediate
priority. Use idle .40 for meaningful validation and push each completed,
validated milestone to main. This review does not itself initiate a console run.

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
client that can execute a whole model. Therefore client linkage, an executed
operator inventory and model-memory integration are immediate work, alongside
the current weighted-normalization/QKV numerical gate. Do not finish an entire
operator catalogue before attempting the client.

Use the existing llama.cpp loader, tokenizer, graph construction and scheduling.
Provide the smallest ggml backend adaptation that routes the required operations
through genuine public HIP allocation, module loading, dispatch and synchronization.
Reuse selected upstream kernels and existing adapter code where applicable.
Do not replace the SDK with a private direct-submission inference runtime, write
another model parser, or require the entire upstream HIP backend to compile all
unused operator specializations. The client integration gate below must settle
the exact source selection and backend boundary before further expansion.

Full hipBLAS/rocBLAS, TheRock distribution packaging, Python/PyTorch, graph
optimization, VMM, managed memory, RCCL and on-console compilation are not v1
requirements. If an actual required operation needs a library function, either
provide its real bounded implementation or adapt that operation explicitly.
Never remove dependency checks and substitute success stubs.

## Evidence at this checkpoint

| Area | Proven | Still missing |
| --- | --- | --- |
| Inputs and CPU oracle | Locked model/tokenizer/conversion; seven repeated inference cases and full logits | Native client results against these references |
| Public HIP foundation | Bounded native launches, memory/stream/event/graph/lifecycle paths; aggregate arguments and operator-specific resources | Exact full-client call coverage, model-scale residency and remaining operator resources |
| Native arithmetic | Unweighted RMSNorm, Q8/Q4 and the learned weighted-normalization/full-QKV chain passed on .40 | Remaining layer operators and client integration |
| Weighted QKV | M1 passed: 2048 weighted values, 2304 Q8 bytes, 2560 projections, all 30 buffers and clean lifecycle; independently reviewed | General input coverage and integration into the client |
| Client | Native full-model load, 512-token context, two actual graph rejections/reuse and 219 balanced owned allocations observed on .40; vocabulary/tokenizer and HIP storage previously passed | Resolve context heap warning, complete accepted post-run checks and per-HIP-call qualification; current full-model run is partial |
| Full inference | CPU reference only | One GPU transformer layer, all 22 layers, GPU logits and generated tokens |

See STATE.md and the referenced compact evidence for individual outcomes.
A running package build is not a passed build. Historical compile-probe limits
are not current capability declarations. No native LLM inference is claimed.

## Next implementation checkpoints

1. Source-only Make reproduction of the native client, bounded vocabulary fixture
   and native application package now passes with independent freeze admission.
   Preserve that path without dependencies on historical build objects or
   local-only assets. Offline package success is not PS5 startup evidence.
2. Validate native loading, tokenization and real HIP storage on .40. Enforce
   rejection in the actual client graph execution path, including graph reuse.
   A vocabulary-only asset and a rejected synthetic ADD are useful preliminary
   tests, but cannot alone close M0's model graph routing requirement.
3. Integrate GPU embedding and the proven normalization/QKV chain into that
   backend. Then add RoPE, KV writes, attention, output/residual and FFN until
   one complete layer passes sequence-level comparison. Do not build a separate
   diagnostic implementation of each operation.
4. In parallel with layer integration, validate full weight/KV residency,
   64 MiB buffer chunking and cleanup. Finish all 22 layers and full logits
   before adding generation behavior or optimizing throughput.
5. Connect sampling and incremental decode, then qualify repeated requests,
   errors, source-only rebuilds and measured performance under M4/M5.

The corrected source-only native vocabulary package is 19,922,944 bytes. Its reviewed
32 MiB transfer ceiling is specific to this fixture; legacy fixture limits
remain unchanged. Source/input provenance and real freeze admission pass in
`evidence/offline-native-llm-application.json`. Neither packaging nor tokenizer
success establishes inference. Installed vocabulary-title validation passed;
see `evidence/console40-native-vocab-success.json`. Actual-model validation remains open.

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
of M0; the native application and actual backend routing remain open.

The source-only payload-link prerequisite and weight-free vocabulary fixture
also pass Make reproduction. See [native linkage](LLM_NATIVE_CLIENT.md) and
[vocabulary fixture](LLM_VOCAB_FIXTURE.md). The installed vocabulary app now
passes on PS5, but the fixture cannot construct a real model graph. The v7
full-model test now demonstrates loading, context creation and actual-client
rejection/reuse. A context heap warning rejected its strict transcript, leaving
clean-run/post-execution acceptance and per-HIP-call qualification open.

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

Implement RoPE and F16 KV updates, then scores/softmax/value aggregation,
attention output/residual, and the normalized gated FFN/residual. Integrate each
into the same client/backend path; reuse those implementations in diagnostics.
Keep intermediate activations on the GPU. CPU readbacks are validation tooling,
not replacements for missing tensor operations.

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
One complete layer and then real logits are the next product-level proof points.
Report completed exit criteria and remaining blockers, not API counts or a
single speculative percentage. Keep compiler, host-model, native-operator,
client-integration and release evidence distinct.

M1 and M0's source-only CPU graph inventory are complete. Promote M0's native
link/backend prototypes into the supported build and qualify the actual
application boundary.
Every additional SDK capability must name the model
operation or release failure test it unblocks. Avoid new infrastructure unless
existing tools cannot establish that specific acceptance criterion.

Training, LoRA, other models/quantizations, batched prefill optimization, Flash
Attention, general BLAS, framework ports and broad ROCm coverage are deferred.
They retain their wider SDK goals; this inference release does not complete them.
