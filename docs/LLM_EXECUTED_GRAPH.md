# Executed CPU graph inventory (M0)

`make llm-executed-graph LLM_EXECUTED_GRAPH_DIR=results/llm-executed-graph-v1`
fetches/builds the locked client and model prerequisites, then runs the genuine
CPU backend with context 512, batch/ubatch 1, F16 KV and flash attention disabled.
Four fixed prompt tokens are processed sequentially, followed by 508 greedy
decode tokens, deliberately continuing through EOG for bounded coverage.

The scheduler evaluation callback returns true on `ask` and records only the
post-compute callback. It forces one-node compute splits and synchronization:
this inventories actual graph nodes, not normal fusion or performance.
Every position saves complete finite 32000-way logits and its greedy token and
completed-node count. Full tensor/source dtype, shape, byte strides, view offset,
source slot and operation parameters are retained at positions 0, 1, 2, 3, 31, 32,
255, 256, 510, 511. Receipts bind the model, clean locked source revision, CPU
libraries, producer/harness/tests, compiler headers and tools. Runtime library
resolution must match the authenticated CPU libraries; all resolved libraries
are hashed. Changed inputs withhold success.

Observed attention scores use 256 padded cache slots through position 255 and 512
from position 256. K is F16 `[64,N,4,1]`, strides `[2,512,128,262144]`; transposed V
is F16 `[N,64,4,1]`, strides `[2,1024,65536,262144]`. Both require grouped-query
mapping to 32 query heads and masking unwritten slots. V cache updates are scalar
SET_ROWS scatter. Raw trace and inventory retain the exact caller tensor names.

The table distinguishes isolated diagnostics from a client backend. The current
storage prototype supports **zero tensor operations**. CPU observation does not
identify HIP calls, GPU kernel specializations or library calls; those columns
require instrumentation when real backend routing is implemented.

| Observed op | Actual graph caller in locked client | Implemented | Host-tested | Native-tested | Client-integrated |
|---|---|---|---|---|---|
| GET_ROWS | llama-graph.cpp build_inp_embd; models/llama.cpp output selection | No | No | No | No |
| DUP | llama-graph.cpp mixed embedding input | No | No | No | No |
| SET_ROWS | llama-graph.cpp mixed input; llama-kv-cache.cpp cpy_k/cpy_v | No | No | No | No |
| RMS_NORM | llama-graph.cpp build_norm | Diagnostic kernel | Yes, diagnostic | Weighted and unweighted diagnostics | No |
| MUL | llama-graph.cpp build_norm learned weight | Weighted fusion diagnostic | Diagnostic | Weighted QKV chain | No |
| MUL_MAT | llama-graph.cpp build_lora_mm/build_attn_mha | Q4 diagnostic subset; F16 attention missing | Q4 diagnostic | Q4 diagnostic subset | No |
| RESHAPE | QKV and cache layout construction | No backend route | No | No | No |
| ROPE | models/llama.cpp Qcur/Kcur | No | No | No | No |
| VIEW | llama-kv-cache.cpp cache views | No backend route | No | No | No |
| PERMUTE | llama-graph.cpp build_attn_mha | No backend route | No | No | No |
| SOFT_MAX | llama-graph.cpp build_attn_mha | No | No | No | No |
| CONT | llama-graph.cpp build_attn_mha output | No | No | No | No |
| ADD | models/llama.cpp attention/FFN residuals | No | No | No | No |
| SWIGLU (GLU) | llama-graph.cpp build_ffn | No | No | No | No |

Metadata-only views may avoid a GPU dispatch but must preserve ownership and
alias semantics. Q4 projection coverage must include outputs 256,2048,5632,32000
and widths 2048/5632. The two attention matmuls have F16 strided inputs and are
not covered by Q4 diagnostics. No context-reset, native full inference, numeric
GPU tolerance or HIP API coverage is established by this trace.

If the model already exists, Make skips fresh-only conversion and the trace
builder verifies its exact qualified size and SHA-256. A mismatched model is
rejected without replacement. The trace output directory itself must be new.

## Implementation tasks and call coverage

The tasks below turn the observed inventory into backend work. They are not
implemented routes. The evidence states in the table above remain unchanged.
M0 now has native actual-client rejection and the bounded call
qualification described in [M0-CALLS](#m0-calls). Completing the operators
belongs to M2; the existence of a task is not evidence of implementation.

| Observed operation | Backend task and specialization | Actual locked-client caller |
| --- | --- | --- |
| GET_ROWS | [M2-GATHER](#m2-gather): Q4_0 token embedding to F32; separately, F32 output-row selection | `llama-graph.cpp::build_inp_embd`; `models/llama.cpp` output selection |
| DUP | [M2-COPY](#m2-copy): materialize the observed mixed-input copy | `llama-graph.cpp::build_inp_embd` mixed input |
| SET_ROWS | [M2-KV](#m2-kv): mixed-input row insertion plus F32-to-F16 K/V cache scatter | `llama-graph.cpp::build_inp_embd`; `llama-kv-cache.cpp::cpy_k/cpy_v` |
| RMS_NORM | [M2-NORM](#m2-norm): graph normalization and legal learned-weight fusion | `llama-graph.cpp::build_norm` |
| MUL | [M2-NORM](#m2-norm): learned F32 weight multiplication, fused only when graph semantics permit | `llama-graph.cpp::build_norm` |
| MUL_MAT | [M2-MATMUL](#m2-matmul): Q4 projections and two separate strided F16 attention products | `llama-graph.cpp::build_lora_mm/build_attn_mha` |
| RESHAPE | [M2-VIEWS](#m2-views): validate shape and alias ownership without materializing a copy | QKV/cache construction in `llama-graph.cpp` and `llama-kv-cache.cpp` |
| ROPE | [M2-ROPE](#m2-rope): Q32/K4 heads of width 64 using actual model positions | `models/llama.cpp` Qcur/Kcur |
| VIEW | [M2-VIEWS](#m2-views): validate offsets, strides, extents and backing lifetime | `llama-kv-cache.cpp` cache views |
| PERMUTE | [M2-VIEWS](#m2-views): preserve the observed attention stride permutation | `llama-graph.cpp::build_attn_mha` |
| SOFT_MAX | [M2-SOFTMAX](#m2-softmax): scaled, masked causal reduction over padded cache slots | `llama-graph.cpp::build_attn_mha` |
| CONT | [M2-COPY](#m2-copy): materialize noncontiguous attention output | `llama-graph.cpp::build_attn_mha` |
| ADD | [M2-RESIDUAL](#m2-residual): F32 attention and FFN residual addition | `models/llama.cpp` residual paths |
| SWIGLU | [M2-FFN](#m2-ffn): observed split activation/multiply at width 5632 | `llama-graph.cpp::build_ffn` |

### M0-CALLS

Eleven bounded per-call counters are now implemented, with attempts recorded
before HIP entry and completions/results after return. Requested bytes, failures,
last result and first observed error are retained. Invalid spans reject before
counting a HIP attempt. Graph-statistics reset does not erase call accounting.

Seventeen host injection cases cover success, bounds, every error/cleanup site,
concurrent calls and a call that never returns. Fresh genuine-runtime host
qualification also passes in eager/deferred modes with the actual loaded model.
See `evidence/offline-hip-call-qualification.json`. Native qualification now passes:
ten exercised sites and an explicitly zero cleanup-free row, exact partial-fill
surroundings, clear, synchronization and balanced teardown. See
`evidence/console40-native-hip-call-success.json`; cleanup-free remains host-only.

Current `GateStatistics` measures wrapper calls, rejection, forwarding, direct
backend graph calls and the last graph pointer. The actual-client harness also
checks post-compute callbacks and upstream `n_reused`. Runtime observations count
allocations, frees, queues, retirements and launches. These establish the scope
of the rejection/lifetime tests; they are not a per-HIP-call or kernel-route trace.

| Public HIP/library call | Current native backend caller | Existing evidence and remaining task |
| --- | --- | --- |
| `hipInit`, `hipGetDeviceCount` | `ps5_backend.cpp::init`; also title startup | Native vocabulary initialization passed. Caller counts/results implemented and host-tested; exact native snapshot passed. |
| `hipMalloc`, `hipFree` | `allocate`, `free_buffer`, allocation-error cleanup | Native vocabulary reports one allocation/free; buffer requests are capped at 64 MiB. Calls/errors host-tested; native successful snapshot passed; cleanup failure remains host-only. Full model GPU residency remains M3. |
| `hipMemcpy` H2D/D2H/D2D | `set`, `get`, `copy` | Native vocabulary validates 16 copied values. Directions and byte counts host-tested; native snapshot passed for H2D 64/D2H 192/D2D 64 requested bytes; no general stride coverage. |
| `hipMemset` | `fill`, `clear` | Native vocabulary validates clear. Both callbacks host-tested; guarded native 16-byte fill and 512-byte clear snapshot passed. |
| `hipDeviceSynchronize` | `sync` | Native storage lifecycle passed. Call/result instrumentation host-tested; exact native snapshot passed. |
| Module load, function lookup, launch, synchronization and unload | No current tensor route in `ps5_backend.cpp::graph` | [M2-DISPATCH](#m2-dispatch): integrate existing genuine public HIP APIs and instrument selected routes. Isolated kernel diagnostics do not establish client calls. |
| BLAS or other math-library calls | No current client math route | M2-MATMUL must select an explicit bounded implementation for attention. No required library call is established by the CPU trace; do not import an entire library or claim no library is needed without that decision. |

### M2-DISPATCH

Implement the shared backend dispatch path using genuine public HIP allocation,
module loading, function lookup, launch and synchronization. Associate every
admitted operation with its exact selected symbol, code-object hash, dtype,
shape/stride restrictions and resource profile. Record launch geometry, dynamic
LDS, call result and completion/lifetime outcome without exposing raw pointers.
Update the coverage table from those host and native receipts. Unsupported
operations, layouts and resource profiles continue to reject before execution.
Keep the mandatory no-fallback boundary as routes become available.

### M2-GATHER

Implement Q4_0 embedding-row selection/dequantization to 2048 F32 values and the
separate F32 output-selection form. Validate actual row IDs and offsets, compare
the full output, and reject out-of-range IDs before dispatch.

### M2-NORM

Integrate the proven normalization/learned-weight kernel into actual graph
execution. Fuse RMS_NORM and MUL only when graph dependencies, aliasing and
intermediate consumers permit it; otherwise provide the separate operation.
Use the actual GPU output for subsequent Q8 quantization and retain independent
end-to-end comparisons.

### M2-MATMUL

Integrate Q8_1 activation quantization and Q4_0 projections for input widths
2048/5632 and output widths 256/2048/5632/32000. Separately implement attention
scores and values using the observed strided F16 cache, F32 inputs/results and
32-to-4 grouped-query mapping. Choose and document their exact upstream kernel
specializations or bounded library calls before recording call coverage.

### M2-VIEWS

Support observed RESHAPE, VIEW and PERMUTE as metadata aliases when valid.
Validate backing ownership, extent, offset, dtype and byte strides, including
the cache's padded 256-to-512 transition. Metadata-only support must not pretend
to execute a kernel or silently materialize a CPU tensor operation.

### M2-COPY

Implement real GPU copies/conversions for DUP and CONT, including observed
noncontiguous attention output. Preserve alias semantics and reject unsupported
layouts; a metadata alias cannot substitute for required materialization.

### M2-KV

Implement mixed-input SET_ROWS separately from K/V cache updates. Convert F32
K/V to the observed F16 layout, including scalar V scatter, and verify untouched
positions and guard regions. Test positions 0, 1, 255, 256 and 511 plus session
reset without reading unwritten cache entries.

### M2-ROPE

Use the locked model's rotation convention and parameters for 32 query heads
and four key heads, each width 64. Compare positions 0, 1, intermediate positions
and 511, then integrate directly with the actual Q/K projection outputs.

### M2-SOFTMAX

Implement stable scale/mask/max/sum/exp behavior for 32 query heads over padded
256/512 cache slots. Future and unwritten positions must contribute nothing;
validate finite outputs and attention normalization against the matched oracle.

### M2-RESIDUAL

Implement the observed 2048-element F32 additions after attention and FFN,
including actual aliasing. Compare each residual and the complete layer output.

### M2-FFN

Implement the observed 5632-element split SWIGLU with correct gate/up operand
ordering. Combine it with M2-MATMUL's gate/up/down projections and M2-RESIDUAL;
validate intermediates and the full layer without CPU tensor fallback.
