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
| RMS_NORM | llama-graph.cpp build_norm | Diagnostic kernel | Yes, diagnostic | Unweighted diagnostic | No |
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
