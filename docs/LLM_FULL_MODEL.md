# Full-model execution prerequisite

The selected quantized first layer is native-qualified. Full-model GPU execution
is not implemented yet. The next path must execute the original llama scheduler
graph successfully, so llama retains its actual KV position state; returning
success after a diagnostic cloned layer is not sufficient.

## Reproduce the CPU contract

`make llm-full-graph-trace` prepares or authenticates the CPU model prerequisites
and records 514 successful teacher-forced requests: positions 0–511, then reset
and positions 0–1. Seven checkpoints include complete graph metadata, stable
tensor/source IDs, compute flags, actual input controls and all 32,000 logits.
Reset logits must match exactly. Use `LLM_FULL_GRAPH_MODEL`,
`LLM_CPU_BUILD_DIR` and `LLM_FULL_GRAPH_TRACE_DIR` to select existing inputs.
An existing receipt is authenticated, never silently refreshed after changes.

`make llm-full-graph-admission` then compiles the sanitized metadata test against
those graphs. Choose a fresh `LLM_FULL_GRAPH_ADMISSION_DIR` for each test run.
The selector checks all 22 blocks, source links, weight and cache identities,
shapes, strides, op parameters, view owners and the output head. Ten mutation
classes must reject. This is metadata admission only: it does not authorize a
GPU launch or establish actual device ownership.

## Observed integration constraints

The actual graph has 801 nodes and 1,056 tensors. Nodes 1–3 are an inactive
mixed-input branch without `GGML_TENSOR_FLAG_COMPUTE`; their storage can contain
reused activation bytes. Validate their structure and skip their computation.
Never interpret their inactive indices as live SET_ROWS controls. The remaining
798 nodes are active. The last block includes two F32 GET_ROWS operations using
`out_ids=0`, followed by final normalization and a 32,000-row Q4 output projection.

Configure the backend instances that llama creates, not only the separate
diagnostic handle. Full layer offload leaves the embedding table on CPU in the
pinned loader; use its supported tensor-buffer override for that table and
verify all actual weight owners. Retain the 64 MiB limit and upstream allocation
at tensor boundaries. Measure cumulative residency and cleanup before claiming
model-scale memory qualification.

The scheduler may reuse activation storage and pass single-node graph views to
an evaluation callback. Validate the complete graph before dispatch, then bind
each ordered split to that validated request. Full-mode MUL must multiply its
direct normalized input by its weight; the diagnostic weighted-RMS helper
recomputes from an earlier activation that may already have been overwritten.
Advance request validation state only after the real scheduler succeeds and all
required active nodes complete. Never accept a CPU-selected tensor computation.

Use llama's real token, position, KV indices, mask and output-index setters.
Reset must synchronize, clear actual llama memory and reset validation state.
Compare layer boundaries and complete logits before adding sampling. GPU
full-model execution, model-scale memory, generation, the dense floating-point
layer fixture, near-zero RMS and native extreme-input tests remain open.

## CPU numerical reference

`tools/build_llm_full_model_reference.py` records all 22 layer residuals, final
weighted normalization and 32,000 logits at positions 0, 1, 255, 256 and 511,
then reset positions 0 and 1. It executes the real CPU decoder for all 514
requests and checks committed KV positions. Its completed-node callback reads
each selected output before the scheduler can reuse its memory. CPU fusion is
disabled, and the existing CPU libraries and model are reused. The 168 files
have an exact inventory; all 24 reset boundaries must match bitwise.
Run the tool with `--cpu-build`, `--model` and a fresh `--output` under results.
`authenticate_full_model_reference` verifies the saved inputs and outputs.
This supplies the full-model comparison oracle; it does not establish GPU
execution or numerical agreement.
