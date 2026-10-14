# Public HIP allocation bridge (offline)

This candidate combines the genuine upstream HIP stream/event admission with
the separately qualified ROCclr SVM allocation patch. The added model calls
the exported `hipMalloc` and `hipFree` entry points, checks backing allocation
and retirement through the PS5 device model, and rejects an interior-pointer
free and a second free. The same ten-case closure retains the earlier stream,
event, reset-rejection and failed-join checks.

`tools/compose_hip_memory.py` verifies the exact admission source and both
follow-on patch hashes before producing `results/source-v3`. The qualifier
binds the composed source, compiler-discovered headers, tools and link inputs.
Fresh full receipts are under `.local/g5-public-hip-bridge/results/asan-memory-v5`
and `.local/g5-public-hip-bridge/results/tsan-memory-v3`, each covering 69
upstream units and ten modes. Independent review is recorded in
`results/public-memory-bridge-full-review-v1/review.json`.
TSan leak-report symbolization stalled intermittently during the intentionally
failed-join cases. The TSan harness disables symbolization but still requires
exit 66, exactly one thread-leak warning, the finished unjoined worker line,
and no data-race warning for each such case. Both failed joins and all other
modes passed in the fresh closure.

The transport is synthetic and this is a Linux model. It does not prove native
public HIP startup, GPU execution, module loading, launch, copied input
consumption, graphics interop, or G5 acceptance. The SVM allocation profile
remains one device, zero flags and at most 64 MiB.
