# Native tracked memory fills and discovery

Frozen `9bb191ba7ef76188ef6a2e367fd21ce524f7c43d` passed the bounded
`hip_memory_fill` cycle on console .40, firmware 12.70. Single-device discovery
completed before backend initialization. The unchanged `affine_words` kernel
then consumed filled input through six genuine HIP launches.

Two rounds at 1, 73 and 257 elements exercised 24 fills, byte truncation,
negative values and an unaligned three-byte patch. The second round changed
inputs on the same allocations. No H2D copy masked the fills. All 662 active
values, 746 input words and 746 output words matched the independent oracle,
including seven guard words at both ends. Compiler unregister, State destruction
and natural shutdown passed independent lifecycle, service and lock audit.

This qualifies that finite workload. It does not establish DMA copies,
performance, arbitrary memory/cache behavior, general HIP compatibility or full
G4. The title and APIs are integrated following separate reviewed SDK qualification
with 62 passing CMake tests; see [the integration record](../evidence/offline-fill-sdk-integration.json).
Exact observed PID/application/log and source/build/package/deployment/execution
authorities are in [the evidence record](../evidence/native-hip-memory-fill.json).
