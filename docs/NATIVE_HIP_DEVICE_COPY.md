# Native tracked synchronous device copies

Frozen `9628fa6cbb5e1e6e32459e0ee22474ce38e57fec` passed the bounded
`hip_device_copy` workload on console .40, firmware 12.70. Two rounds at
1, 73 and 257 elements across distinct allocations and disjoint ranges of one
allocation used GPU producer → tracked synchronous D2D → GPU consumer.

The 24 launches, 12 copies, 2,648 outputs and 5,926 whole-buffer comparisons
passed, including 630 guards and source preservation. Changed second-round
inputs used the same storage without resetting the copy destination. All 21
allocations were freed before genuine compiler/runtime cleanup and natural stop.

This is a finite visibility workload using CPU copies. DMA, bandwidth,
asynchronous transfers, general cache coherence, framework readiness and full
G4 are unproven. Root title/API integration remains separate.
Exact source, package, deployment, execution and lifecycle authorities are in
[the evidence record](../evidence/native-hip-device-copy.json).
