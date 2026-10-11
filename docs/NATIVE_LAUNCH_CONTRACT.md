# Initial native launch contract

Status: host preparation and packet-fragment tests pass; native memory code
cross-builds with mocked cleanup tests. Submission ownership and the AGC adapter
also cross-compile with model tests. Initial cache/state commands and a dedicated
canary title now link; packaging/runtime and GPU execution remain unverified.
This is not a passed gate.

## Source provenance

The native boundary was inspected in a clean local checkout of
[mihawk-99/PS5_Mesa](https://github.com/mihawk-99/PS5_Mesa) at
`0b2d6d1a61d9bbf89cf8beb88a696144f67c61f8`. This is newer than the initial research
pin in RESEARCH_BASELINE.md. It is a source reference, not a build-qualified
dependency lock. No platform implementation is vendored or linked yet.

Relevant paths within that revision:

- `src/amd/vulkan/winsys/ps5/radv_ps5_platform.[ch]`: direct memory, address ranges,
  CPU cache operations, AGC initialization and native submission; MIT, Mihawk.
- `src/amd/vulkan/winsys/ps5/radv_ps5_cs.c`: completion packet and queue ownership;
  MIT, Mihawk.
- `src/amd/vulkan/radv_cmd_buffer.c`: compute program/resource registers, thread
  dimensions, user data and dispatch emission.
- `src/amd/vulkan/radv_pipeline_compute.c` and `src/amd/common/ac_gpu_info.c`:
  compute resource-limit calculation.
- `src/amd/registers/gfx10.json`, `src/amd/common/sid.h` and
  `src/amd/common/ac_cmdbuf.h`: register fields and packet format.
- `src/amd/common/ac_cmdbuf_cp.c`: GFX10 ACQUIRE_MEM encoding.

The raw compute path in PS5_Vulkan `src/diagnostics.cpp`, pinned at
`3f3ee69607013b345d2baa6d6a37c86745649a08`, additionally supplies the observed
destination-mask, accumulator and CS_PARTIAL_FLUSH ordering reference.

The implementation here encodes the inspected register/packet contract. Any
subsequent reuse of source files must preserve their original license notices
and lock the complete native build inputs.

## Implemented host boundary

`prepare_initial_launch` consumes an inspected code object, a kernel name, live
GPU allocation extents, a one-dimensional group count and argument values.
The initial profile uses 64x1x1 workgroups and wave32. It rejects unsupported
scratch/LDS requirements, insufficient maximum workgroup size, conflicting
compiler-required workgroup shapes, zero/overflowing group counts, invalid
addresses, undersized storage, bad alignment and argument count/size mismatches.

Shader allocations must lie wholly within `[0x200000000, 0x300000000)` and satisfy
ELF load alignment. Full kernel argument/data pointers must lie below `2^47`.
Code and kernarg allocations may not overlap; argument buffers may alias one
another but may not alias either launch allocation. Scalar values are raw bits;
32-bit scalar overflow is rejected. Argument storage is zero-filled, then packed
at metadata offsets in little-endian order.

The native allocator must supply true live extents. These host checks do not
prove mapping ownership or that a kernel's computed accesses fit its buffers.
The canary harness must independently size buffers from count and element width.

For the current arithmetic canaries, private-buffer slots s0:s3 are reserved and
zeroed, the full kernarg pointer occupies s4:s5, and group ID X follows at s6.
Group IDs are system-generated; they are not written into the user-SGPR packet.
The two-user-SGPR layout is also represented when the descriptor omits the four
private-buffer slots. Neither layout has been run here on hardware.

## Packet fragment

`encode_initial_dispatch` emits GFX10 ACQUIRE_MEM with GCR `0x4381` (instruction,
scalar, vector, L1 and L2 invalidation), PFP_SYNC_ME, then these state writes and
DISPATCH_DIRECT. This assumes CPU upload flushes and no unretired earlier work.

| Register | Initial value |
| --- | --- |
| COMPUTE_STATIC_THREAD_MGMT_SE0/1 | `0xffffffff` |
| COMPUTE_STATIC_THREAD_MGMT_SE2/3 | Zero |
| COMPUTE_USER_ACCUM_0..3 | Zero |
| COMPUTE_PGM_LO/HI | Full entry address shifted by 8, split into register fields |
| COMPUTE_PGM_RSRC1/2 | Inspected descriptor values |
| COMPUTE_PGM_RSRC3 | Zero in this profile |
| COMPUTE_RESOURCE_LIMITS | Zero for two waves per group, no occupancy cap, one CU group |
| COMPUTE_NUM_THREAD_X/Y/Z | 64, 1, 1 |
| COMPUTE_USER_DATA | Reserved zeros and kernarg address according to descriptor |
| DISPATCH_DIRECT | Group count X, 1, 1; enable, force zero start, wave32 |

The initial dispatch uses in-order mode. Optional out-of-order and tunneling
bits used by Mesa are not enabled. Program high bits are explicitly written to
avoid inheriting an old value. Tests decode packet lengths and register addresses
independently and check the resulting semantics for both real compiler fixtures.

**This fragment must never be submitted alone.** It still needs native AGC
initialization, CPU flushes, completion and allocation lifetime management.
It contains 56 dwords for the current six-user-SGPR canaries.

## Remaining native work

The platform source initializes AGC inside the title process and submits PM4
through `sceAgcDriverSubmitDcb`. Its host model completes submissions without GPU
execution, so model completion cannot validate kernel results. The queue owns
command storage until completion; application storage must not be handed to a
GPU indirect buffer based solely on the host model working.

Before the first native launch, finish qualifying:

1. Startup/runtime and packaging for the linked title in [Native build](NATIVE_BUILD.md).
2. The combined cache, compute-state and completion sequence against inspected ISA.
3. External bounded control and reliable capture of failure/quarantine output.
4. The development-title identity and frozen source, tool and binary hashes.

The existing driver uses a RELEASE_MEM completion packet with event 20, index 5,
GCR actions `0x30c`, and a 32-bit memory marker. The submission queue now appends
CS_PARTIAL_FLUSH (event 7, index 4) before this event, ten words total. Packet-model
checks validate the encoding; they cannot prove hardware completion ordering.
