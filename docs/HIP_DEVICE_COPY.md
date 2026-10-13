# Tracked synchronous device copies

`hipMemcpy(..., hipMemcpyDeviceToDevice)` now copies between live ranges owned by
the single-device data registry. It resolves both endpoints before writing and
holds the existing runtime gate across the entire CPU copy. Prior launches are
blocking, so successful return means the copy has completed. No AGC queue or DMA
copy is created. H2D/D2H behavior is unchanged; async, pitched, peer and default-
direction copies are not added.

Both ranges must fit their logical allocations, not native page padding. Interior
and unaligned byte ranges work. Disjoint ranges of the same allocation also work.
Any nonempty overlap, including an identical pointer, is explicitly rejected with
hipErrorInvalidValue before a write. This is a bounded implementation policy for
inputs forbidden by the pinned HIP contract, not a promise of upstream overlap
error precedence or memmove semantics.

The pinned HIP header d3b75fbb82113374e2d6103baa96a9c61425da48,
include/hip/hip_runtime_api.h:3932-3961, states that source and destination must not
overlap. Pinned CLR65d174c3e35423bf25c9c369513780c9d9ca760c,
hipamd/src/hip_memory.cpp:358-395 validates null/range bounds; ihipMemcpy at509
handles zero bytes before pointer validation. Our existing zero-byte policy is
preserved: no endpoint resolution, but gate/lifecycle/poison checks still apply.

Invalid extents/nulls/overlap return InvalidValue; unowned, freed and one-past-end
pointers retain the existing tracked-copy InvalidDevicePointer mapping. CPU/GPU
ownership, poisoned registry, terminal submission and closed lifecycle checks are
shared with the existing copy path. A failed endpoint validation leaves both
buffers unchanged; failed infrastructure unlock is still uncertain as documented
for all runtime calls.

The pinned Prism inventory in HIP_CLIENT_REQUIREMENTS.md:84 requires D2D tensor
transfers. This supplies only the synchronous storage primitive; that backend's
hipMemcpyAsync/2D, streams, libraries and other blockers remain unresolved.

Offline coverage uses registry ASan/UBSan tests, actual runtime/compiler model
ASan/UBSan, and CPU-only TSAN. It checks both endpoint failures, guard bytes,
source immutability, overlap, shared-allocation disjoint ranges, error isolation,
zero bytes, freed pointers, GPU-owned endpoints and fatal/closed lifecycles.
A paused synthetic submit proves a second D2D caller encounters the real gate
busy and copies the completed data only after release. This is not GPU ISA,
console cache visibility, native DMA or performance qualification.
