# Bounded upstream reduction admission

This offline path admits the unmodified upstream `warp_size_reduction.hip`
kernel at its original 1024x1x1 block size and4096 dynamic LDS bytes. It does
not advertise a general1024-thread capability or establish GPU correctness.

The sample is pinned to rocm-systems d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d
with SHA25642ffe47b3d8dfdd59d2c8ebaabcf7ae6451c97f1055fe9ca6fafebdcc274a57a.
Its absent popcount header is replaced by an explicitly local host-only pair
of32/64-bit overloads. The upstream sample file and kernel remain unchanged.
The original main prints errors but returns success after mismatches, and uses
random masks; its exit status is not an acceptance oracle.

`probe_hip_reduction.py` captures actual Linux/PS5 host and device include
closures with dependency-only compiler invocations before compilation. It binds
selected LLVM18 tools, SDK wrapper/dispatcher/config/revision, prepared HIP
headers and device libraries, rechecking inputs afterward. The saved checker
requires these inventories and rejects omitted artifacts, headers and wrappers.

The ReductionLds route now also supports the unchanged upstream RMSNorm
operator's resource layout. It requires wave32, no private/static LDS storage,
no spills, properties 0x409, fixed rsrc1 mode bits 0x60af0000, and SGPR/VGPR
allocation fields at most 3. Descriptor rsrc2 must be exactly 0x8c or 0x38c.
Explicit arguments use the shared validated scalar/pointer/aggregate layouts;
13 COv5 geometry fields and the dynamic-LDS field occupy the final 256 bytes.
The dynamic size field is at implicit-base + 120, not a fixed module offset.

The old canary still requests 4096 bytes, uses rsrc1 0x60af0080/rsrc2 0x8c,
and has a 288-byte kernarg area. The real RMSNorm case requests 128 bytes,
uses rsrc1 0x60af00c3/rsrc2 0x38c, and has 488 bytes for 26 explicit arguments
plus implicit storage. Existing LDS policy rounds its allocation to 1024 bytes;
the hidden argument remains 128 and the encoded rsrc2 is 0x1038c.

Dynamic requests are now admitted over the existing allocator's nonzero
1..65536-byte range instead of requiring 4096 for every reduction. This is
resource capacity, not algorithmic safety or hardware qualification of every
size. As with ordinary HIP dynamic shared memory, the caller must request the
bytes its kernel actually accesses. Metadata cannot infer that minimum. In
particular, launching the old canary with fewer than 4096 bytes is a caller
error; these changes do not make that invocation safe. Native numerical fixtures
must preserve the selected upstream launcher's exact dynamic size.

Preparation and packet encoding independently validate resource fields, the LDS
plan, 1024x1x1 geometry and all 256 implicit bytes, including reserved zeros.
Unsupported register modes, spills, private/static LDS and other geometry
remain rejected. No public CLR resource-capacity change is included here.

Resource policy comes from pinned PS5_Mesa0b2d6d1a61d9bbf89cf8beb88a696144f67c61f8:
`radv_pipeline_compute.c:38-44` computes32 waves and one CU group with no occupancy
cap; `ac_gpu_info.c:2481` sets SIMD_DEST_CNTL when waves/group is divisible by4.
The gfx10/gfx103 register descriptions place it at bit22, giving
COMPUTE_RESOURCE_LIMITS=0x00400000. LDS allocation4096 produces resource bits
0x40000, combined rsrc2=0x4008c. This is source-derived, not hardware validation.

The actual-codeobject sanitizer test independently checks packet register
addresses/values, all hidden/reserved bytes, max-workgroup/resource admission,
and oversized SGPR/VGPR allocations at selector, direct prepare and encode.
Compiler proof, host packet/model proof and native execution remain separate.
Before a native experiment, require a decisive actual-runtime host oracle and
independently verified deterministic masks/results/guards while preserving the
same kernel and16-block launch. No native package or console test is included.

## RMSNorm preparation evidence

`evidence/offline-rmsnorm-resource-preparation.json` records actual-object packet
and host-mapped preparation tests, old reduction regressions and native library
cross-compilation. Configure `PS5HIP_RMSNORM_OBJECT` with an authenticated
unchanged upstream `norm.hsaco` to register `rmsnorm_packet` and, with SDK
platform headers available, `native_rmsnorm_prepare`. Both tests explicitly
report `gpu_execution=false`; diagnostic argument patterns are not numerical
RMSNorm inputs. Source-only fixture generation and public CLR resource routing
remain separate work before a native numerical experiment.
