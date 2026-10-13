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

The separate ReductionLds route requires wave32, no private/static LDS storage,
no spills, exact measured rsrc1=0x60af0080, X-only rsrc2=0x8c, properties0x409,
and a288-byte argument area: three64-bit buffers, size_t,13 COv5 geometry fields
and dynamic-LDS size at offset152. Existing routes retain their rejection rules.
The preparation and packet boundaries separately enforce1024x1x1/4096 and
check the complete implicit argument area, including reserved zeros.

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
