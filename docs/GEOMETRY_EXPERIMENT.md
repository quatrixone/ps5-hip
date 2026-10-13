# Measured 2D/3D geometry experiment

This additive development unit starts at a049853 (persistent visibility). It does
not broaden the original initial launch API or qualify multidimensional hardware.
The original initial packets and 64x1x1 hidden packing remain unchanged. Native
candidates use the newer base loader/queue and this explicit new geometry route.

The two full-header HIP kernels retain blockIdx/threadIdx/blockDim/gridDim from
ROCm 6.2.4 with LLVM 18.1.8. Four genuine combined host/device compilations cover
O1/O2 and gfx1013/gfx1030. The bounded workitem library is unchanged upstream OCKL
workitem code, not a complete ROCm device-library implementation. The recipe checks
source archives, generated version, exact bitcode, Clang resource inputs and tool
identities before/after; the selected O2 object/report/source lock is enforced at
build, stage and verification. Recompilation at a different output path changes
the report and requires an explicit lock refresh even if code bytes match.

| Profile | Groups | Threads | Launched slots | Record words | Logical tail | Valid / inactive |
| --- | --- | --- | ---: | ---: | --- | ---: |
| geometry_2d |3x2x1|8x8x1|384|6144|21x13x1|273 /111|
| geometry_3d |2x3x4|8x4x2|1536|24576|13x9x7|819 /717|

Each slot records 16 exact uint32 words: block coordinates, thread coordinates,
observed block sizes, observed grid sizes, global coordinates and input value.
Independent CPU-supplied pitches and fixed measured block constants bound every
write even if a builtin is wrong. No observed dimension authorizes allocation
size. Tail output is the exact integer pattern only for logical points; every
other slot keeps poison. Inputs and the full rounded guards of all three distinct
allocations must remain unchanged. The host callback model uses axis-explicit
loops, not the native oracle helper, and is not ISA/cache emulation.

All eight descriptors have wave32, maxgroup 64, private/LDS 0, properties 0x409,
rsrc2=0x138c, rsrc3/preload 0, zero spill counts and 296 kernarg bytes. The 40 explicit
bytes are three pointers and four uint32 scalars. The 256-byte implicit area uses
the existing strict 13-field COv5 layout: all three group counts and block sizes,
grid_dims 2 or 3, zero remainders/global offsets and zero reserved/padding bytes.
No descriptor bit is rewritten to make it eligible.

0x138c is six user SGPRs (0xc), XYZ workgroup IDs (0x380) and
TIDIG_COMP_CNT=2 (0x1000), which requests XYZ workitem IDs. This is not a packed-TID
mode bit. Pinned LLVM 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff:
SIDefines.h:1093-1104 and AMDHSAKernelDescriptor.h:145-148 define those fields;
AMDGPUAsmPrinter.cpp:904-909 maps 2 to XYZ. SIISelLowering.cpp:2186-2220 allocates
v0/v1/v2 when hasPackedTID is false; gfx1013/gfx1030 lack that feature. Actual
compiled entry ISA consumes v0/v1/v2 and system s6/s7/s8 after kernarg s4:s5.
The register-requirement check is exact, not an arbitrary new allowed-bit mask.
Raw pinned reference excerpts/hashes are in results/geometry-entry-reference.

prepare_geometry_launch accepts only uniform 64-thread 8x8x1 (dim 2) or 8x4x2 (dim 3).
Each axis and total invocation product is checked before deriving geometry;
total launched invocations may not exceed UINT32_MAX. Dim 2 requires groups Z = 1.
Required-workgroup metadata must be absent or match exactly. Scratch/LDS, wave64,
stack, extra SGPR contracts, preload/resource3, spills, unknown/duplicate hidden
fields and malformed layouts reject. The original initial route still rejects
these XYZ entries. Encoding rechecks the complete 256-byte implicit block against
the dispatch shape, including all reserved zeros. Encoding programs all three NUM_THREAD registers and all
three DISPATCH_DIRECT counts, preserving existing cache/resource limits and
wave32 zero-start initiator. No offsets, nonuniform groups or arbitrary block sizes.

The canary submits once with a 2000 ms bound, retains all resources on uncertain
queue outcomes, and reports success only after exact records, tail poison,
unchanged inputs/guards and explicit allocation closure. The existing shell exit
path remains required. Expected failure includes incorrect/missing/duplicate
coordinates, axis transposition, bad dimensions, wrong tail writes, guards,
timeout, faults or uncertain ownership; no automatic retry.

A future native cycle uses the reviewed idle/locks/mounted-byte/fresh-log and
natural process lifecycle protocol. A first O2 pass would qualify only its exact
named grid/block shape and image. O1 needs independent hardware evidence for an
optimization-independent claim. No native execution or broad G3 qualification
is claimed by compilation, packet tests, host callbacks or package validation.
