# Experimental buffer scratch launch

This is an offline G3 implementation, not scratch hardware qualification or G3
completion. The existing scratch-free admission and packet API remain unchanged.
The separate buffer-scratch API accepts wave32, 64x1x1 threads, X groups, explicit
arguments, six user SGPRs, fixed private storage, no LDS, no flat initialization,
no dynamic stack, no preload/resource3 and no extra entry state. Inspection still
does not validate arbitrary ISA. The native profile remains the inspected,
LLVM18-compiled `register_spill` fixture, not arbitrary client kernels.

## ABI and planning

`register_spill` in `kernels/resources.hip` has 212 private bytes per work-item and
52 VGPR spills in the retained **O2** gfx1013/gfx1030 resource objects. Its disassembly
contains 52 buffer stores and 52 buffer reloads. The prologue copies the full V#
from s0:s3 and adds the incoming wave offset in s7; kernarg occupies s4:s5,
workgroup X s6, and local X v0. Flat global input/output loads/stores are distinct
from scratch operations. This entry uses properties 0x409 and RSRC2 0x8d.

`buffer_scratch_ring_bytes` uses checked 64-bit arithmetic to multiply private
bytes by 32 lanes and round up to a 1024-byte wave stride. It checks the 13-bit
WAVESIZE field and 32-bit descriptor extent before returning allocation size.
`plan_buffer_scratch` additionally checks the supplied GPU range, 4096-byte base
alignment and backing capacity. It does not allocate or establish GPU mappings.

The fixed capacity is 1280 waves. This follows pinned Mesa's conservative PS5
policy: its provisional topology has two shader engines, two shader arrays per
engine and 36 active CUs; `max_good_cu_per_sa` rounds to ten, then scratch capacity
is 32*10*2*2. These counts are explicitly provisional in that source, not measured
by ps5-hip. We do not reduce backing to the number of waves in one dispatch.
For the retained fixture the stride is 7168 bytes, ring payload 9175040 bytes and
COMPUTE_TMPRING_SIZE (0xb860) 0x7500 (1280 waves, seven KiB per wave).

Descriptor words are base low32, base high16 with SWIZZLE_ENABLE, ring payload
bytes, then 0x21c14fac. Word3 has XYZW selectors, UINT32 format 20,
INDEX_STRIDE=2, ADD_TID=1, RESOURCE_LEVEL=1 and OOB_SELECT=2. ROCr's descriptor
constructor leaves INDEX_STRIDE for CP; our direct packet path completes it using
ACO's wave32 value. OOB mode 2 only tests a zero extent: it does **not** protect
allocation boundaries. Host validation, whole-allocation guards and bounded
qualification remain necessary.

Preparation rejects scratch overlap with module, kernargs or pointer argument
extents. Both preparation and encoding recompute/check the plan to reject mutated
fields. The encoder writes the ring and all four V# words, retaining the existing
cache prefix and dispatch packet. The scratch-free encoder still rejects SCRATCH_EN.

## Ownership and validation

The native harness must pass the backing allocation in `SubmissionQueue` resources.
The pure planner cannot prove allocation ownership. The existing queue flushes
uploads, retains all resources before entering submit, drains compute before its
completion event, then flushes retained ranges before CPU inspection. Failure or
timeout leaves scratch quarantined with other resources; no resize/release/reuse is
safe until that ownership is resolved. Ring stride cannot change while it is in use.

The native fixture uses one 64-thread group, 4096 distinct input/output words,
a 64-KiB aligned prefix before the scratch payload, and checks the entire remaining
suffix and input/output guards. The payload starts poisoned, must show activity,
and is not compared against a guessed internal spill layout. Exact outputs detect
cross-lane aliasing. This harness prepares a later qualification; it has not run.

`buffer_scratch_model` runs without external files. Two real-object CTests run the
same preparation/independent packet-field checks on the retained gfx1013/gfx1030
objects. Negative tests cover zero/overflow/undersized/misaligned ranges, aliases,
mutated descriptors/ring fields, mismatched private size, wave64, flat initialization,
dynamic stack, extra entry SGPRs, hidden arguments, LDS and wrong group shapes.
The submission model uses actual scratch-sized native allocations for two successful
rounds and second-round submit/kick/timeout failures, including refusal to close
scratch while GPU ownership is held. Host mocks do not emulate GPU scratch addressing.

## Source provenance and remaining proof

- PS5 Mesa `0b2d6d1a61d9bbf89cf8beb88a696144f67c61f8`:
  `src/amd/vulkan/radv_queue.c:488-527,890-895` (state/allocation),
  `src/amd/common/ac_shader_util.c:1039-1079` (granularity/lifetime),
  `src/amd/compiler/aco_ir.cpp:1908-1965` (wave stride and V#),
  `src/amd/common/ac_descriptors.c:811-885` (packing/OOB semantics),
  `src/amd/registers/gfx10.json` COMPUTE_TMPRING_SIZE,
  `src/amd/vulkan/winsys/ps5/radv_ps5_winsys.c:140-177,389-394` and
  `src/amd/common/ac_gpu_info.c:1302-1305` (provisional capacity).
- ROCr 6.2.4 `df7549038b458c9387a2c6ea8d9328e3c9e6620c`:
  `src/core/runtime/amd_aql_queue.cpp:1575-1726` (descriptor/ring), SHA256
  `d324bc96405ee3851a93d0c95530bc949c40595dd622aeacb1338584f2193351`;
  `src/core/inc/registers.h:103-118,169-185,259-292` (bitfields), SHA256
  `bf11347c0c2eb744a39040f087d79b84e701f9d875c4bff1b8685aa91cd093c4`.
- PS5 Vulkan `3f3ee69607013b345d2baa6d6a37c86745649a08`,
  `docs/HARDWARE_FINDINGS.md:3454-3465`: buffer-scratch graphics shaders worked
  where flat scratch initialization faulted. This is separate evidence from our
  compute/HSA entry contract.
- LLVM `llvmorg-18.1.8`, `llvm/docs/AMDGPUUsage.rst`, initial kernel execution
  state and private segment buffer ABI; local exact fixture disassembly and
  `evidence/offline-volatile-private-lowering.json` provide the emitted counterpart.

Unproven: AGC preservation of our compute ring state, SPI's expected s7 offset,
PS5 execution of this full HSA V#, scratch cache visibility and safety under
multiple groups/repeated dispatches. No console run, performance claim or general
HIP private-storage support follows. Address-taken locals, indexed arrays, nested
calls and generic-address behavior remain mandatory G3 work after this first slice.

## Validation of this implementation

The isolated LLVM18 ASan/UBSan host build passed all 30 registered CTests,
including the completed Python profile/log/staging integration, before the O1
extension changed only the scratch test. After that extension, the rebuilt
sanitizer test passed its model, both O2 objects and both O1 objects; three wrong
profile/flag cases were rejected. The complete 30-test suite was not repeated
for this test-only change. Initial receipt and raw logs remain preserved. The native harness,
code-object parser and launch implementation each compile to a PS5 x86-64 object
for both gfx1013/gfx1030 target macros using the actual SDK and elevation headers
(six objects). These macros select the device fixture; they do not make the host
objects AMDGPU binaries. No new native link, import-provider check, SELF conversion,
package or console execution was performed. Unrestricted Python unittest discovery
requires an unrelated absent native-runtime artifact; the affected 39-test Python
suite and registered CTests are the supported evidence here.

## Independent O1 compiler and planner check

Fresh O1 objects for both targets use the exact pinned LLVM18 baseline compile
arguments, changing only `-O2` to `-O1` and the output destination. Source bytes,
all other flags and the actual device linker remain verified. Both entries emit
177 buffer stores and 177 buffer reloads, with 712 private bytes and 177 reported
VGPR spills. Neither sets up flat scratch or requests flat-scratch initialization.
Their 64 flat global input loads and 64 output stores remain separate operations.
Compared with O2, the descriptor changes only its private-byte requirement;
wave32, six user SGPRs and the remaining resource fields stay the same.

The original test initially was not run on O1 because its expected fixture was
212 private bytes/52 spills. This was a test-profile limitation, not compiler
incompatibility. `buffer_scratch_test object.hsaco --o1-profile` now explicitly
selects the measured 712-byte/177-spill fixture, a 23552-byte wave stride,
30146560-byte ring and `COMPUTE_TMPRING_SIZE=0x17500`. The same independent
descriptor/packet checks and malformed-plan, overlap and entry-state rejection
checks run for both profiles. The default synthetic/O2 behavior is unchanged;
crossed or unknown profiles reject.

Only the test executable was rebuilt with ASan/UBSan for this follow-up. The
default model, both O2 objects and both O1 objects pass; three wrong-profile
checks reject. The original mismatch observation and subsequent results are
retained in [`offline-buffer-scratch-o1.json`](../evidence/offline-buffer-scratch-o1.json).
Core admission and the native harness were not changed. The native candidate
remains fixed to O2's 212-byte/52-spill profile and 9175040-byte ring. O1 provides
additional offline lowering/planner evidence; no O1 title or GPU execution is
qualified, and G3's private-storage requirements remain open.
