# Volatile private-address lowering research

Status: isolated compiler experiment, not an accepted compiler route. Rewriting
proven private pointers in the existing volatile-array fixture produces buffer
scratch instructions while retaining all volatile accesses. **Native private
storage remains unimplemented: the descriptor requests eight user SGPRs and
144 private bytes.** Its flat-scratch-init flag reserves two incoming SGPRs;
inspection below shows this kernel does not consume those incoming values or
initialize hardware flat scratch. The initial loader still rejects this kernel. Nothing here
qualifies PS5 scratch execution or changes the frozen indexed-store candidate.

## Source-level cause

The LLVM 18.1.8 optimized fixture already allocates the array in private address
space 5 (AS5), then casts its addresses to generic pointers for volatile accesses.
`InferAddressSpaces.cpp:1009-1023` changes a volatile access only when the target's
`hasVolatileVariant` permits it. AMDGPU supplies no override, so it inherits the
false default in `TargetTransformInfoImpl.h:311`. The TTI API explicitly describes
this as preventing inference for volatile loads/stores. Ordinary accesses do not
hit that guard, explaining the earlier ordinary/volatile contrast.

This is a semantic guard, not merely a missed optimization. LLVM's
[pinned volatile rules](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/docs/LangRef.rst#L3276)
say address spaces may have different invalid-pointer trapping behavior; a
volatile operation's address space cannot generally change without target-specific
context. Removing `volatile` is not a solution.

The pinned TheRock LLVM revision `6da731510c233517c1eeb691d3446d80e9e32525`
retains the same false default and volatile guard, with no AMDGPU override.
That source observation does not substitute for its eventual compiler comparison.

## Bounded experiment and evidence

The input source and original optimized IR are the exact volatile fixture from
`evidence/offline-private-inference.json`. For each target, the experiment:

1. Collected the explicit AS5-to-generic SSA casts.
2. Replaced only the 33 volatile i32 access pointer operands with their original
   AS5 operands. Every pointer had to be in that map.
3. Asserted that all access text and order matched after normalizing only pointer
   types/values: 32 volatile stores and one volatile load remain, with their
   values, alignments and metadata intact.
4. Ran LLVM verification/dead-code elimination, then compiled and linked V5
   objects with the pinned LLVM/LLD 18 tools. Inspected descriptors and opcodes.

The raw script is `results/volatile-private-research/reproduce.py`. This is a
fixture-specific research transformation, not a reusable IR parser or SDK pass.
`evidence/offline-volatile-private-lowering.json` records its hash, source/IR/tool
hashes, pinned source references, all eight object hashes, descriptor fields and
opcode counts. Raw logs, IR, assembly and objects remain in that ignored directory.

Results were the same for gfx1013 and gfx1030:

| Variant | Volatile IR accesses | Private-memory instructions | Flat-scratch setup instructions | Descriptor flat-scratch init |
| --- | --- | --- | --- | --- |
| Original optimized IR | 32 stores + 1 load | 32 flat stores + 1 flat load | Present | 1 |
| Repeat inference, combine, inference | 32 stores + 1 load | 32 flat stores + 1 flat load | Present | 1 |
| Proven-cast rewrite | 32 stores + 1 load, all AS5 | 32 buffer stores + 1 buffer load | Absent | 1 |
| Rewrite, normal O2 optimization and AMDGPU Attributor | 32 stores + 1 load, all AS5 | 32 buffer stores + 1 buffer load | Absent | 1 |

Repeating `infer-address-spaces,instcombine,infer-address-spaces` produced linked
objects byte-identical to the original IR's objects. Pass repetition alone does
not bypass the TTI guard. The bounded follow-up used supported
`default<O2>,amdgpu-attributor`; LLVM 18's AMDGPU target registers that pass name.
It retained the descriptor requirement. No descriptor bits were manually changed.

Both rewritten variants emit RSRC1 `0x60af0044`, RSRC2 `0x91`, properties `0x429`,
RSRC3/preload zero, wave32, kernarg size 16/alignment 8, no LDS and 144 private
bytes. They request the private buffer, kernarg pointer, flat-scratch init,
private-wave offset and group X. The inspector reports that private storage
requires a qualified scratch implementation. Absence of flat-memory instructions
does not authorize deleting requested ABI slots or changing descriptor bits.

## Reserved inputs versus executed flat scratch

The pre-existing `private_explicit` fixture has exactly the rewritten kernel's
resource fields, private size and eight-user-SGPR layout on both targets. It
also emits 32 buffer stores and one buffer load without flat-scratch register
setup. This demonstrates that the requested flat-init SGPR pair is not, by
itself, evidence of unavoidable hardware flat-scratch execution.

The pinned LLVM 18
[SGPR setup order](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/docs/AMDGPUUsage.rst#L5186)
gives these layouts; actual entry instructions confirm their use:

| Input | Rewritten volatile / private_explicit | register_spill |
| --- | --- | --- |
| Private buffer descriptor | s0:s3, used | s0:s3, copied to s8:s11 before use |
| Kernarg pointer | s4:s5, used | s4:s5, used |
| Flat-scratch initialization | s6:s7 reserved; overwritten by kernarg loads before any incoming-value use | Not requested |
| Group X | s8, used | s6, used |
| Scratch wave offset | s9, added to buffer base s0:s1 | s7, added to copied buffer base s8:s9 |
| Local X | v0 | v0 |

In `private_explicit` and the rewrite, the first `s_load_dwordx4 s[4:7], s[4:5]`
loads the input/output pointers over the reserved flat-init pair. Subsequent
uses of s6:s7 refer to that loaded output pointer, not incoming flat-init data.
There are no calls or flat-scratch register accesses in these inspected entries.
By contrast, `private_array` consumes incoming s6:s7, adds s9 and writes
`HW_REG_FLAT_SCR_LO/HI` before its 32 flat stores and one flat load.

`register_spill` requests only six user SGPRs (properties `0x409`, RSRC2 `0x8d`),
212 private bytes and 52 VGPR spills. Its scratch traffic is 52 buffer stores
and 52 buffer loads, without flat-scratch setup. It also emits 64 flat loads and
64 flat stores for the volatile **global input/output** pointers; these are not
evidence of private flat-scratch addressing.

[SIFrameLowering.cpp:700](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/Target/AMDGPU/SIFrameLowering.cpp#L700)
separately decides whether to emit hardware flat-scratch initialization from
actual FLAT_SCR register use, calls and the selected scratch mode. Buffer
descriptor setup follows a separate path. The descriptor request and executed
prologue are therefore distinct. The current rejection means the initial
loader/runtime has no private-storage implementation; it does not establish a
fundamental inability to run these buffer-scratch kernels natively.

## Native dependencies left for G3

Pinned Mesa `0b2d6d1` provides implementation references, not qualification of
this HIP ABI. `radv_emit_compute_scratch` in `radv_queue.c:488` keeps scratch
storage resident, emits the pre-GFX11 base/swizzle state through compute user
data and writes `COMPUTE_TMPRING_SIZE`. `ac_get_scratch_tmpring_size` in
`ac_shader_util.c:1054` encodes aligned bytes per wave and scratch wave capacity,
and requires wave size to remain constant while the backing buffer is in use.
Mesa's two-word graphics-compiler scratch convention is not the complete
four-word HSA buffer descriptor and must not be copied blindly.

A future native path needs the correct wave32 scratch descriptor/swizzle layout,
adequate aligned backing storage and wave capacity, `SCRATCH_EN`/temporary-ring
state, correct hardware-provided wave offsets, all requested SGPR slot positions,
and allocation lifetime through verified completion. Existing PS5 memory/submission
code does not yet supply that contract. No descriptor bits were forged and no
scratch setup was implemented by this study.

## Options and possible adaptation

The supported `-mattr=-flat-for-global` and disabling both alloca promotion paths
leave flat scratch in the original fixture. LLVM 18 does not recognize
`-mattr=-flat-scratch`; it warns and ignores it despite returning zero. That is
not a working flag. These checks found no flag or tested pass-order remedy.

A restricted AMDGPU `GCNTTIImpl::hasVolatileVariant` override is the smallest
plausible inference adaptation. Start with non-atomic scalar i32 loads/stores to
AS5 whose pointers are proven to refer to the local allocation. Preserve volatile
flags, access order and alignment; leave unknown generic pointers, other address
spaces and atomic accesses unchanged. Establish the target-specific trapping and
volatile semantics before broadening this rule. The isolated rewrite only
demonstrates the backend's ability to emit buffer instructions for these accesses.

The next compiler experiment should test that narrow adaptation with the bounded
fixture and negative cases for ambiguous/escaped pointers, global/LDS volatile
accesses and atomics. Preserve requested SGPR slots while implementing and
qualifying the actual buffer-scratch contract; do not equate reserved flat-init
inputs with executed hardware initialization or forge descriptor bits. Native
scratch allocation, descriptor programming and GPU readback remain separate G3
work. No full compiler build or hardware experiment was performed for this study.

## Follow-up: indexed arrays and retained calls

Fresh, unmodified generic-pointer probes are recorded in
[`offline-private-call-followup.json`](../evidence/offline-private-call-followup.json).
LLVM18 O1/O2 on both targets emits buffer-only scratch for ordinary indexed arrays
and inlined address-taken locals: 144 private bytes, eight user SGPRs, properties
0x429 and RSRC2 0x91. Their incoming s6:s7 values are overwritten before use;
group X and wave offset occupy s8 and s9. A separate, identity-pinned eight-slot
buffer-scratch profile is the smallest proposed runtime extension after the
six-slot spill slice. **That extension is not implemented or hardware-qualified.**
The same descriptor fields also occur in volatile entries that actually consume
s6:s7 and initialize flat scratch, so metadata alone cannot admit this profile.

The nested-call probe has buffer stores in its entry, but its two retained callees
perform actual flat private loads/stores and use the callable stack/frame ABI.
Both calls remain in the linked object. All 55 function inventories across five
new linked objects were checked against their actual disassembly. The volatile
inline probe retains 33 stores and two loads at O2; a separate runtime choice
between global and private pointers remains generic. The recognized feature
`-mattr=-enable-flat-scratch` does not remove these flat accesses or initialization.

Completed pinned Clang24 eliminates private storage from the simple ordinary
probes by using registers, and removes conservative flat setup from a global-only
call. It still emits flat scratch for volatile and nested-private probes; its
objects also remain outside the selected loader's metadata contract. These
results require inspecting the full reachable call graph and preserving generic
and volatile semantics. They do not relax G3's requirements for indexed arrays,
address-taken locals and actual nested calls, or repeat the earlier IR rewrite.
