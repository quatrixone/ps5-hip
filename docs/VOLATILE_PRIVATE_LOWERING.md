# Volatile private-address lowering research

Status: isolated compiler experiment, not an accepted compiler route. Rewriting
proven private pointers in the existing volatile-array fixture produces buffer
scratch instructions while retaining all volatile accesses. **The emitted
descriptor still requests flat-scratch initialization, eight user SGPRs and
144 private bytes.** The initial loader rejects this kernel. Nothing here
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
does not authorize deleting requested ABI state.

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
accesses and atomics. Independently trace why scratch descriptors retain
flat-scratch initialization after optimization; satisfy or validly eliminate that
requirement through compiler analysis, never by forging descriptor bits. Native
scratch allocation, descriptor programming and GPU readback remain separate G3
work. No full compiler build or hardware experiment was performed for this study.
