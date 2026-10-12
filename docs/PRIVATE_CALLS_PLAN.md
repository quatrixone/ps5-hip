# Experimental retained private-memory calls

This branch prepares an explicit, unqualified flat-scratch launch route offline.
It does not enable general HIP calls or establish that PS5 private-address
translation works. Existing initial, buffer-scratch and LDS routes keep their
entry restrictions. No native title for this route is ready to execute.

## Source and observable behavior

`kernels/private_calls.hip` retains the generic-pointer source forms from the
private-storage investigation. `nested_private` calls `call_middle`, which calls
`call_leaf`; both callees have `noinline`. The leaf modifies an inner array
element and returns a value. The middle function reads that element again after
the call, before its frame ends. This last read is essential: checking only a
value forwarded through a register would not detect a missing memory store.

`tools/probe_private_calls.py --output build/private-calls` requires a fresh
output directory and the pinned LLVM 18 tools. It compiles O1/O2 for gfx1013 and
gfx1030, retains driver traces, linked code, assembly, IR and inspection output,
and checks compiler/library identities before and after compilation. Its audit
resolves both direct PC-relative call targets in the linked object, checks
retained private accesses and the post-call reload, and rejects changed frame,
descriptor or noinline contracts. These are fixture checks, not a general ISA
validator. Other entries are useful negative/contrast fixtures; their presence
does not qualify their call graphs.

For 257 groups of 64 threads, input `x = global_id * 4099 + 17` gives 16,448
distinct inputs and outputs. The exact unsigned result is:

```
(4*x + 5*(x & 31) + 2*((x + 7) & 31) + 6) modulo 2^32
```

An independent array model matches that formula, including 96 index/wrap boundary
inputs. Suppressing the leaf's store while retaining its updated register result
changes the checked output by three. This is a CPU mutation model, not fault
injection into an actual GPU. The compiler-contract test also mutates saved
instructions, call displacements, IR attributes and descriptors and requires
rejection.

## Runtime versus compiler responsibilities

The inspected wave32 entry requests properties `0x429`, RSRC2 `0x91`, eight user
SGPRs and 288 fixed private bytes per lane. Its 64x1x1 group has two waves.

| State | Provider and meaning |
| --- | --- |
| s0:s3 | Runtime: existing four-word wave32 scratch descriptor |
| s4:s5 | Runtime: kernarg pointer |
| s6:s7 | Runtime: raw 64-bit address of the same scratch backing |
| s8 | SPI: workgroup X |
| s9 | SPI: scratch wave offset in bytes |
| v0 | SPI: local X |
| Flat scratch registers | Compiler entry adds the wave offset to the raw base and writes both registers |
| s32/s33 and return addresses | Compiler: stack/frame and retained-call ABI |
| Private aperture | Platform prerequisite, currently unproven for this route |

The high word of s6:s7 must not contain the scratch descriptor's control bits.
It is not the legacy per-lane size field. Runtime code must not program s32/s33 as
user data, invent a wave offset, remove flat-init descriptor bits, or initialize
return addresses itself.

The compiler's 144-byte entry frame and 144-byte middle frame are already summed
in the 288-byte metadata requirement; the leaf has no frame. With the inherited
conservative 1,280-wave policy, the rounded wave stride is 9,216 bytes, backing
payload is 11,796,480 bytes, and TMPRING is `0x9500`. This capacity policy comes
from the pinned PS5 Mesa investigation; it is not an occupancy measurement.
Indirect calls, recursion, dynamic stacks and arbitrary call graphs require
separate analysis. No cumulative frame size may be guessed from only the entry.

The experimental host API prepares packets and checks ranges, layout and scratch
capacity. It cannot infer shader safety from metadata, establish the private
aperture, or prove that an allocation stays resident. A future native caller must
bind the audited source and whole linked module and retain every referenced
allocation through completion, including uncertain submission and quarantine.

## Hardware prerequisites and alternate compiler route

Earlier pinned PS5 graphics work observed faults around flat-scratch setup and
disabled that path. This is adverse evidence, not proof that every compute use
fails or that the cause is understood. Correct eight-word user data alone does
not resolve the separate private-aperture requirement used by generic pointers.

Before considering an actual private access, prepare a diagnostic that only
reports the 64-bit private-aperture special operands with ordinary global stores.
Inspect it for absence of scratch, flat setup and private accesses. Any later
hardware experiment needs its own frozen candidate, bounded lifecycle and clear
failure decision; reading plausible aperture values alone would not prove
translation or flat-register support. Do not automatically retry a fault.

If platform flat state cannot support this ABI, investigate address-space-aware
callee specialization and backend lowering. Such a route must keep real calls,
observable stores, volatile semantics and a correct generic fallback for mixed
pointers. Inlining the fixture or accepting only ordinary buffer-only arrays
would not satisfy the retained private-call requirement.

G3 remains incomplete until these requirements execute correctly on hardware,
with validated outputs, guards, ownership, visibility and teardown. The broader
compute and HIP application gates remain unchanged.
