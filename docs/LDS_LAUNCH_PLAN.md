# Experimental LDS launch contract

Status: offline implementation and compiler probes. No LDS or barrier execution
has been qualified on a PS5. G2's frozen arithmetic candidate remains separate.

## Allocation and entry state

AMDHSA kernel descriptors leave `COMPUTE_PGM_RSRC2.LDS_SIZE` to the runtime.
Copying the descriptor's zero field does not allocate the static storage reported
by metadata. This route adds the kernel's fixed group-segment size and the caller's
requested dynamic bytes using checked 64-bit arithmetic, rounds to 1,024 bytes,
and encodes the allocation in 512-byte units in bits 15–23 of RSRC2.

The allocation policy follows the pinned PS5 Mesa GFX10.3 path; it is not inferred
from the LLVM `gfx1013` target name. Both total and rounded allocation are limited
to 65,536 bytes per group. These are experimental admission limits, not a hardware
capability query or occupancy measurement. Zero allocation can be planned, but
the explicit LDS launch route rejects it. Encoding recomputes the plan and checks
the field and entry registers before emitting packets.

The supported entry is explicit-argument wave32, 64×1×1 threads, six user SGPRs,
descriptor properties `0x409` and descriptor RSRC2 `0x8c`. Preserve the compiler's
RSRC1, including WGP and memory-ordering mode. Kernargs occupy `s[4:5]`, group X
is in `s6`, and local X is in `v0`. Requested scratch, hidden arguments, dynamic
shared-pointer arguments, preload, additional resource registers and unsupported
launch shapes are rejected. This route does not combine LDS with scratch yet.

The initial and buffer-scratch routes retain their existing restrictions. Metadata
alone cannot identify every LDS access: an extern-shared kernel may report zero
fixed group bytes and no special argument. Such a kernel remains structurally
eligible for the initial route, which is not a certification that it needs no
LDS. The native LDS profiles explicitly select the LDS route and supply capacity.

## Compiler fixtures and exact oracles

`tools/probe_lds_compiler.py` builds `kernels/lds.hip` at O1 and O2 for gfx1013 and
gfx1030 using the pinned LLVM/LLD 18 toolchain. It requires a new output directory,
checks the driver's actual linker, preserves inspection output, and checks genuine
DS reads/writes and barriers. All four objects contain these three entries:

| Entry | Fixed bytes | Native requested dynamic bytes | Rounded allocation | RSRC2 |
| --- | ---: | ---: | ---: | --- |
| `lds_static` | 256 | 0 | 1,024 | `0x1008c` |
| `lds_dynamic` | 0 | 1,280 | 2,048 | `0x2008c` |
| `lds_mixed` | 256 | 1,280 | 2,048 | `0x2008c` |

Each native profile dispatches 257 groups of 64 threads: two waves per group.
Input lane `l` in group `g` is `4099*g + 17*l + (l >= 32 ? 65536 : 0)`.
The exact reduction is `262336*g + 2131424`. Distinct group values detect swapped
group results; the upper wave has a separate contribution. The mixed entry XORs
the sum with `((64*g + 33) ^ 0x53a971bd)`, loaded from a separate static array.
This exercises static and compiler-positioned dynamic storage together.

Dynamic entries use an explicit scalar stride, set to five by the native harness.
The requested capacity is `64 * stride * sizeof(uint32_t)`. Compiler probes and
host launch checks also cover strides one and sixteen. The linked LLVM 18 objects
use no relocations, hidden dynamic-size field or `dynamic_shared_pointer` argument;
their externally declared shared array is addressed through linked LDS offsets.
This does not implement other dynamic-LDS argument conventions.

Every thread reaches every barrier. Host scalar and phase-separated padded-array
oracles agree for all 257 groups and three strides. Those simulations do not model
GPU scheduling, barrier completion or the memory system. Actual-object host tests
check descriptor interpretation, argument packing and emitted allocation fields.

The native harness checks 257 exact outputs, all 16,448 unchanged inputs, and the
entire prefix/suffix of each rounded global allocation. Global guards are not
guards around internal LDS. Resources remain retained through completion and
quarantined on uncertain submission, using the existing queue contract. A
two-second polling deadline cannot cancel a blocked native driver call.

## Evidence and next decision

The research inputs and exact pinned references are retained in the main
workspace's ignored `results/g3-lds-readiness/lds-launch-contract.json`. Its LLVM
documentation source is revision `6da7315`; actual fixture code is separately
produced by LLVM 18.1.8. The original `reduce_lds` O2 fixture and pinned Mesa
register definitions provide additional independent checks of the entry state.

Freeze and inspect the final native candidate before any console use. Hardware
acceptance requires exact results and unchanged global guards/inputs, completed
submission, allocation closure, clean title teardown and service reachability.
Wrong data, a timeout or unresolved ownership fails the experiment and returns
to offline analysis. Dynamic capacity, barriers and LDS reuse across groups remain
unproven until that execution evidence exists. Broader workgroup shapes, mixed
scratch/LDS, generic local pointers and full G3 qualification remain separate work.
