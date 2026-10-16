# Public HIP quantized-operator dispatch

This milestone connects the source-built Q8_1 activation and nonfused Q4_0
matrix/vector modules to the existing Geometry and GeometryLds paths. It does
not execute their ISA on a GPU or qualify LLM inference.

| Contract | Q8_1 activation | Q4_0 matrix/vector |
| --- | --- | --- |
| Local shape | 256 x 1 x 1 | 32 x 4 x 1 |
| Launch dimensions | 1 | 2 |
| Explicit arguments | 9 | 19 |
| Kernarg bytes | 328, including 13 hidden fields | 192, explicit-only |
| Static/dynamic LDS | 0 / 0 | 384 / 0 |
| Private bytes | 0 | 0 |
| Base resource2 | 0x38c | 0xb8c |
| Resource2 after LDS allocation | 0x38c | 0x10b8c |
| Waves per group | 8 | 4 |

The exact measured shape/resource tuple controls admission. Q4 retains XY
work-item coordinates; it is not flattened to 128 x 1. Q8 receives the full
COv5 geometry suffix, while the Q4 entry receives no invented hidden suffix.
Both use the workgroup scheduling bit required for a multiple of four waves.
Existing 64-thread routes and 1024-thread RMSNorm reduction remain separate.

Shape and descriptor validation run before native temporary allocation. The
owned preparation test checks copied arguments after caller mutation, exact
code bytes, all hidden bytes, resource retention, packet registers, invalid
requests and failures allocating either temporary. Null IDs are not retained;
the nonfused Q4 aggregate contains no nonnull nested pointers. This does not
establish fused gate/up ownership.

The CLR adapter uses one explicit profile classifier for module admission and
kernel attributes. It reports the required per-kernel workgroup/static-memory
limits, accepts only the qualified routes, and exposes four threads in the
local Y dimension. Unsupported shapes/resources still fail explicitly.

CLR's available-LDS field includes static storage: it subtracts the kernel's
static bytes when reporting the dynamic allowance. Q4 must therefore advertise
384 total bytes to report zero dynamic bytes. The first public model run caught
the incorrect zero-total value (reported allowance -384); retain this attribute
assertion as a regression alongside the packet and ownership checks.

## Validation and next gate

The source-built modules are in `docs/LLM_Q4_PATH.md`. Core packet and native
owned-memory tests use those actual objects; prior geometry, LDS, scratch and
RMSNorm cases remain regressions. Genuine public CLR qualification additionally
checks both modules, eager/deferred loading, arguments, clean ownership and a
negative packet oracle. These host checks do not emulate GPU arithmetic.

The independent numerical reference is implemented in
[LLM_Q4_REFERENCE.md](LLM_Q4_REFERENCE.md). GPU Q8_1 stores a 36-byte block: half scale, half
sum of original floating-point activations, and 32 signed quantized bytes.
Its `roundf` uses the unrounded floating-point scale. The CPU quantizer's sum
is derived differently, so its bytes cannot serve as the exact GPU oracle.
Q4 consumes that stored sum directly; a naive dot product of dequantized
Q4/Q8 arrays is also not an exact oracle. The reference models the upstream
formula, half conversions and reduction order, with an a priori numerical
budget and corruption controls. Its frozen inputs must remain bound to the
native package before collecting GPU results.

Next build the numerical package with complete raw input/output readbacks,
guards and clean lifecycle checks, review its frozen identities, and validate
on idle .40 under the existing console protocol. No numerical tolerance is
accepted merely because a candidate output fits it.

## Halfword-load diagnostic resource envelope

The first native Q4 comparison failed; see
[the retained failure](../evidence/console40-public-q4-failure.json).
An offline address-rounddown model matches that result exactly. The separate
`probe_llm_q4_alignment.py` experiment preserves the original object and emits
four volatile halfword payload loads in a diagnostic variant. It changes
VGPR usage from 32 to 33 (allocation 32 to 40), so core admission accepts exactly
resource1 `0x60af00c3` or `0x60af00c4` for the existing Q4 ABI. Other modes,
geometry, LDS and argument checks remain unchanged. This does not qualify
GPU arithmetic, the new variant through public CLR, or a new console package.

The packet and owned native preparation tests accept an optional third object,
which must use c4. Set `PS5HIP_Q4_ALIGNMENT_OBJECT` to the diagnostic object
when configuring CMake to register both additional tests. Packet tests exercise
both allowed allocation fields and reject the other 62 values. The native
model checks exact descriptor propagation, owned arguments and cleanup for
both original and diagnostic modules.

The variant also changes cache flags and scheduling. A successful numerical
run alone would not establish alignment as the cause. Prefer a matched raw-load
canary with identical cache flags for causal diagnosis; any workaround run
must retain the frozen numerical oracle and undergo fresh package review.
