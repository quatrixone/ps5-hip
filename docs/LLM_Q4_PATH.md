# Next inference gate: Q4_0 multiplication and activation preparation

This is an offline integration checkpoint, not native operator acceptance.
The six-case RMSNorm hardware gate passed; the next useful path is activation
quantization followed by real Q4_0 matrix/vector multiplication.

## Activation compilation

Unchanged upstream `quantize.cu` fails a whole-module device link with the
existing authenticated libraries. Unsupported-format `no_device_code` bodies
import `__ockl_printf_begin`, `__ockl_printf_append_string_n` and
`__ockl_printf_append_args`. The required Q8_1 kernel has no such call.

A diagnostic compilation of the same unchanged source to LLVM bitcode, followed
by `llvm-extract-18 --recursive --func=<exact Q8_1 symbol>`, `llc-18` for gfx1013
and `ld.lld-18 -shared`, produces one kernel with zero undefined symbols.
The selected kernel's LLVM instruction body is byte-identical before and after
extraction. Metadata reports wave32, maximum workgroup 256, 328 kernarg bytes,
and zero static LDS/private segment. See
`../evidence/offline-llm-q8-selected-link.json`.

This is a promising explicit supported-kernel build strategy, not permission to
pretend missing runtime services exist. Integrate it into authenticated source-only
Make tooling, audit its ABI/ISA and transitive dependencies, and ensure host
registration cannot expose omitted kernels. It currently reuses diagnostic
HIP6.2/hipBLAS2.2 inputs; it does not resolve release dependency selection.
Validate Q8_1 output against its actual GPU block layout/rounding and verify
Q4_0 multiplication against the matching quantized reference. CPU activation
quantization is not a substitute for final GPU inference acceptance.

## Multiplication resource contract

The locked client selects the generic nonfused Q4_0 batch-one specialization on
gfx1013: block `(32,4,1)`, grid `(output_rows,1,1)`, static LDS384 and dynamic
LDS0. Historical compiled metadata records private0, kernarg192 and 19 explicit
arguments, including an 80-byte fusion aggregate and three uint3 values. Rebuild
before using those historical descriptor values in a new candidate. See
`../evidence/offline-llm-q4-mmvq-inventory.json` for exact source locations,
arguments and actual GGUF tensor shapes.

Do not flatten the workgroup: the kernel uses threadIdx.y. The selected nonfused
gate has null pointers inside its fusion aggregate. Full-model fused gate/up
routing may use nonnull nested pointers and needs corresponding lifetime support;
passing the nonfused gate does not prove that path.

## Current hipBLAS interface

With existing HIP6.2 and official hipBLAS3.8 snapshot headers, the client chooses
a deleted legacy datatype branch using HIP_VERSION alone. An isolated one-line
adaptation selecting the existing modern branch when `hipblasVersionMajor >= 3`
(or the original HIP-version condition) passes actual backend syntax for Linux
host, Prospero host and gfx1013 device. The unmodified baseline fails.
See `../evidence/offline-llm-current-hipblas-interface.json` for exact source,
header, tool and command identities. No version spoof or success stub was added.

Next implement a hash-guarded client adaptation and authenticated current header
recipe. Modern GemmEx distinguishes buffer `hipDataType` from arithmetic
`hipblasComputeType_t`; do not reuse legacy enum numeric values. Handle creation,
destruction, stream ownership, SGEMM and the three imported GemmEx variants need
real implementations before final linking. The pre-existing client math-mode
success macro is not validated support.

## Work order

1. Reproduce the selected Q8_1 and exact nonfused Q4_0 kernels with source-only
   tooling, coherent headers, complete metadata checks and explicit exclusions.
2. Qualify genuine public HIP resource routing for their measured workgroup/LDS
   contracts; do not widen advertised capabilities without implementing them.
3. Freeze independent numerical references, then validate quantization and
   multiplication on .40 under the normal candidate protocol.
4. Complete remaining client operators and linking, integrate a transformer
   layer, then full-model prefill/decode and release qualification.

No console was contacted for this checkpoint. No full model, performance or
training claim follows from these compiler results.
