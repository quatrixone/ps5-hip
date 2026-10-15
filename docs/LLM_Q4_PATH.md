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
LDS0. Fresh source-built selected metadata confirms private0, kernarg192 and 19
explicit arguments, including an 80-byte fusion aggregate and three uint3 values.
Q4 has no hidden arguments; Q8 has thirteen hidden fields in its 328-byte segment. See
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

The hash-guarded client adaptation is implemented below; automated fetching of
the authenticated current header inventory remains separate. Modern GemmEx distinguishes buffer `hipDataType` from arithmetic
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

## Implemented interface preparation

`tools/prepare_llm_hip_interface.py` now verifies the clean locked client and
supplied official header inputs, generates version/export headers, and emits an
isolated VFS overlay plus the exact one-line patch. It never modifies the
upstream checkout. Run with LLVM18.1.8/CMake and a fresh output:

```sh
python3 tools/prepare_llm_hip_interface.py \
  --headers /path/to/authenticated-official-header-inputs \
  --output results/llm-hip-interface
```

The supplied directory must contain the complete official file inventory in
`evidence/offline-llm-hip-compile-probe.json` under
`header_comparisons.current_main.files`; the tool checks each size and hash.
This helper does not fetch those inputs. Its receipt provides the generated
compiler overlay/include flags. Three focused tests and actual Linux, Prospero
and gfx1013 backend syntax checks using the generated files pass; see
`../evidence/offline-llm-hip-interface-preparation.json`.
The helper is not yet connected to the selected-kernel fixture or a runnable
client, and it supplies no BLAS implementation.

## Source-built selected kernels

`make llm-q4-fixture` now builds its RMSNorm/device-library prerequisite and
then the two selected modules from unchanged, clean locked source. See
[SOURCE_ONLY_REBUILD.md](SOURCE_ONLY_REBUILD.md) for fresh output overrides.
The fresh prerequisite Make run and subsequent Q4 target execution passed;
`../evidence/offline-llm-q4-source-fixture.json` records their identities.
Seven focused tests and independent review cover changed bodies, retained
helpers, named types, globals, argument metadata and descriptors. No unqualified
module is published with a success receipt.

Q8 emits one kernel with 328-byte kernargs and no LDS/private storage. Q4 emits
one kernel with 192-byte explicit-only kernargs, 384-byte static LDS and no
private storage. Both have zero undefined symbols. The compiler can externalize
an uninitialized LDS declaration during extraction; only the exact measured
384-byte address-space-3 type/alignment transformation is permitted, and the
linked descriptor is independently checked. This is not a numerical proof.

Next extend the genuine public runtime coherently: 256x1x1 Q8 and 32x4x1 Q4
workgroups; correct group/wave limits and SIMD_DEST_CNTL; distinct explicit-only
and hidden-argument contracts; CLR module admission, attributes and device
Y-dimension reporting. Preserve existing route regressions and reject invalid
combinations before allocating. Then freeze numerical fixtures and validate .40.
