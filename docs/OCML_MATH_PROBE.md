# Pinned OCML math compile/link probe

`tools/probe_ocml_math.py` reproduces a positive, partial device-library slice
using the already fetched ROCm 6.2.4 inputs and installed LLVM 18.1.8. It builds
OCML and eight configuration constants, then compiles the full-header HIP
`fp32_math` kernel for gfx1013/gfx1030 at O1/O2. It does not build LLVM, CLR or
the full OCKL library, execute a GPU, or select a native application profile.

Run from this checkout under WSL, supplying the existing reference preparation
and an existing `ps5hip-inspect` executable:

```sh
python3 tools/probe_ocml_math.py \
  --reference-root /path/to/reference/results/hip-host-abi-spike/rocm-6.2.4 \
  --output "$PWD/build/ocml-math-v1" \
  --inspector /path/to/reference/build/fat-binary-host/ps5hip-inspect
python3 tests/ocml_math_probe_test.py --fixtures "$PWD/build/ocml-math-v1"
```

The output must be new and inside this checkout's `build` or `results` directory.
Prior successes and failures are never overwritten. No downloads occur. The
reference preparation must include `device-libs-source`, its complete manifest,
the authentic generated `package-stage`, and the previously recorded unmodified
OCKL workitem-source slice. Missing or substituted inputs fail before building.

## Inputs and derived build

`tools/ocml_math_inputs.json` pins the exact source manifest, workitem bitcode and
preparation receipt, complete 71-file header/configuration inventory, and the
fixture source. The source manifest binds 704 original device-library files at
llvm-project commit `1e2c94795ee0d6ab8e2ff3035965a6b74e11b475`, subtree
`773cdc752dbe3099317134dcd2b8ec5d4b086473`. The header origins are HIP
`d3b75fbb82113374e2d6103baa96a9c61425da48` and CLR
`65d174c3e35423bf25c9c369513780c9d9ca760c`.

The recipe copies those inputs before building. Its only upstream source
adaptation changes the obsolete CMake CMP0053 setting from OLD to NEW in the
copy. A small outer CMake project supplies imported installed LLVM tools; it does
not replace OCML implementations or fabricate configuration macros. Upstream's
small `prepare-builtins` utility is built as an OCML dependency. Preserve the
`clang++` invocation name: resolving it to `clang` changes host C++ link behavior.

The selected libraries are OCML, the existing workitem-only bitcode, ABI 500,
the actual target's ISA constant, wave32, finite-only off, unsafe-math off,
DAZ off, and correctly-rounded-sqrt on. Each configuration bitcode is disassembled
and its value checked. Driver traces must contain precisely those libraries in
order and the selected device linker with `--no-undefined`. Compiler math mode
is `-fno-fast-math`. These settings describe the build; they do not establish an
error bound for all functions or inputs.

## What the output proves

The recipe saves each exact command, diagnostics, tool hashes, source inventories,
library hashes, caller IR, direct ELF, disassembly, metadata and loader inspection.
`--no-gpu-bundle-output` makes the expected ELF representation explicit. The
separate prelink IR must contain actual calls and unresolved declarations for
`__ocml_sin_f32`, `__ocml_log_f32` and `__ocml_tanh_f32`. Final objects must have
no undefined symbols or relocations. The fixture also uses fma, sqrt and exp;
their frontend builtin lowering is not evidence of three additional OCML bodies.

All four objects must retain six global result-word stores and actual math ISA,
wave32, no scratch/LDS/spills, and the measured 280-byte kernarg profile with
three explicit arguments and thirteen hidden geometry fields. The existing
bounded loader must report structural initial-profile eligibility. This checks
the code-object boundary, not instruction semantics or hardware support. The
ISA inventory is diagnostic, not a general binary verifier.

The focused tests reject source/header/workitem substitutions, extra or missing
source files, symlinks, duplicate manifest paths, reused output, wrong constants
and missing OCML calls. With a completed output they additionally mutate actual
metadata, loader decisions, ISA, unresolved symbols, relocations and driver
library/linker/target settings. CTest always registers the input checks; set
`PS5HIP_OCML_FIXTURE_DIR` to register the real-artifact checks as well.

## Limits and preserved failures

This is not numerical validation, native host compilation, GPU execution, full
OCKL support, a complete device-library port, or G3/G6 completion. The reused
workitem bitcode is explicitly a source slice, not a successful OCKL build. The
installed LLVM headers/resource tree and static host archives are not fully
inventoried, so this is not a hermetic toolchain package. The
earlier full-library failure and `results/ocml-math-v1` offload-bundle inspection
failure remain historical evidence in the reference checkout. The isolated
recipe's first build likewise preserves its `clang++` invocation-name failure.
None is relabeled as a successful full-library build.

The separately frozen [native math profile](NATIVE_MATH_PROFILE.md) now supplies
a fixed257-input corpus, independent directed MPFR reference, special-value policy
and full output/input/guard checks. Its offline package and numerical checks are
recorded separately. Console qualification remains missing; this compiler recipe
does not establish device accuracy or execution.
