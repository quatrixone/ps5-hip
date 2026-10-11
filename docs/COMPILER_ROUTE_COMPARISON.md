# Compiler route for native bring-up

Use the pinned standalone LLVM 18.1.8 route for G1/G2. It already produces
inspectable AMDHSA V5 objects for both candidate targets, including arithmetic,
reduction and resource probes. Separate-directory rebuilds reproduce the current
objects exactly on the current host. None of this proves PS5 GPU execution.

Keep TheRock as the upstream component-build route to evaluate for device
libraries, compiler evolution and later framework work. It cannot supply the PS5
host runtime, memory ownership or native submission contract automatically.

## Pinned source comparison

The inspected [TheRock revision](https://github.com/ROCm/TheRock/tree/4d1b9e560fa23ea2340fcf17b423bb281ab29aae)
pins ROCm/llvm-project at `6da731510c233517c1eeb691d3446d80e9e32525`.
This is a different compiler source identity from the installed Ubuntu LLVM 18.1.8;
similar command-line options are not proof of equivalent generated code.

| Question | Standalone route here | Pinned TheRock route |
| --- | --- | --- |
| Device compiler | Installed Clang/LLD 18.1.8, exact version checked | Pinned AMD Clang/LLD 24; configured compiler component built, staged and assembled |
| Target configuration | Both gfx1013/gfx1030 fixtures compile | gfx1030 declared; gfx1013 needs a custom declaration |
| Component scope | Device-only HIP, without headers/device libraries | Compiler project plus bundled runtime/device-library components |
| Artifact evidence | V5 objects, ISA inventories, metadata and host parser tests | Four V5 fixtures compared; new target spelling rejected by current loader |
| Native PS5 host code | Separate pinned PS5 SDK and native converter | Still requires a distinct PS5 port/integration |
| GPU qualification | Named gfx1013 store/vector fixtures passed; broader capabilities unqualified | Pending |

The [compiler pre-hook](https://github.com/ROCm/TheRock/blob/4d1b9e560fa23ea2340fcf17b423bb281ab29aae/compiler/pre_hook_amd-llvm.cmake)
configures AMDGPU, Native and SPIRV backends. Its non-Windows path enables a wider
project/runtime set, and includes device-library and SPIR-V translator projects.
The [compiler declaration](https://github.com/ROCm/TheRock/blob/4d1b9e560fa23ea2340fcf17b423bb281ab29aae/compiler/CMakeLists.txt)
also ties in ROCm build dependencies. Selecting its compiler component is therefore
different from invoking the already-installed standalone Clang on one HIP file.

## Configuration experiment

The exact upstream [target configuration](https://github.com/ROCm/TheRock/blob/4d1b9e560fa23ea2340fcf17b423bb281ab29aae/cmake/therock_amdgpu_targets.cmake)
was evaluated with CMake in temporary directories:

| Input | Result |
| --- | --- |
| gfx1030, no custom hook | Target validation passed |
| gfx1013, no custom hook | Rejected as unknown target |
| gfx1013, custom target hook | Target validation passed |

The hook declares an explicitly unqualified experimental target. It does not
enable or validate any ROCm libraries. In particular, the upstream gfx1030 entry
excludes hipBLASLt, hipSPARSELt, composable_kernel, rocWMMA, hipTensor and
rocprofiler-compute. Do not copy an unrelated target's capabilities to PS5.

Hashes, Git blob identities and the compiler submodule pin are in
`evidence/offline-therock-comparison.json`. To repeat the isolated configuration
probe, place that exact target file under
`.deps/therock-reference-4d1b9e560fa23ea2340fcf17b423bb281ab29aae/cmake/`, then run
`python3 tools/probe_therock_targets.py`. The script checks the source digest
before evaluation. It does not configure or compile the entire TheRock tree.

## Completed compiler comparison and limits

The pinned TheRock build-tree tools compiled the four frozen fixtures with
explicit V5 and wave32 settings. The [binary comparison](THEROCK_BINARY_COMPARISON.md)
records target spelling, descriptors, kernargs, resource use and instruction
differences, including shared compiler-library identities. A second comparison
brackets execution with stable executable, resolved-library, source and inspector
hashes; see `evidence/offline-therock-verified-comparison.json`. Output-path-dependent
compilation IDs change symbol/string/hash tables, while the four objects' code,
descriptor and metadata sections match the first comparison exactly. Whole-object
equality is explicitly false; the additional fixed-ID diagnostic did not establish it.

Keep LLVM 18 selected. Comparing the relevant built compiler tools satisfies the
compiler-selection comparison. The configured `amd-llvm` target and staged/component
dist installation subsequently completed, including Flang; see
`evidence/offline-therock-component-completion.json`. Full ROCm artifact packaging
and upstream tests remain separate. Any changed device code needs native qualification.

The selected route has
fresh isolated-userspace reproduction of all four objects, with networking and
the development compiler filesystem excluded; it shares the WSL kernel and CPU.
See `evidence/offline-isolated-compiler.json`. Source-notice identities are recorded
in `evidence/offline-compiler-provenance.json`. A future binary-distribution audit
is separate from this offline compiler selection. G1's requirement-by-requirement
decision is recorded in `evidence/offline-g1-acceptance.json`.

## Executable comparison

`tools/compare_device_compiler.py --compiler-bin /path/to/compiler/bin --output
/path/to/new-results` runs the exact current baseline argument profiles with the
selected compiler and its verified device linker. It checks baseline source,
recipe and object hashes first, preserves driver/compile/metadata/disassembly
logs, and runs the current host loader against each result. Output must be a new
directory. Compiler failures are reported per case; they do not become capability
claims. A nonzero exit indicates at least one failed compile or inspection.

The baseline self-check reproduces all four LLVM 18 objects exactly and accepts
their metadata, ISA disassembly and loader structure. This validates the runner,
not TheRock: see `evidence/offline-compiler-comparison-runner.json`. Different
compiler outputs need instruction/resource comparison even if structural checks
pass. No target is GPU-qualified by these checks.
