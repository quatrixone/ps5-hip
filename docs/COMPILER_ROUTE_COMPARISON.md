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
| Device compiler | Installed Clang/LLD 18.1.8, exact version checked | Pinned AMD LLVM submodule, not built here |
| Target configuration | Both gfx1013/gfx1030 fixtures compile | gfx1030 declared; gfx1013 needs a custom declaration |
| Component scope | Device-only HIP, without headers/device libraries | Compiler project plus bundled runtime/device-library components |
| Artifact evidence | V5 objects, ISA inventories, metadata and host parser tests | Source/configuration evidence only so far |
| Native PS5 host code | Separate pinned PS5 SDK and native converter | Still requires a distinct PS5 port/integration |
| GPU qualification | Pending | Pending |

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

## Remaining comparison work

Build the pinned TheRock compiler component in isolation, record its actual
compiler/version/features, and compile the same frozen fixture sources with
explicit V5 and wave32 settings. Compare descriptors, kernarg layouts, resource
requirements and instruction inventories; investigate differences before changing
the selected compiler. Run native qualification separately for any changed device
code. A successful custom-target configuration is not a substitute for these steps.

G1 stays open pending the built-compiler comparison. The selected route now has
fresh isolated-userspace reproduction of all four objects, with networking and
the development compiler filesystem excluded; it shares the WSL kernel and CPU.
See `evidence/offline-isolated-compiler.json`. Source-notice identities are recorded
in `evidence/offline-compiler-provenance.json`. A future binary-distribution audit
is separate from this offline compiler selection.

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
