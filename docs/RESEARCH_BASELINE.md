# Research baseline

Recorded 2026-10-11. These are research inputs, not ps5-hip validation results.
References to `main`, `develop` or `latest` can move; use the recorded commits for
the initial investigation and record any replacement before comparing results.

## 1. Source identities

| Project | Inspected revision | Purpose |
| --- | --- | --- |
| [ProsperoAI](https://github.com/blackbearreloaded/ProsperoAI/tree/45fe9bbbc8a34d7ff49f1e281a7e144321474ee1) | `45fe9bbbc8a34d7ff49f1e281a7e144321474ee1` | Current integration and workload baseline |
| [TheRock](https://github.com/ROCm/TheRock/tree/4d1b9e560fa23ea2340fcf17b423bb281ab29aae) | `4d1b9e560fa23ea2340fcf17b423bb281ab29aae` | ROCm component builds and target exclusions |
| [rocm-systems](https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c) | `65575fbbe6f73b9d9d96bfbcb523c856709e664c` | HIP/CLR, ROCr and kernel-interface architecture |
| [PS5_Mesa](https://github.com/mihawk-99/PS5_Mesa/tree/47e9eb61ee7e1112194557e1e047178562827bc6) | `47e9eb61ee7e1112194557e1e047178562827bc6` | Native platform backend and compiler feature adjustments |
| [BC-250 ROCm research](https://github.com/akandr/bc250-rocm/tree/bd897eccac05cb02f1c0c1e3cb9d4e6a490f58b2) | `bd897eccac05cb02f1c0c1e3cb9d4e6a490f58b2` | Related-hardware precedent and hypotheses to test |

These are reference pins, not a qualified dependency lock. G1 must select mutually
compatible build versions and identify the compiler and PS5 SDK revisions. In
particular, the newest TheRock and runtime revisions are not automatically the
best bring-up combination. The external BC-250 investigation uses its own older
ROCm configuration; its patches cannot be assumed applicable to current sources.

## 2. What the existing PS5 projects establish

The public [compute findings](https://github.com/blackbearreloaded/ps5-gpu-research/blob/main/docs/compute.md)
record native GPU execution, stable GPU-visible model storage, ordered compute
graphs and reference-checked transformer inference. They also record explicit
CPU/GPU visibility boundaries and failures caused by wave-mode mismatch and
remapping descriptor-backed storage. These are useful design constraints, not
HSA or HIP conformance evidence.

The newer [ProsperoAI active state](https://github.com/blackbearreloaded/ProsperoAI/blob/45fe9bbbc8a34d7ff49f1e281a7e144321474ee1/docs/VK_ACTIVE.md)
supersedes the old dedicated-AGC inference architecture. It records a common
Prism llama/ggml Vulkan runtime, removal of the dedicated AGC inference path,
successful Bonsai 2 27B tests at approximately 11.2 tokens/s, model/context
switching, and functional media checks. The platform driver still underlies RADV.

Read the [runtime description](https://github.com/blackbearreloaded/ProsperoAI/blob/45fe9bbbc8a34d7ff49f1e281a7e144321474ee1/docs/VULKAN_RUNTIME.md)
and [media comparison](https://github.com/blackbearreloaded/ProsperoAI/blob/45fe9bbbc8a34d7ff49f1e281a7e144321474ee1/docs/MEDIA_VULKAN_COMPARISON.md)
before selecting benchmark controls. Historical AGC and Vulkan media comparison
builds differ from the final unified runtime. Retained contexts and initial
requests measure different things. Their reported improvements do not isolate
the driver and cannot be credited to a future HIP backend.

The older compute evidence often concerns firmware 6.02. The October 11 media
comparison names firmware 12.70. No cross-firmware support follows from combining
these records. ps5-hip must identify the actual test console and firmware anew.

## 3. Hardware profile: evidence before labels

The [PS5 Mesa platform description](https://github.com/mihawk-99/PS5_Mesa/blob/47e9eb61ee7e1112194557e1e047178562827bc6/src/amd/vulkan/winsys/ps5/radv_ps5_winsys.c)
records working GFX10.3 shader compilation alongside selected GC 10.1.3 traits.
The most relevant compiler constraints are the disabled accelerated dot-product
path, different wave-ID behavior and replacement of a faulting flat-scratch path
with buffer-based access. Treat these as constraints observed through this driver
and execution environment, not proof that every instruction in a family is absent
from the silicon or that an alternative setup can never work.

[LLVM's AMDGPU guide](https://llvm.org/docs/AMDGPUUsage.html) lists `gfx1013`
and documents AMDHSA code objects and execution state. Availability of a compiler
target does not establish that its default features or ABI setup match PS5.

An initial comparison should inspect `gfx1013` and a restricted GFX10.3 candidate,
with exact compiler versions and feature settings recorded. No production target
is selected yet. In particular, do not use a device-identity override as a substitute
for implementing the runtime or checking emitted instructions.

## 4. ROCm component boundaries

[ROCr](https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c/projects/rocr-runtime)
provides HSA services including queues, signals and memory interfaces, with the
usual Linux implementation relying on AMDGPU/ROCK and its thunk layer.
[HIP's runtime architecture](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api.html)
describes a device abstraction beneath HIP. Native PS5 adaptation can therefore be
investigated at more than one layer; no existing PS5 backend was identified.

[TheRock's target configuration](https://github.com/ROCm/TheRock/blob/4d1b9e560fa23ea2340fcf17b423bb281ab29aae/cmake/therock_amdgpu_targets.cmake)
contains per-target component exclusions. Building core HIP and obtaining all
ROCm libraries are separate qualifications. The initial target list does not
provide a complete PS5 platform definition.

AMD identifies `gfx101X` and `gfx103X` as incompatible with rocWMMA's required
matrix capability in [TheRock issue 1944](https://github.com/ROCm/TheRock/issues/1944).
Software fallbacks or different algorithms can be explored, but no plan should
count on adding absent matrix hardware through a runtime port.

## 5. Related-hardware precedent

The [BC-250 investigation](https://github.com/akandr/bc250-rocm/tree/bd897eccac05cb02f1c0c1e3cb9d4e6a490f58b2)
reports working HIP, rocBLAS, llama.cpp and source-built PyTorch after modifications
to a Linux system. It documents failures and recovery limitations as well as
successful workloads. Its benchmark results favor different backends for different
cases; HIP is not a universal performance improvement over Vulkan.

This is independent, single-board evidence that makes the project more plausible.
It does not validate native PS5 queue access, firmware behavior, memory services,
compiler settings or any ps5-hip implementation. Use its minimal reproductions and
kernel techniques as research leads, with provenance retained. Do not transplant
Linux driver parameters, firmware substitutions, CU-unlock assumptions or clock
settings to the console.

## 6. Compatibility distinctions

| Claim | Required proof |
| --- | --- |
| HIP source compatibility | Named source builds against supported headers and runtime and runs correctly |
| Device code-object compatibility | Qualified compiler outputs load and execute with required metadata and features |
| Host binary compatibility | Host executable ABI, imports and operating-system services match; not an initial goal |
| Library compatibility | Real library entrypoints, selected kernels and host dependencies pass |
| Framework compatibility | Real model/operator/optimizer paths pass through the framework |
| Performance improvement | Matched workloads, numerical checks and repeated comparative measurements |
| Native execution | GPU computation occurs on the PS5; host tools may compile, control and collect results |

The initial SDK aims at HIP source compatibility and qualified device code objects.
It does not promise that Linux Python wheels, Linux host executables or arbitrary
precompiled AMD GPU binaries will run unchanged.

## 7. Questions that remain open

- Which LLVM target/features produce reliable native PS5 AMDHSA kernels?
- Can the current submission interface establish every required kernel entry state?
- Can flat/global pointer operations, local/private apertures and spilling be made
  correct without extensive compiler changes?
- Which subset of stream, event and system-memory semantics is implementable?
- How much upstream CLR/ROCr can be reused more economically than a narrow runtime?
- Which rocBLAS/MIOpen kernels are useful and valid for this capability profile?
- How much application memory remains after realistic runtime and workspace costs?
- What remains CPU-bound in current ProsperoAI, and what could HIP actually improve?
- Can native LibTorch and a compatible Python runtime meet the host ABI requirements?
- What can recover cleanly from failures, and what requires ending the title?

No one of these questions is answered by successfully compiling TheRock.
