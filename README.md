# ps5-hip

An experimental native HIP compute platform for PlayStation 5 homebrew.

The ambition is to make the PS5 a programmable GPU compute machine: compile real
HIP C++ kernels, reuse useful ROCm libraries, accelerate applications such as
ProsperoAI, and ultimately run native PyTorch inference and training.

**Status: offline compiler, loader and native canary development. Native startup
and GPU canary candidates link and fourteen host tests pass; no console validation or
general HIP runtime yet.**
The repository is private.

## Start here

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): architecture, twelve goals,
  acceptance gates, performance targets, risks and the first implementation tasks.
- [Research baseline](docs/RESEARCH_BASELINE.md): source revisions, established
  findings, unresolved assumptions and external references.
- [Current state](STATE.md): the next action and completed milestones.
- [Offline bring-up](docs/OFFLINE_BRINGUP.md): compiler recipe, loader tests and limitations.
- [Native build](docs/NATIVE_BUILD.md): pinned SDK, native candidates and remaining runtime work.
- [Agent instructions](AGENTS.md): how to continue without losing the evidence trail.

## What success looks like

1. A native PS5 application executes independently compiled HIP kernels with
   verified arithmetic, memory, synchronization and register-spill behavior.
2. Developers build useful HIP applications against a documented PS5 SDK, with
   supported operations working and unsupported operations failing explicitly.
3. Selected ROCm libraries and ProsperoAI use the runtime successfully, with
   measured correctness, performance and memory use against existing implementations.
4. Native PyTorch performs inference, backward propagation, optimizer steps and
   checkpoint/resume on the console. Small-model training and bounded LoRA
   experiments provide real applications of that capability.

The later goals explore on-console compilation, graph compilers and collaboration
between multiple consoles. These are planned research directions, not capabilities.

## Relationship to ProsperoAI

[ProsperoAI](https://github.com/blackbearreloaded/ProsperoAI) is the first intended
application and integration benchmark. It owns models, inference routing and the
user experience. This repository owns the reusable compute runtime, compiler
configuration, library integration and validation.

ProsperoAI's current Vulkan runtime is the integration baseline. HIP adoption is
gated on a demonstrated capability or performance benefit and stable operation.

## Technical direction

Start with device code compiled on a development PC, a bounded AMDGPU code-object
loader, and the existing native PS5 GPU submission and memory mechanisms. Prove
that boundary before choosing how much upstream HIP/CLR/ROCr machinery to reuse.
TheRock supplies reproducible component builds where useful; it is not a PS5
driver or a substitute for validating the console's execution model.

The GPU requires a measured capability profile. Existing PS5 Mesa findings show
that accepting GFX10.3 shader code does not establish full desktop `gfx1030`
compatibility. Compiler targets, dot products, wave state and scratch behavior
must be qualified explicitly.

## Scope and provenance

This is independent homebrew research for owner-controlled, compatible consoles.
No model weights, credentials, proprietary SDK components, firmware or system
module dumps belong in this repository. Dependencies retain their licenses and
will be pinned and attributed before incorporation. No project-wide code license
has been selected yet.

Repository creation and this plan do not authorize an automatic console launch.
Hardware execution follows the owner's current environment authorization and the
native development protocol described in the plan.
