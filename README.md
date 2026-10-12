# ps5-hip

An experimental native HIP compute platform for PlayStation 5 homebrew.

The ambition is to make the PS5 a programmable GPU compute machine: compile real
HIP C++ kernels, reuse useful ROCm libraries, accelerate applications such as
ProsperoAI, and ultimately run native PyTorch inference and training.

**Status: real HIP-compiled indexed-store and vector-add kernels executed natively
on PS5 with exact results, unchanged guards and clean title teardown on FW 6.02.
G1 passed offline; G2 remains in progress. No general HIP runtime is qualified.**
The first GPU candidate faulted on a CPU import boundary. Changing only the
`sceAgcSuspendPoint` provider produced the successful run. See
[the execution receipt](evidence/native-provider-fixed-store.json) and
[the fault analysis](docs/FIRST_GPU_FAULT.md). A later title cycle passed two
vector dispatches through one queue: [vector evidence](evidence/native-vector-twice-cycle1.json).
Host tests and ten exact-candidate sanitizer checks pass. A strict audit
identified the remaining G2 requirement: one frozen application must run both
indexed-store and vector-add, then pass repeated clean title cycles. The existing
separate profile images do not establish that combined-candidate proof.
The [G1 acceptance receipt](evidence/offline-g1-acceptance.json) records each
requirement, selected LLVM 18 route, TheRock comparison and remaining limits.
The repository is private.

## Start here

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): architecture, twelve goals,
  acceptance gates, performance targets, risks and the first implementation tasks.
- [Research baseline](docs/RESEARCH_BASELINE.md): source revisions, established
  findings, unresolved assumptions and external references.
- [Current state](STATE.md): the next action and completed milestones.
- [Runtime architecture](docs/RUNTIME_ARCHITECTURE.md): measured CLR comparison,
  selected native HIP approach and the threshold for reconsidering upstream CLR.
- [Compiler launch capture](docs/HIP_HOST_LAUNCH_CAPTURE.md): owned arguments,
  hidden geometry and packet preparation through a genuine stub; no GPU submission.
- [Read-only teardown diagnostic](docs/TEARDOWN_SNAPSHOT.md): bounded observations
  with explicit visibility limits; the first .40 mount query remains inconclusive.
- [Offline bring-up](docs/OFFLINE_BRINGUP.md): compiler recipe, loader tests and limitations.
- [Native build](docs/NATIVE_BUILD.md): pinned SDK, native candidates and remaining runtime work.
- [HIP fat-binary boundary](docs/HIP_FAT_BINARY_BOUNDARY.md): bounded bundle reader,
  inspector support and the next normal-launch registration step;
  [offline validation](evidence/offline-fat-binary-reader.json) covers 12 real objects.
- [HIP registration control](docs/HIP_HOST_REGISTRATION_CONTROL.md): genuine
  compiler constructors, owned module/token registration and explicit unsupported
  launch errors; Linux execution and PS5 link/conversion checks only.
- [Atomic native package](evidence/offline-atomic-native-readiness.json): six
  contended integer kernels and exact host oracles, packaged but unrun.
- [Experimental private-call compiler](tools/experimental/private-calls/README.md):
  reproduced bounded compiler recipe and its limits; the separate 304-byte native
  candidate is recorded in [offline evidence](evidence/offline-private-buffer-native-readiness.json).
  Its .40 deployment was [blocked before upload](evidence/native-private-buffer-console-no-run.json)
  because another title was running. A later arithmetic control
  [passed its compute checks on .40](evidence/native-arithmetic-fw1270-cycle1.json),
  but left explicit loader-layer release unproven; the private-call candidate
  remains unrun pending that prerequisite.
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
