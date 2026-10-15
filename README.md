# ps5-hip

An experimental native HIP compute platform for PlayStation 5 homebrew.

**First release: native LLM inference.** The active [v1 plan](docs/LLM_V1_PLAN.md)
targets one small Llama-family model, GGUF Q4_0, batch one and a 512-token context
through a standalone PS5 HIP client. Model execution is not yet qualified.
Training, broad frameworks and general SDK expansion remain later goals.

The ambition is to make the PS5 a programmable GPU compute machine: compile real
HIP C++ kernels, reuse useful ROCm libraries, support standalone inference clients,
and ultimately run native PyTorch inference and training.

**Status: a bounded native HIP runtime now executes genuine compiler-registered
clients on PS5 firmware 12.70. G3/G4 remain incomplete; frameworks and application
integration are still planned.** The [reusable client](evidence/native-hip-generic.json)
passed eleven blocking launches through two kernels, with exact results and
complete compiler/runtime cleanup. The [corrected math fixture](evidence/native-math-exp-corrected.json)
passed 1,542 values under an unchanged numerical contract. The
[tiled matrix client](evidence/native-hip-matmul.json) passed eight launches and
982 exact outputs across four shapes using combined 2D geometry and static LDS.
An [unedited HIPIFY translation](evidence/native-hipify-client.json) also passed
six native launches and 662 exact unsigned sums with complete cleanup. This
qualifies that ordinary CUDA source example, not general CUDA compatibility.
The [CPU threading proof](evidence/native-host-threads.json) passed two joined
workers, isolated thread-local values and synchronization cleanup on the console.
Concurrent HIP calls and GPU access from worker threads remain unqualified.
The runtime currently
requires one module and serialized host calls; unsupported features fail explicitly.
See [current state](STATE.md) for the latest candidate and its exact limits.

The earlier G2 milestone checked 292 exact values over two clean cycles of one
image, with unchanged guards and title teardown.
The [G2 acceptance record](evidence/native-g2-acceptance.json) maps all six
requirements to the exact gfx1013, wave32, 1D arithmetic fixture. Mount checks
establish unchanged visible title records; persistent registration is expected.
The first GPU candidate faulted on a CPU import boundary. Changing only the
`sceAgcSuspendPoint` provider produced the successful run. See
[the execution receipt](evidence/native-provider-fixed-store.json) and
[the fault analysis](docs/FIRST_GPU_FAULT.md). A later title cycle passed two
vector dispatches through one queue: [vector evidence](evidence/native-vector-twice-cycle1.json).
Host tests and exact-candidate sanitizer checks support the qualified subset;
general compute semantics and framework compatibility remain unqualified.
The [G1 acceptance receipt](evidence/offline-g1-acceptance.json) records each
requirement, selected LLVM 18 route, TheRock comparison and remaining limits.
The repository is private.

## Start here

For a fresh WSL/Linux checkout, install Python 3, GNU Make, CMake, Ninja,
LLVM/LLD 18.1.8 and .NET SDK 10.0.112, then run `make` to fetch the locked
public dependencies, run host tests, cross-compile the native core, and build
the 74-object public HIP bootstrap package offline. Set `RESULTS_DIR` to the
intended console results path. ProsperoEden and ps5-homebrew-dev-protocol are
snapshots of their public `main` branches; no unpublished dependency branch is
needed. See the [source-only build](docs/SOURCE_ONLY_REBUILD.md) for outputs
and the distinction between a new package and the image already installed.

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): architecture, twelve goals,
  acceptance gates, performance targets, risks and the first implementation tasks.
- [Research baseline](docs/RESEARCH_BASELINE.md): source revisions, established
  findings, unresolved assumptions and external references.
- [Current state](STATE.md): the next action and completed milestones.
- [Agent handoff](HANDOFF.md): exact branches, reviewed evidence, console state,
  and the safe resume point for the public HIP bootstrap.
- [Source-only rebuild](docs/SOURCE_ONLY_REBUILD.md): fresh-clone build command,
  outputs, and package limits.
- [Runtime architecture](docs/RUNTIME_ARCHITECTURE.md): measured CLR comparison,
  selected native HIP approach and the threshold for reconsidering upstream CLR.
- [Compiler launch capture](docs/HIP_HOST_LAUNCH_CAPTURE.md): owned arguments,
  hidden geometry and packet preparation through a genuine stub; no GPU submission.
- [Pinned OCML math probe](docs/OCML_MATH_PROBE.md): reproducible partial library
  build and four device objects. The [cutoff correction](docs/MATH_EXP_CORRECTION.md)
  passed its finite native corpus; all-domain accuracy remains unqualified.
- [Native math profile](docs/NATIVE_MATH_PROFILE.md): independent MPFR oracle,
  full guards and 280-byte argument checks; see the
  [corrected native result](evidence/native-math-exp-corrected.json).
- [Owned native data](docs/NATIVE_DATA_REGISTRY.md): checked logical allocations,
  copies and free. These ownership components now support the qualified
  [genuine HIP clients](evidence/native-hip-generic.json).
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
- [Atomic native result](evidence/native-atomic-suite.json): six contended integer
  dispatches passed 98,694 semantic values and complete teardown under the fixed
  relaxed-order contract.
- [Experimental private-call compiler](tools/experimental/private-calls/README.md):
  reproduced bounded compiler recipe and its limits. The separate 304-byte
  private-call fixture has [exact computation evidence with delayed completion](evidence/native-private304-computation.json).
  Its original bounded cycle remains unaccepted; this does not admit arbitrary
  private-call kernels into the reusable runtime.
- [Agent instructions](AGENTS.md): how to continue without losing the evidence trail.

## What success looks like

1. A native PS5 application executes independently compiled HIP kernels with
   verified arithmetic, memory, synchronization and register-spill behavior.
2. Developers build useful HIP applications against a documented PS5 SDK, with
   supported operations working and unsupported operations failing explicitly.
3. Selected ROCm libraries and standalone inference clients use the runtime
   successfully, with
   measured correctness, performance and memory use against existing implementations.
4. Native PyTorch performs inference, backward propagation, optimizer steps and
   checkpoint/resume on the console. Small-model training and bounded LoRA
   experiments provide real applications of that capability.

The later goals explore on-console compilation, graph compilers and collaboration
between multiple consoles. These are planned research directions, not capabilities.

## Standalone SDK scope

This repository owns the reusable compute runtime, compiler configuration,
library integration and validation. G7 delivers a standalone ggml/llama inference
client and a reproducible SDK preview. ProsperoAI integration is out of scope.
Historical application research remains provenance, not an integration requirement.
Benchmark controls must use pinned, matched workloads and qualified backends.

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
