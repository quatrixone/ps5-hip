# Current state

Updated: 2026-10-11, America/New_York.

## Actual status

- Repository initialization and implementation planning are complete.
- G1 is in progress; G2-G12 are planned. No goal has passed yet.
- LLVM 18.1.8 builds real AMDHSA V5 device-only HIP canaries for gfx1013 and
  gfx1030. A host code-object inspector/materializer and sanitizer tests pass.
- No native launcher, HIP runtime or hardware-qualified SDK exists yet.
- No console has been contacted or modified for this project.
- No hardware throughput, compatibility or stability claim belongs to this repository yet.

## Next action

Continue G1 offline: investigate generic private-address lowering, freeze the
platform dependency and license selection, and translate the inspected descriptor
contract into a tested native launch description. See
[offline bring-up](docs/OFFLINE_BRINGUP.md) for reproducible host checks and limits.
The first implementation changes remain local on `development/offline-bringup`;
push only after a substantial coherent milestone, as requested.

The first hardware goal is exact execution of a small HIP-compiled kernel through
the native PS5 submission path. G3 then requires shared memory, atomics, wave
operations and deliberate spilling; vector addition alone does not qualify a runtime.

## Entry points

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Research baseline](docs/RESEARCH_BASELINE.md)

Read only the active goal and relevant source slices. Raw logs stay in ignored
`results/`; compact, sanitized validation receipts live under `evidence/`.
`evidence/offline-resource-probes.json` records the current source hash, compiler
versions, object hashes and instruction inventories without local user paths.

## Milestones

- 2026-10-11 | planning | twelve measurable goals and pinned research references recorded | next: G1 offline inventory
- 2026-10-11 | G1 progress | arithmetic and resource probes compiled for two targets; host parser checks pass; private-array flat-scratch incompatibility identified | not a completed gate
- 2026-10-11 | G1 progress | unmodified compiler emits buffer scratch for explicit-private accesses and 52 real VGPR spills; generic private pointers remain unresolved | not hardware-qualified
