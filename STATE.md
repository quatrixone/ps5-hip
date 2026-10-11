# Current state

Updated: 2026-10-11, America/New_York.

## Actual status

- Repository initialization and implementation planning are complete.
- All implementation goals G1-G12 are planned, not passed.
- No compiler, runtime, SDK, code-object loader or HIP library has been built here.
- No console has been contacted or modified for this project.
- No hardware throughput, compatibility or stability claim belongs to this repository yet.

## Next action

Begin G1 with an offline inventory: inspect the pinned PS5 Mesa submission/memory
interfaces and the host LLVM/HIP toolchain, identify reusable public components
and their licenses, and select the first code-object format/compiler candidates.
Produce a device-only canary and inspect its instructions and metadata before
designing or executing a native launcher.

The first hardware goal is exact execution of a small HIP-compiled kernel through
the native PS5 submission path. G3 then requires shared memory, atomics, wave
operations and deliberate spilling; vector addition alone does not qualify a runtime.

## Entry points

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Research baseline](docs/RESEARCH_BASELINE.md)

Read only the active goal and relevant source slices. Raw logs stay in ignored
`results/`; compact, sanitized validation receipts will be committed under
`evidence/` when experiments exist. That directory is not populated in this seed.

## Milestones

- 2026-10-11 | planning | twelve measurable goals and pinned research references recorded | next: G1 offline inventory
