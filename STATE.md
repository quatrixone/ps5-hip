# Current state

Updated: 2026-10-11, America/New_York.

## Actual status

- Repository initialization and implementation planning are complete.
- G1 is in progress; G2-G12 are planned. No goal has passed yet.
- LLVM 18.1.8 builds real AMDHSA V5 device-only HIP canaries for gfx1013 and
  gfx1030. A host code-object inspector/materializer and sanitizer tests pass.
- Initial launch preparation and compute packet fragments pass six host test
  cases across the two targets; no fragment is submitted or GPU-qualified.
- Pinned SDK preparation and LLVM 18 PS5 cross-build succeed. The native memory
  library and CPU-only smoke executable link. Submission lifecycle and AGC adapter
  cross-compile; fifteen C++ host sanitizer tests pass, including timeout quarantine,
  driver failure injection, AGC ABI models, guarded canary oracles and log capture.
- Initial cache/state commands and compute-drain/completion packets are encoded.
  Indexed-store and vector-add native candidates for both targets link with real
  AGC module imports and convert to integrity-checked FSELF containers.
  Packaging/runtime/GPU behavior is unverified.
- A CPU-only startup control checks constructors, heap, exception cleanup, TLS and
  the clock. All candidates use exclusive persistent logs. Native conversion uses
  hash-verified static zlib built offline from pinned source.
- No HIP runtime or hardware-qualified SDK exists yet.
- No console has been contacted or modified for this project.
- The user selected `PPSA99012` as the dedicated test title. Hash-verified staging
  and UFS2 packaging tools pass: five packages have clean filesystem checks and
  exact extracted-file hashes. Seventeen host test cases pass (fifteen C++, two Python).
- Pinned title-aware launch/close controllers cross-compile. Host mocks and a
  bounded application-log checker are implemented; neither sends console traffic.
- The console host/firmware and authoritative environment/lock configuration have
  been requested from the user. No project or workspace-root ENVIRONMENT.md exists.
- No hardware throughput, compatibility or stability claim belongs to this repository yet.

## Next action

Continue G1/G2 offline: prepare title-specific
deployment/log retrieval/close control and freeze the first bounded experiment.
See [console workflow](docs/CONSOLE_WORKFLOW.md) for the remaining host integration.
Generic private-address
lowering remains a G3 compiler issue. See
[offline bring-up](docs/OFFLINE_BRINGUP.md) for reproducible host checks and limits.
The first offline build/package milestone is on `development/offline-bringup`.
Further pushes remain reserved for substantial coherent milestones, as requested.

The first hardware goal is exact execution of a small HIP-compiled kernel through
the native PS5 submission path. G3 then requires shared memory, atomics, wave
operations and deliberate spilling; vector addition alone does not qualify a runtime.

## Entry points

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Research baseline](docs/RESEARCH_BASELINE.md)
- [Native launch contract](docs/NATIVE_LAUNCH_CONTRACT.md)
- [Native build](docs/NATIVE_BUILD.md)

Read only the active goal and relevant source slices. Raw logs stay in ignored
`results/`; compact, sanitized validation receipts live under `evidence/`.
`evidence/offline-resource-probes.json` records the current source hash, compiler
versions, object hashes and instruction inventories without local user paths.
`evidence/offline-native-canaries.json` records linked native candidates and their
source/dependency hashes; it is not hardware qualification.
`evidence/offline-native-packages.json` records all five `PPSA99012` package
manifests, hashes, source commit and the pinned image tool's binary provenance.
`evidence/offline-controllers.json` records controller hashes and host validation.

## Milestones

- 2026-10-11 | planning | twelve measurable goals and pinned research references recorded | next: G1 offline inventory
- 2026-10-11 | G1 progress | arithmetic and resource probes compiled for two targets; host parser checks pass; private-array flat-scratch incompatibility identified | not a completed gate
- 2026-10-11 | G1 progress | unmodified compiler emits buffer scratch for explicit-private accesses and 52 real VGPR spills; generic private pointers remain unresolved | not hardware-qualified
- 2026-10-11 | offline milestone | 0c2e8c1 | PPSA99012 | packaged, no-run: five verified images and fourteen host tests | evidence/offline-native-packages.json | next: console controller
