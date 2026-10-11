# Current state

Updated: 2026-10-11, America/New_York.

## Actual status

- Current development identity: **PPSA99014**, selected by the user after a
  pre-existing PPSA99012 installation was found on .30. Controllers and the
  elevation helper are rebuilt for the new exact title.
- First PPSA99014 run on .40 registered and created its native process, then failed
  with PRX_SCE_MODULE_LOAD_ERROR before an application checkpoint. The kernel
  reported a missing sce_module companion. All declared services stayed reachable;
  exact-title close was sent, but normal exit/complete teardown is unproven.
  No GPU work ran. See `evidence/first-native-cpu.json`.
- The builder now reproduces the pinned, independently authored libc.prx loader
  companion. Its raw and signed hashes match the native template's known hashes;
  FSELF integrity passes. Staging rejects its absence. This corrected candidate
  is packaged as `gfx1013-runtime-4a0334b256d6` at source `1b6f717`, with clean
  filesystem/extraction and artifact checks. See `evidence/offline-loader-companion.json`.
  On .30 (FW 6.02), this candidate passed constructors, heap, exception unwinding,
  TLS and clock checks with zero GPU API calls, then exited abnormally with SIGSYS
  in libkernel after logging exit_value=0. Title stop, sandbox removal and image
  unmount were observed; services remained reachable. The CPU startup/exit gate
  still fails. See `evidence/native-cpu-shutdown.json`.
- On .40, PID 204 from the earlier loader failure remained present after one
  identity-guarded SIGKILL. No repeated termination or relaunch was attempted;
  complete teardown remains unproven. Do not use .40 for another experiment
  without establishing recovery. See the first-run evidence's cleanup followup.
- A PPSA99012 CPU image was uploaded and byte-verified on idle .40 (FW 12.70).
  ShadowMount mounted UFS2 but rejected registration because icon0.png was absent.
  No title was launched. The icon fix and staging regression test pass; the old
  uploaded image remains preserved. Evidence: `results/cpu-deploy-20261011T170003Z/`.

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
- User-directed storage revision: Eden's elevation client plus an exact-title
  Lapy owned-root helper; configurable ordinary results directory, no download0
  dependency. CPU and indexed-store native builds pass. The revised CPU package
  is frozen at source 955c989; filesystem and full extraction/hash checks pass.
  See `evidence/offline-elevation.json`. Its configured directory is
  `/data/ps5-hip/results`; this is a staging choice, not an executable constant.
  Nineteen host cases pass, including the exported Eden/Lapy wire contract.
- Preflight contacted both authorized consoles; PPSA99012 and PPSA99014 images
  were deployed on .40. Only PPSA99014 was launched, once, and failed in loading.
- The original `PPSA99012` staging identity is historical; use `PPSA99014`. Hash-verified staging
  and UFS2 packaging tools pass: five packages have clean filesystem checks and
  exact extracted-file hashes. Nineteen host test cases pass (sixteen C++, three Python).
- Pinned title-aware launch/close controllers cross-compile. Host mocks and a
  bounded application-log checker are implemented; neither sends console traffic.
- The user authorized 192.168.4.30 or 192.168.4.40 only when idle. Preflight under
  the protocol's workspace `lock.txt` found FTP, klog and elfldr reachable on both.
  The later preflight established shell focus after PPSA99008 stopped on both;
  .30 became busy again before title inspection. Locks were released after every
  bounded phase. Reported firmware: .30 = 6.02, .40 = 12.70. Both the protocol
  default lock and Eden's PS5-Claude lock are now coordinated.
- No hardware throughput, compatibility or stability claim belongs to this repository yet.

## Next action

The opt-in CPU shell-managed lifecycle passed its single declared experiment on
.30, firmware 6.02, using PPSA99014. Source `3346166` and frozen image
`gfx1013-runtime-9ac5d150b5db` (SHA `d9a9e95cfc3e10b0fd8b547e54da3d2898291700537a27ecd9da3d63a6cc9cb0`)
passed runtime checks, application callbacks, destructor observation and application
fini. The shell request returned zero; the title stopped before host cleanup,
with no observed fatal signal or loader error. Sandbox/image release and service
health were confirmed. Previous image and log bytes were preserved. See
`evidence/native-shell-exit.json`. All 22 current host CTest cases pass.

Ordinary C exit remains unresolved: both previous CPU cases reached SIGSYS after
runtime checks. Shell-managed closure deliberately leaves remaining platform
finalizers to process reclamation. This one CPU run establishes neither general
C exit, worker-thread/framework cleanup nor any GPU capability. The builder still
rejects the shell probe for GPU candidates. Do not rerun the failed diagnostics.
The failed .40 process must have recovery established before another run there.

Next prepare an explicit finite-GPU lifecycle offline, including completion,
resource cleanup and failure handling before shell closure. Inspect its ISA,
commit and freeze the candidate, and define bounded acceptance before deployment.
The first hardware compute goal remains exact execution of a small HIP-compiled
kernel through the native submission path. G3 additionally requires shared memory,
atomics, wave operations and deliberate spilling; vector addition is insufficient.
Generic private-address lowering remains an unresolved G3 compiler issue.

G1 is still open. The pinned TheRock configuration accepts gfx1030, rejects gfx1013
by default, and accepts gfx1013 through its custom-target hook. This is target
configuration only. The selected LLVM 18.1.8 route reproduces four device objects
byte for byte on the same host; built-TheRock binary comparison, clean-machine
reproduction and dependency license review remain outstanding. See
`docs/COMPILER_ROUTE_COMPARISON.md`, `evidence/offline-therock-comparison.json`
and `evidence/offline-compiler-reproduction.json`.

Console access remains restricted to idle consoles, both shared locks and the
exact-title controls. Read `docs/CONSOLE_WORKFLOW.md` and the local environment
before a new console phase. Pushes remain reserved for coherent milestones.

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

- 2026-10-11 | CPU lifecycle milestone | 3346166 | PPSA99014 | one shell-managed startup/finalization/self-close passed on FW 6.02; 22 host cases pass | evidence/native-shell-exit.json | no G1-G12 gate passed
