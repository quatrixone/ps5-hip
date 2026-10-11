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

The pinned TheRock compiler component is configured and its build has started.
WSL task directory: `/tmp/ps5hip-therock-comparison`; build log: `build.log` there.
The active exec session was `40611` at this checkpoint. Poll that handle and
inspect the same build before restarting anything. Eight compile jobs, one LLVM
link job and two Flang jobs are configured. No compiler binary comparison has
passed. Exact source heads/options: `evidence/offline-therock-build-setup.json`.
The ignored `results/build-therock-compiler.py` contains the configure/build
invocation. All changes and build outputs remain local; no console run or push.
The LLVM subproject itself configured successfully and has begun its 8,739-step
build. Its detailed logs are `build/logs/amd-llvm_configure.log` and
`build/logs/amd-llvm_build.log` under that task directory. This is a running build,
not an unavailable compiler or a completed comparison. The dependency notice
inventory is in `docs/DEPENDENCY_PROVENANCE.md`. The logging header exactly matches
ps5-agc-gears `1ae1f918`, whose NOTICE explicitly licenses it GPL-3.0-or-later;
`evidence/offline-logging-provenance.json` records source and notice identities.
Linked-member accounting and binary-distribution requirements remain unresolved.
`docs/INITIAL_CAPABILITY_MATRIX.md` now records the initial entry-state contract,
exact resource fields, inspected first-kernel instructions and hardware unknowns.
The object inspector prints raw descriptor resource fields for compiler comparison.

Isolated-userspace reproduction exposed and fixed a linker-selection bug:
Clang 18's HIP driver ignored `-fuse-ld` and invoked host `lld` 21.1.8.
The recipe now supplies the pinned program-search directory and verifies the
actual device-link command before compiling. All four corrected objects match
byte-for-byte in a freshly extracted Ubuntu userspace with networking disabled;
22 host tests pass. See `evidence/offline-isolated-compiler.json` and
`docs/OFFLINE_BRINGUP.md`. This shares the WSL kernel, not its compiler filesystem.
The earlier `63b6e0402149` indexed-store image is superseded and explicitly
revoked by local verification. The corrected candidate below replaces it.
TheRock binary comparison and dependency license review remain outstanding.

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
C exit, worker-thread/framework cleanup nor any GPU capability. The builder now
allows an explicit shell probe for finite GPU candidates; none has run.
Command/marker teardown is checked before reporting submission success, with
injected unmap/release failure tests. Do not rerun the failed diagnostics.
The failed .40 process must have recovery established before another run there.

The finite-GPU indexed-store package is frozen at source `15bfc35`, stage
`gfx1013-indexed_store-3d295773d2fb`, image SHA
`fb634e34a1a6420fbd257e3929a3fbd37b5be8f6121db1eb6ee43362015febbb`.
It preserves standard/unknown exception reasons before the existing quarantine
park; focused host oracle/log tests pass. Ownership and retry behavior are unchanged.
The earlier `1288205d1a2a` image remains preserved as superseded evidence.
Startup ordering and emitted AGC/ShellCoreUtil imports were independently inspected;
the corrected device object was disassembled and inspected. UFS2 filesystem,
full extracted-file hashes and local candidate/controller verification pass.
See `evidence/offline-diagnostic-gpu-package.json`. No GPU run or push followed.
Native GPU builds now reject stale source/recipe/object/driver hashes and embed
the verified device report. Complete the remaining G1 compiler comparison
before using the console. The compiler source-notice inventory is now
recorded in `evidence/offline-compiler-provenance.json`; binary redistribution
accounting remains a later release obligation. Isolated userspace reproduction passes;
a separate physical host has not been tested. This alone does not close G1.
The first hardware compute goal remains exact execution of a small HIP-compiled
kernel through the native submission path. G3 additionally requires shared memory,
atomics, wave operations and deliberate spilling; vector addition is insufficient.
Separate wave32 broadcast/shuffle/ballot/reduction probes compile at O1 and O2 for
both targets, including lanes 4-19 active with lane zero inactive. Scalar expected
outputs and ISA inventories are retained in `evidence/offline-wave-probes.json`.
They have no native harness or GPU result and do not change the frozen candidate.
The tested generic volatile private-address path remains a G3 compiler issue.
An isolated contrast without `volatile` retains 144 private bytes but uses buffer
scratch on both targets, with 33 private-address IR accesses and no flat-scratch
setup. See `evidence/offline-private-inference.json`. This does not qualify
arbitrary pointers or native scratch execution and is not a volatile-semantics fix.

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
