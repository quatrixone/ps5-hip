# Offline native build

The cross-build uses LLVM 18.1.8 and the SDK inputs in `dependencies.json`.
The SDK is prepared under this repository's ignored `.deps` directory; shared
SDK installations and other projects are left untouched.

## Prepare and build

Run under WSL/Linux. Supply a local clone containing the locked SDK revision and
the cached upstream v0.42 archive. The archive is checked against SHA-256
`8cfbc7cd5811e719eb4f0c47eea668d3dc7b40bc8ab11c4a5031d40c23ec02da` before use.

```sh
python3 tools/prepare_sdk.py --sdk-source /path/to/PS5_PayloadSDK --release-archive /path/to/ps5-payload-sdk.zip
cmake -S . -B build/ps5 -G Ninja -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/ps5-toolchain.cmake" -DBUILD_TESTING=OFF
cmake --build build/ps5
llvm-readobj-18 --file-headers --program-headers --needed-libs build/ps5/native-memory-smoke
```

The preparation script exports the locked Git commit, rather than building from
uncommitted source. It delegates installation to that revision's SDK setup script
with the verified archive already cached. The compiler wrapper pins an absolute
`LLVM_CONFIG` path: upstream's wrapper interprets a relative value against the
current build directory. The same selection is retained during Ninja builds,
not merely CMake configuration.

Produced and link-checked offline:

- `ps5hip-inspect`: AMDHSA code-object inspection compiled for the PS5 CPU target.
- `libps5hip_native_memory.a`: allocation, submission lifecycle and AGC adapter.
- `native-memory-smoke`: a separate, opt-in CPU direct-memory write/read test.

The smoke executable requires `--memory-only` to do anything. It contains no AGC
initialization or GPU submission. It was **not run on a console**. The current SDK
linker produces a payload-style ELF64 x86-64 shared object, not a packaged native
title. Static inspection showed 16 KiB PT_LOAD alignment and imports from
`libkernel_web.sprx`, `libSceLibcInternal.sprx` and `libSceNet.sprx`. The SDK's first
load segment is RWX. Dedicated-title linking/packaging and its startup contract
remain separate work before an AGC launch experiment.

The memory smoke does not pull in the AGC archive object. A separate native-title
link experiment below now links the adapter and records actual AGC imports.
Host boundary tests use explicit mock AGC functions; no system GPU API is called.

## Dedicated native canary candidate

```sh
python3 tools/prepare_host_zlib.py --archive /path/to/zlib-1.3.2.tar.gz
python3 tools/build_native_canary.py --native-source /path/to/PS5_Vulkan --target gfx1013 --kernel indexed_store
```

The target must be `gfx1013` or `gfx1030`; the kernel must be `indexed_store` or
`vector_add`, or `runtime` for the CPU-only startup control. Each executable runs
only its selected experiment. The builder
exports the exact native-tool revision in `dependencies.json`, preserving its
notices, and builds the host converter with Clang 18 and pinned static zlib 1.3.2.
`prepare_host_zlib.py` checks the cached source archive's SHA-256 before extracting
or building it, retains the source license, and records installed library/header
hashes. The candidate builder verifies those hashes before linking. Neither step
downloads anything or uses the reference checkout's local modifications.

The candidate uses native title startup, the pinned SDK's static libc++, ABI and
unwind libraries, and direct imports from libSceAgc, libSceAgcDriver, libkernel
and libSceLibcInternal. The SDK unwinder needs explicit start/end symbols for
`.eh_frame` and `.eh_frame_hdr`; the build derives them from linked sections.
Link-only AGC stubs are not packaged. No proprietary implementation is copied.

Output is under `build/native/<target>-<kernel>/`: an intermediate PIE, converted
native ELF, a validated plaintext FSELF container in `candidate/eboot.bin`, the
matching `canary.hsaco` for GPU cases, and a receipt with source/artifact hashes.
The runtime control needs no GPU code object. The converted
ELF uses type `0xfe10`, FreeBSD OSABI 9/ABI 2 and 16 KiB load alignment. Native
imports and container integrity are offline structural checks, not loader proof.
No title ID, title metadata or deployment is generated yet.

The GPU harness reads `/app0/canary.hsaco`, checks its target, performs the runtime
self-check before AGC initialization, and submits two 64-thread groups for 73
elements. All rounded data allocations are poisoned, with a 64-byte prefix guard
and the entire remaining tail checked. Integer output is exact; vector-add inputs
and expected results are exactly representable binary fractions, so their bit
patterns are compared exactly too. Input mutation and cleanup failures fail the
experiment. Two host oracle tests inject missing writes, wrong values at group
boundaries, input corruption and guard corruption. They do not execute GPU code.

An unresolved submission parks the title while retaining allocations and prints
`QUARANTINED`; it does not retry or return through process teardown. The external
controller must impose the run deadline and follow the shared-console protocol.
Native runtime behavior, development-title packaging, artifact freeze and the
first experiment record remain to be qualified before
console use. A successful link does not justify skipping those steps.

### Runtime control and persistent log

Build `--target gfx1013 --kernel runtime` for the first startup control. It checks
a dynamic global constructor, heap-backed strings, exception type/message and
cross-function destructor unwinding, main-thread emulated TLS, and monotonic clock
ordering. It never calls an AGC API or allocates GPU direct memory. This is a
single-thread startup check, not general TLS/threading qualification.

All variants first open `/download0/ps5-hip-<target>-<kernel>.log` exclusively and
redirect stdout/stderr there without buffering. Existing files cause refusal;
before a later authorized run the controller must preserve the previous log and
remove only that exact file. The harness never deletes logs. A host child-process
test verifies both streams are captured, old buffered output is excluded, and an
existing log is preserved. Checkpoints are flushed; runtime completion, final
result and quarantine attempt `fsync`. A missing log is failure/inconclusive,
never evidence of startup success. Native mount permissions and persistence still
require the CPU-only control on hardware.

Static inspection found no unresolved payload-loader hooks in the linked title.
The SDK's `libcxx.sh` builds libunwind in bare-metal mode; the explicit ELF frame
boundaries match that configuration. These findings remove a known payload-only
dependency concern, but do not prove that the console loader, system libc and
our static C++ runtime cooperate correctly. The startup control precedes every
GPU experiment for that reason.

## Memory behavior

`NativeMemory` allocates direct memory of type 12 with CPU/GPU read/write mapping
permissions. Sizes round to 16 KiB; physical/mapping alignment is at least 64 KiB,
and at least 2 MiB for allocations of 2 MiB or more. Code mappings must remain in
the observed 4 GiB shader window; all pointers must fit below `2^47`.

The initial allocator asks the kernel to choose the mapping address. It does not
yet reserve a separate device-memory arena, so large application allocations and
efficient reuse are not qualified. Callers must serialize access to each object.

GPU-use references prevent unmapping/releasing memory before completion. Timeout
must retain these references and poison the submitting queue. A failed unmap does
not release physical backing; a failed physical release can be retried without
unmapping twice. Destruction with outstanding GPU use or failed cleanup logs the
condition and retains memory rather than reclaiming potentially active storage.
This is a fail-closed lifetime rule, not GPU cancellation or a recovery mechanism.

The host mock injects allocation, mapping, unmap and release failures, checks
alignment/placement rejection, and confirms two GPU references both have to retire
before cleanup proceeds. It does not simulate GPU execution. After SDK export,
reconfigure the normal host build to enable this additional test; CMake reports
explicitly if the required platform declarations are absent.

## Submission lifecycle

`SubmissionQueue` owns a fresh command allocation and completion-marker allocation
for each submission. It copies the caller's words into owned storage, appends
CS_PARTIAL_FLUSH and the reference RELEASE_MEM sequence, deduplicates ownership, retains every
allocation and flushes CPU cache lines before entering the driver. Callers must
list every allocation reachable from the shader. The caller must also provide
the complete initialization/cache prefix. `encode_initial_dispatch` now supplies
the initial GFX10 cache invalidation and compute-state writes; this combination
is still unqualified on PS5.

Only an exact marker match releases the references, after CPU readback visibility
operations. A submit/kick error, post-submit exception, backward clock or timeout
leaves the queue poisoned and its allocations retained. A late marker cannot
silently restore the queue. Each backend allows only one queue lifetime; the native
AGC adapter allows only one initialization attempt per process.

The initial API accepts at most 16,384 caller command words and a completion wait
of 1-10,000 ms. A finite poll budget also terminates a clock that stops advancing.
This bounds polling, **not a driver call that never returns**. The dedicated-title
experiment still needs an external bounded run/health-check protocol. Neither
timeouts nor destructor cleanup implement GPU cancellation.

The host model covers immediate/delayed completion, duplicate buffer references,
submit and kick errors, driver exceptions, pre-submit flush failure, frozen and
backward clocks, late completion and refusal to replace a queue. AGC boundary
models verify version 8 initialization, the 16-byte submit ABI with a zero flag,
argument rejection, initialization failure and native error propagation. CPU cache
instructions are cross-compiled but not exercised by these mock tests. Thirteen host
CTest cases pass with the sanitizer build; none constitutes GPU evidence.

## Dependency provenance

The SDK fork is `mihawk-99/PS5_PayloadSDK` at
`95c08f27386fc698f6bbe21dde3030140a41d10b`. Its default license and the platform
headers/library are GPL-3.0-or-later. FreeBSD headers and imported components have
their own notices (including BSD and MIT components). The ignored SDK export keeps
those notices. No SDK binary, proprietary library implementation or firmware dump
is added to Git. The project has not selected a distribution license or published
native binaries. This dependency selection must remain visible when that decision
is made; it must not be described as an entirely MIT-licensed SDK.
