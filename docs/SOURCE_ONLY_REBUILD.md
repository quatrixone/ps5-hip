# Build the native public HIP package from GitHub source

The supported source and build recipe live on the private `main` branch. A
fresh WSL/Linux clone needs no copied `.local`, `.deps`, `build`, `dist`, or
`results` directory and no checkout of another ps5-hip branch.

Install Git, Python 3, GNU Make, CMake, Ninja, `patch`, LLVM/LLD **18.1.8**
(`clang-18`, `clang++-18`, `ld.lld-18`, `llvm-config-18`, `llvm-readelf-18`,
`llvm-readobj-18`, `llvm-objdump-18`), and .NET SDK **10.0.112**. The Makefile
checks the versions its builders consume. NuGet access is needed for the
locked UFS2Tool restore.

```sh
git clone --branch main git@github.com:blackbearreloaded/ps5-hip.git
cd ps5-hip
make RESULTS_DIR=/absolute/console/results/path
```

`make` fetches and verifies the locked upstream commits and archives, builds
the SDK, host zlib, UFS2Tool, Eden/Lapy helper, controller ELFs, host device
fixtures, and native core, runs the host suite, and builds the public HIP title.
Its package is `build/public-hip/package/PPSA99014.ffpkg`; the exact inputs,
74 freshly compiled object hashes, native PIE, signed executable, package
hash, and six extracted file hashes are in
`build/public-hip/link-receipt.json`. `make public-hip-package` runs just the
dependency and package path. All console-facing output is offline and marked
`native_executed: false`.

After source changes, preserve earlier build evidence by selecting a fresh output:
`make PUBLIC_BUILD_DIR=build/public-hip-v2 RESULTS_DIR=/data/ps5-hip/results`.
The bootstrap target always invokes the builder, which validates cached inputs and
rejects stale artifacts. Memory and copy targets likewise accept `MEMORY_BUILD_DIR`
and `COPY_BUILD_DIR`; module, launch and fill use their corresponding overrides.
Default directories remain unchanged. These overrides do not delete old outputs.

A fresh GitHub-only clone of `main` at `7ef291c` passed the default `make`
on 2026-10-14: 54 host tests, the native core, 74 public HIP objects, strict
native link, FSELF extraction audit, UFS fsck/extraction, and cached
revalidation. The compact offline receipt is
`evidence/offline-main-source-build.json`. Its package SHA-256 is
`051e326a191049a344c3cbf7c173357e7375c863a60a44406f635695ddcba923`;
this hash applies to that exact source and build environment.

`RESULTS_DIR` is written into `results-dir.txt`. The default is
`/user/homebrew/ps5-hip`; choose a path that actually exists and is writable
on the target console before any deployment. To change it or rebuild after
source changes, preserve any package or receipt you need, then run
`make clean-build` followed by `make RESULTS_DIR=...`.

The build uses a SHA-locked ROCclr and HIP header revision, plus public
ProsperoEden and ps5-homebrew-dev-protocol `main` snapshots. A lock hash is
for repeatability; it does not require an unpublished commit or a maintained
fork. Generated source, compiler objects, converter, AGC link stubs, clean-room
libc, Lapy helper, and title metadata are produced from committed recipe/source
and the fetched inputs. The six packaged files are `eboot.bin`, `lapy.elf`,
`results-dir.txt`, `sce_module/libc.prx`, `sce_sys/param.json`, and
`sce_sys/icon0.png`.

This creates a **new candidate**. The package already registered on `.40`
(`836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8`)
and its one-use deployment receipt are historical hardware evidence. They
cannot be recreated by Make, and a new package must not be run under the old
execution control. Compare the new candidate's hashes, perform independent
package/control review, then plan any console experiment separately under
`AGENTS.md` and the shared-console runbook. Building this package does not
prove public HIP startup or GPU execution.

## Fresh deployment controls

`make public-hip-fill-package RESULTS_DIR=/data/ps5-hip/results` builds the
native contiguous and pitched-fill test in `build/public-hip-fill`; use `FILL_BUILD_DIR` for
a fresh output directory. The six-file package has no GPU-driver imports. Its
client checks two changed-pattern rounds, byte/16-bit/32-bit fills, all guards,
16 rejected inputs and exact allocation/queue retirement. The pitched cases cover
2D row padding, 3D slice gaps and equal pitch/width with extra slice rows. The strict
expected transcript contains 4,096 byte comparisons. Packaging does not establish console
execution; the normal freeze/review/recovery requirements still apply.

`make public-hip-launch-package RESULTS_DIR=/data/ps5-hip/results` builds the
native public kernel-launch fixture in `build/public-hip-launch`; set
`LAUNCH_BUILD_DIR` to a new directory to preserve an earlier attempt. It packages
`thread_words.hsaco` with the title and verifies all seven extracted files.
The fixture loads the module through public HIP, performs two changed-input
launches, checks146 active values and16 tail guards, then checks exact resource
cleanup. Its strict transcript also requires both unsupported-block rejections.
The native artifact requires independent review, a committed frozen candidate
and a separately accepted .40 cycle before any console GPU result is claimed.

`make public-hip-module-package RESULTS_DIR=/data/ps5-hip/results` builds
74 fresh native objects and a seven-file package in `build/public-hip-module`.
The additional `canary.hsaco` is compiled from `kernels/canary.hip` for gfx1013
and checked byte-for-byte after UFS extraction. The fixture opens two modules
simultaneously in each of two rounds, checks distinct function handles, unloads
one module, verifies the survivor, and unloads it. Its backend factory refuses
GPU backend creation; any queue, allocation, launch or uncertain state fails.
The strict module transcript requires four loads, eight lookups, four unloads
and clean finalization. This is a loader ownership test, not kernel execution.
The package and controls still require exact-candidate review and a separately
accepted console cycle before native module support is claimed.

`make public-hip-copy-package RESULTS_DIR=/data/ps5-hip/results` builds the
contiguous and rectangular-copy fixture into `build/public-hip-copy`. Its PS5-specific
HIP patch preserves the caller's synchronous request through D2D queue selection
and returns success only after a successful completion wait and `CL_COMPLETE`.
An uncertain result must retain all host/device buffers and quarantine; the
fixture uses persistent host storage and never proceeds after such an error.
The fixture also issues one asynchronous 3D D2D copy per round and synchronizes
the device before readback; native execution remains unqualified.

The copy oracle checks two changed rounds of H2D/D2D/D2H copies, 257-byte
payloads with guards, 2D transfers with distinct pitches and 3D transfers with
nonzero origins and slice gaps. It compares 7,168 bytes and requires 16 exact
rejections, including unchanged output after a synchronous overlap error. It
requires two frees, one retired queue and clean shutdown. The backend factory
denies GPU creation, and package verification requires GPU imports to be absent. The
completion regression compiles the actual upstream function with mocked
completion under ASan/UBSan; an unpatched negative control must fail. This is
offline control-flow evidence, not native transfer or GPU DMA evidence.

The next public HIP fixture is available with
`make public-hip-memory-package RESULTS_DIR=/data/ps5-hip/results`. It builds
74 fresh objects into `build/public-hip-memory` and exercises genuine
`hipMalloc`/`hipFree` for 257-byte and 4097-byte buffers, rejects interior and
double frees, and requires complete shutdown. Its exact-hash preparation adds
the reviewed HIP memory guards over the already-adapted SVM implementation.
The `memory` receipt/manifest fixture selects a separate strict log oracle;
bootstrap receipts and transcripts cannot satisfy it. This recipe is offline
until its own package and console cycle are qualified. The `be6e521` candidate
passed the independently reviewed bounded allocation/free cycle on .40; see
`evidence/native-source-public-memory-success.json`. It makes no data-copy or
GPU-dispatch claim.

`tools/freeze_source_candidate.py` copies a completed build, controllers and
the separately built read-only snapshot into a new `results/` candidate and
binds their hashes. The source tree must be committed first. The current
console control is scoped to `.40`, PPSA99014 and `/data/ps5-hip/results`;
build with that explicit `RESULTS_DIR` when preparing this case.
Supply `--baseline-image-sha256` for the exact installed image and, when
preserving an existing bootstrap log, `--baseline-log-sha256`. Deployment
rejects changed bytes and preserves both under unique digest-named backups.
Select the prior fixture with `--baseline-log-fixture`. An absent baseline log
digest requires that the new fixture's result log is absent.

If a failed deployment already preserved the baseline log, a distinct candidate
may use `--baseline-log-preserved` together with its exact digest and fixture.
This verifies the SHA-suffixed `.preserved` file, requires the original and new
result paths to be absent, and rechecks the preserved bytes before image
replacement and after settling. It never renames that log again. This option
does not authorize replaying a consumed deployment or bypass image verification;
establish current installed-image integrity and obtain independent review first.
FTP receipts now record each read's received bytes, elapsed time and SHA-256,
including failures. A partial read hash is not installed-image verification.

The converter is built from pinned source plus
`patches/native-import-libraries.patch`. Before signing, a serialized SCE
import audit checks module/library bindings, including the shell-exit import.
The first source-Make hardware attempt exposed this missing patch and failed
teardown; see `evidence/native-source-make-bootstrap-failure.json`.
The corrected `7342726` build passed all 56 host tests and the independently
reviewed bounded initialization/device-count/clean-shutdown test on .40; see
`evidence/native-source-make-bootstrap-success.json`. This does not qualify
public HIP allocation or GPU execution.

`tools/source_console.py` requires independent review bound to the frozen
manifest before any socket is opened. Its separate one-use `deploy` phase
preserves the exact historical image, waits for its unmount, promotes the
verified new package, waits two minutes and verifies mounted bytes. The
`execute` phase additionally requires review of that new deployment receipt.
It checks fresh results, observes one bounded public HIP startup, preserves
klog and application evidence, compares read-only mount snapshots, and uses
only the exact-title close controller if cleanup is necessary. It never
replays historical D/E controls. Review and current environment/lock checks
remain required; these commands are deliberately outside the Make build.

For full public HIP host-runtime qualification, run
`make public-hip-host-test QUALIFICATION_DIR=results/public-hip-host-v1`.

The qualifier also links a separate public launch client against the same freshly
compiled runtime. It builds `kernels/thread_words.hip` for gfx1013, tests rejected
launch/capture/graph requests before submission, then checks 73 exact outputs,
eight guards, argument capture after caller mutation, and final resource ownership.
Two negative controls simulate missing and incorrect computation; each must fail
at the output oracle. Providers and packet execution are modeled, so passing this
case is not evidence of GPU execution on a PS5.

The `launch-events` case records a timing-disabled event after a held modeled
dispatch, requires `hipErrorNotReady` before release, then successful event
synchronization/query and destruction with clean ownership. It retains device
synchronization before readback, so it does not independently qualify blocking
event synchronization, timestamps or cross-stream ordering.

A separate stream harness tests two nonblocking logical streams with a
timing-disabled event. While a producer host callback is held outside the adapter
gate, `stream-normal` requires no dependent consumer progress within one second.
`stream-bypass` omits the event wait and requires early consumer progress within
that deadline. Both cases release the producer and verify two retired queues with
no allocation or GPU submission. This bounded test is scheduler-dependent; it
qualifies neither native events nor GPU concurrency. The qualifier builds 79
translation units in total: 69 base units, six alternate client harnesses,
two independently fault-injected memory units and two retirement adapter copies.

Four additional stream cases exercise legacy default-stream ordering in both
directions: explicit producer to null-stream consumer and null-stream producer to
explicit consumer. An ordinary (`hipStreamDefault`) stream must hold the consumer
until the producer callback is released; a `hipStreamNonBlocking` stream must let
the consumer finish while that callback remains held. These cases insert no
explicit event wait. Each checks post-synchronization stream queries and exact
retirement of both queues. The one-second observation and five-second producer
watchdog bound scheduling; they do not prove GPU concurrency or native timing.
Per-thread default-stream behavior is outside this fixture's scope.

The stream client also tests actual asynchronous D2D dataflow. A held producer
copies a 128-byte seed into an intermediate allocation, records an event, and a
consumer copies into a third allocation. With the wait, readback must equal every
seed byte; bypassing it must read the intermediate buffer's original zeros. The
output starts at a different value, so an omitted copy cannot pass either oracle.
Both streams synchronize before readback and each buffer is freed once before
stream destruction. These cases retain the bounded scheduling caveat above and
do not establish native asynchronous transfer or GPU concurrency.

The host-graph client builds two dependent host nodes both explicitly and through
thread-local stream capture. Each executable replays 16 changing inputs, executes
exactly 32 callbacks, updates callback functions and user-data pointers after four
replays, then survives destruction of its source graph after eight. Capture and
instantiation must not execute callbacks. Poisoned intermediates and outputs reject
omitted or reordered computations; a no-op producer control must fail the output
oracle on the first replay. Three logical queues remain constant across replay and
retire exactly, with no device allocation or GPU submission. ASan leak detection
covers normal cleanup. This does not qualify kernel/memory graph nodes, capture
invalidation, concurrent graph updates, host-memory plateau or GPU replay speed.

The dedicated native graph fixture is built with
`make public-hip-graph-package GRAPH_BUILD_DIR=build/public-hip-graph`.
It shares `tools/public_hip_graph_fixture.h` with the host qualifier and runs the
explicit and recovery cases sequentially: two executables, 32 changed-input replays,
64 callbacks, six invalidations and three caller-owned graph checks. Callback data
has static lifetime; uncertain native failure flushes the log and quarantines.
Exactly four of five queues retire before main returns, and all five must retire
before the final transcript. The no-GPU import policy and denied backend factory
remain enforced. A 30-second watchdog bounds the fixture. This is a prepared
console test, not native graph acceptance; deployment still needs a freshly frozen
candidate, recovered console baseline and independent operational review.

Capture recovery also tests the error path after forbidden stream synchronization.
Ending an invalidated capture clears stream state, discards its internally owned
graph and returns a null graph with `hipErrorStreamCaptureInvalidated`. A graph
supplied by the caller remains caller-owned. Starting over before ending the failed
capture is rejected, and repeated end calls produce a deterministic null output.
The regression covers six origin-stream cases across global, thread-local and
relaxed modes, then successfully captures and replays on the same stream. Borrowed
graphs remain queryable and are destroyed by the client. Forked child/event capture
recovery and invalidation of other rejected APIs remain unqualified.

The rectangular-copy client checks public synchronous 2D and asynchronous 3D D2D
copies with changed data, interior pointers, nonzero origins and different row/slice
pitches. Two rounds compare 4,096 bytes including padding and source preservation.
Admission rejects invalid origin/extent geometry and overflow before command creation;
synchronous command failure propagates as `hipErrorUnknown`. Host rectangles, arrays,
capture and graph rectangles remain unsupported and explicitly reject on the qualified
entry paths. Invalid async streams and null 2D descriptors reject without dereference.
The native memory owner checks complete spans and actual row overlap before writes.
This is host qualification plus targeted PS5 compilation, not native copy acceptance.

The retirement client holds a completed fill before worker batch release. The
fixed adapter must keep synchronization pending during a 500 ms observation,
then free the allocation once before stream destruction. The negative control
disables only forced finish-marker admission and must return early during the
same hold; it destroys the stream before freeing to clean up safely. Both retire
exactly one queue. Hooks exist only in generated adapter copies, and their source
and binary hashes are recorded. The absence-of-return observation is scheduler
bounded; this is host regression evidence, not native retirement acceptance.

The fill client checks contiguous `hipMemset` and word-pattern operations using
all 128 expected bytes, including unchanged guards. Synchronous calls now wait
and return command failures; async calls enqueue on the selected stream. Invalid
alignment and overflowing word counts are rejected before enqueue. Pitched 2D/3D
fills validate the complete touched span and preserve row/slice padding; the
additional case checks 2,048 bytes across changed rounds. Allocation failures are
injected separately at vector reservation and command construction, requiring
out-of-memory, unchanged bytes and cleanup. Capture and graph memset nodes remain
unsupported. Synchronization includes retirement of prior command references.
The source builder applies these guards to every package fixture. Host models
and PS5 compilation establish no console fill result.

The current public adapter admits 64x1x1 blocks, one-dimensional grids and no
shared memory. Packed launch buffers and kernel graph/capture routes return an
explicit unsupported error pending their implementation and qualification. These
limits describe this public adapter stage; broader geometry, LDS, graphs and
framework support remain in the roadmap.
Use a new results directory for each run. This builds 69 genuine HIP/ROCclr and
adapter translation units from the pinned sources, with ASan/UBSan, then runs
twelve separate initialization, thread-lifetime, shutdown, allocation and module cases.
The module modes build gfx1013/gfx1030 fixtures, check owned file loading, target
rejection, function lookup, independent modules and unload under both effective
runtime-unbundler settings. Pointer-only image loading remains unsupported.
It needs Linux Clang/LLD 18 and no historical workspace receipts or binaries.
The runner verifies dependencies, records included files and toolchain hashes,
and preserves each compile/run log. Leak detection is enabled for normal cases;
the three intentional retention cases instead check exact retained ownership.
Platform providers and packet transport are modeled: this does not qualify PS5
execution, real GPU work, native public module loading or the pending console copy case.

Before freezing a console candidate, run `make console-controls`. This builds
launch/close controllers and the read-only teardown helper from pinned source;
it sends no payload and makes no console connection. The snapshot defaults to
`build/source-make-snapshot`. To preserve an earlier build after source changes,
use a new `SNAPSHOT_BUILD_DIR=build/<new-directory>` and pass that directory to
`tools/freeze_source_candidate.py --snapshot-build`. The snapshot builder refuses
to overwrite an existing attempt. A built helper does not itself certify console
health or replace the required candidate and deployment reviews.

## Explicit-stream GPU fixture

`make public-hip-stream-package STREAM_BUILD_DIR=build/public-hip-stream RESULTS_DIR=/data/ps5-hip/results`
builds a seven-file package with the same `thread_words` kernel. Two nonblocking
streams test a producer GPU launch followed by an event-dependent D2D snapshot
on the consumer. Two changed-input rounds reuse the event, compare 146 outputs
and 16 destination guards, verify the source, and require two allocations/frees
and three retired queues. Host callbacks make pending-event and stale-copy cases
observable; the bounded nonprogress observation is not a concurrency measurement.

The shared host/native oracle has bypass, no-write and corrupt-write negative
controls in the source qualifier. Host-model evidence does not establish native
GPU execution. Freeze and independently review a fresh candidate before console
use; consult the current `STATE.md` baseline rather than reusing an old receipt.

## Explicit GPU kernel graph qualification

The source qualifier includes `tests/public_hip_graph_gpu_model.cpp`. It exercises
real HIP/ROCclr graph construction and command replay with modeled GPU transport:
two changed-salt kernel replays, deep-copied arguments, executable parameter
updates and replay after destroying the source graph. The module and device
allocation remain alive until the executable graph has been destroyed and its
work synchronized; graph argument copies do not transfer ownership of those resources.

PS5 graph replay uses the command path. ROCr AQL packet capture is disabled for
this backend regardless of the upstream debug flag, because dispatch uses PM4.
Explicit kernel nodes use the existing bounded launch geometry and argument ABI;
extended module-launch capture and 3D memcpy graph admission remain unsupported.

Fault injection targets the graph argument table and each of its three argument
allocations during add, clone, instantiate, child-graph construction and parameter
updates. Failed updates preserve the old arguments. Command-construction failure
is exercised after a successful replay and inside parallel and child graphs;
unsubmitted commands must be released and later callbacks must not execute.
Disabled kernel nodes become dependency-preserving markers and can be re-enabled.
These cases do not establish general allocation-failure coverage, resource-pool
plateaus, graph submission efficiency or native GPU graph execution. A shared
native oracle and a fresh reviewed console candidate are still required.

### Native explicit GPU graph fixture

Build with `make public-hip-graph-gpu-package
GRAPH_GPU_BUILD_DIR=build/public-hip-graph-gpu-v1
RESULTS_DIR=/data/ps5-hip/results` (one shell command). The package includes the
same `thread_words.hsaco` as the previously qualified stream fixture. Host and
native clients share `tools/public_hip_graph_gpu_fixture.h`: two disabled replays
must still run their dependent callbacks, then two changed-salt GPU replays must
produce 146 exact values and preserve 16 guards. Arguments are overwritten after
graph creation and executable update; the source graph is destroyed before the
second GPU replay. The native title requires four callbacks, two actual dispatches,
one freed allocation and three retired queues at finalization.

The new fixture uses `ps5-hip-public-graph_gpu.log`. A fresh frozen candidate and
independent package, deployment and execution reviews are required; an older
host-graph package does not exercise GPU graph nodes. The bounded native fixture passed on .40; see
`../evidence/console40-public-gpu-graph-success.json`. Its frozen candidate phases
are consumed and cannot be replayed. Use STATE.md for the current image/log baseline.

### Captured module-kernel graph fixture

`make public-hip-capture-gpu-package` builds the standard `hipModuleLaunchKernel`
stream-capture fixture. Override `CAPTURE_GPU_BUILD_DIR` for a fresh build and set
`RESULTS_DIR=/data/ps5-hip/results` for the dedicated .40 environment. The fixture
shares the explicit graph replay oracle, but constructs its kernel and dependent
host node through stream capture. It checks two disabled replays, then two
changed-argument launches, update, source destruction and complete retirement.

On this backend, failure while adding a captured module kernel invalidates its
capture session; `hipStreamEndCapture` returns an invalidated error with no output
graph. Caller-owned capture graphs remain owned by the caller. The model checks
36 cases across three capture modes, both ownership forms, unsupported geometry,
extra-buffer arguments and each argument-copy allocation failure. Extended/HCC
launch capture remains rejected. GPU memory graph nodes, cross-stream GPU capture
and longer replay/lifetime measurements are separate open requirements.


### One-dimensional copy graph retirement

The PS5 copy graph path creates a dependency marker for a disabled 1D copy
during command preparation. That same marker receives graph wait lists and is
enqueued during replay. It does not construct an unused copy command that would
retain device buffers. Command-vector capacity is reserved before marker allocation.

The source qualifier exercises explicit and captured H2D/D2D/D2H chains, each
disabled direction with distinct sentinels, re-enabled changing-input replay and
source-graph destruction. A parallel fork/join case checks that a held host
predecessor prevents the joined callback from completing, then verifies completion
and bytes after release. Its 100 ms observation is bounded scheduler evidence.
All buffers and queues must retire under the sanitizers.

This host-model coverage does not qualify native copy graphs, invalidated copy
capture recovery, broader memcpy/memset node types or resource plateaus. A dedicated
shared native copy/kernel fixture remains required before console acceptance.


### Captured copy/kernel graph fixture

Build a dedicated .40 package with:

```sh
make public-hip-copy-gpu-package COPY_GPU_BUILD_DIR=build/public-hip-copy-gpu-v2 RESULTS_DIR=/data/ps5-hip/results
```

The host/native shared oracle captures H2D and a kernel on the origin stream,
then uses an event to join a consumer stream for D2D and D2H copies. A second
event rejoins the origin before its callback. It checks the five-node dependency
graph and resets both capture states, then reuses the events in eager execution.
It first checks 27 failed-copy capture/recovery cases across three
capture modes, four invalid setter calls and two invalid generic-update handles.
The same helper runs in the source qualifier and the native title; the transcript
requires these counts. A compatible generic graph update must also succeed.
Six preliminary replays disable each copy direction while the kernel
is disabled, checking distinct sentinels. Two changing-input rounds compare
eager and graph execution against exact values and guards, including owned
arguments, executable kernel update and replay after source-graph destruction.

Both device buffers are poisoned after the eager control so stale eager output
cannot hide a missing graph copy. The host qualifier rejects omitted H2D, D2D
and D2H edges, no-op kernel writes and corrupt writes. Acceptance requires four
GPU dispatches, 292 exact active values, 32 guards, eight callbacks, two freed
allocations and four retired queues. The exact transcript is
`ps5-hip-public-copy_gpu.log`; a fresh frozen candidate and independent review
are required before a bounded console case. This producer/consumer case does
not prove concurrent GPU dispatch, general multi-kernel capture or sustained
replay efficiency. Hardware validation of this joined variant is pending.

The source qualifier also checks queued graph copies followed by `hipFree`
while the executable and stream remain alive, and executable parameter migration
to replacement allocations before freeing the old buffers. It verifies exact
output, allocation removal and final cleanup. The gated free test observes a
bounded interval; it cannot prove the freeing thread entered the runtime before
the gate was released. These are sanitizer host-model checks, not native
allocation-lifetime acceptance. Buffers must remain valid for future replays.

The bounded copy/kernel fixture passed on .40 at `ac93df8`; see
`../evidence/console40-public-copy-kernel-graph-success.json`. Its candidate phases
are consumed. Future experiments require a fresh frozen candidate.
