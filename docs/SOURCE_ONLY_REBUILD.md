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
next synchronous-copy fixture into `build/public-hip-copy`. Its PS5-specific
HIP patch preserves the caller's synchronous request through D2D queue selection
and returns success only after a successful completion wait and `CL_COMPLETE`.
An uncertain result must retain all host/device buffers and quarantine; the
fixture uses persistent host storage and never proceeds after such an error.
Explicit asynchronous requests keep upstream behavior and remain unqualified.

The copy oracle checks two changed rounds of H2D/D2D/D2H copies, 257-byte
payloads between 16-byte guards, 1,734 device-readback bytes, host/source
preservation and six exact `hipErrorInvalidValue` bounds rejections without
mutation. It requires two frees, one retired queue and clean shutdown. The
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
