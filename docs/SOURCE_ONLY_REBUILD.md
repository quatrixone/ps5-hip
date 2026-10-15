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

`tools/freeze_source_candidate.py` copies a completed build, controllers and
the separately built read-only snapshot into a new `results/` candidate and
binds their hashes. The source tree must be committed first. The current
console control is scoped to `.40`, PPSA99014 and `/data/ps5-hip/results`;
build with that explicit `RESULTS_DIR` when preparing this case.
Supply `--baseline-image-sha256` for the exact installed image and, when
preserving an existing bootstrap log, `--baseline-log-sha256`. Deployment
rejects changed bytes and preserves both under unique digest-named backups.
An absent baseline log digest requires that no prior bootstrap log exists.

The converter is built from pinned source plus
`patches/native-import-libraries.patch`. Before signing, a serialized SCE
import audit checks module/library bindings, including the shell-exit import.
The first source-Make hardware attempt exposed this missing patch and failed
teardown; see `evidence/native-source-make-bootstrap-failure.json`.

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
