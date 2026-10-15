# Rebuild from GitHub source on another computer

Updated: 2026-10-14. This describes a **fresh source build**, with no copied
`.local`, `.deps`, `build`, `dist`, or `results` directory. It does not transfer
or recreate the October 14 deployment receipt. Read `../AGENTS.md`,
`../STATE.md`, and `../HANDOFF.md` before continuing the project.

## Current answer

`make deps-build` now fetches and verifies external source and archives, then
builds the SDK, zlib, UFS2Tool, bundled Lapy helper, and controllers.
`make host-test` runs host tests; `make native-core` cross-compiles the core.
ProsperoEden and ps5-homebrew-dev-protocol are locked snapshots of public
`main`, so their old local-only commits and dedicated branches are not required.
The clean-clone dependency build completed, but its helper ELF digest differed
from the original workspace build. These targets prove source availability and
buildability, not identical bytes for the installed package.

The private GitHub branches contain the source and build scripts, but **they do
not yet provide an end-to-end, tested, source-only rebuild of the exact
PPSA99014 public HIP bootstrap image**. The final build scripts consume several
earlier generated native receipts, source trees, compiler products, controller
ELFs, and package files. Their producers are spread across prior experiments;
there is no checked-in topological build command that recreates the whole chain
from clean checkout. Do not claim that the commands below reproduce image
`836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8`
until a clean-machine build actually produces and verifies that digest.

The exact D receipt for the image already registered on `.40` records a real,
one-use hardware event. **It cannot be built from source.** Source-only work can
produce a new candidate; it cannot safely resume execution E against the old D
without the original receipt and independent review. Do not synthesize those
files or rerun D to make them appear. If a new package differs from the installed
image, treat it as a new deployment candidate with fresh review and controls.

## 1. Get the published source

In WSL/Linux, with access to the private repository:

```sh
git clone --branch main \
  git@github.com:blackbearreloaded/ps5-hip.git ps5-hip
cd ps5-hip
git clone --branch development/g5-hip-native-link \
  git@github.com:blackbearreloaded/ps5-hip.git .local/g5-hip-native-link
git clone --branch development/g5-native-public-memory \
  git@github.com:blackbearreloaded/ps5-hip.git .local/g5-native-public-memory
git clone --branch development/gemm-native-bridge \
  git@github.com:blackbearreloaded/ps5-hip.git ../ps5-hip-gemm-bridge
```

The nested bootstrap location is deliberate: its scripts compute the main
repository root from `.local/g5-hip-native-link`. The memory and GEMM clones
are for later gates; only the bootstrap branch is needed for the first native
startup build. Check the exact commit IDs in `../HANDOFF.md` before modifying a
candidate. The reviewed bootstrap *source* commit is
`ea37055286903f680e43b049d70730743ffa2358`; its later branch commits
contain inert D/E controls and deployment evidence. The packaging script writes
the current checkout commit into its manifest, so use the source commit when
attempting to reproduce the reviewed package.

## 2. Prepare pinned external inputs

Use `../dependencies.json`, `../elevation-inputs.json`, and
`../controller-inputs.json` as the authority for URLs, revisions, and hashes.
Install and verify LLVM/LLD **18.1.8** (`clang-18`, `ld.lld-18`,
`llvm-config-18`, `llvm-readelf-18`, `llvm-objdump-18`), Python 3, CMake,
Ninja, `patch`, and .NET SDK **10.0.112** in WSL/Linux. The native tools and
SDK inputs are fetched from their pinned public Git repositories, rather than
from the installed console or a proprietary SDK. UFS2Tool restores with the
committed NuGet package lock in locked mode. Run:

```sh
make deps-build
make host-test
make native-core
```

The commands below describe individual preparers; the Makefile passes their
locked source and archive paths automatically.

Fetch the PS5 PayloadSDK fork at commit
`95c08f27386fc698f6bbe21dde3030140a41d10b`, and the v0.42 release
archive. Verify the archive against `dependencies.json`, then run the existing
offline preparer:

```sh
python3 tools/prepare_sdk.py \
  --sdk-source /path/to/pinned/PS5_PayloadSDK \
  --release-archive /path/to/ps5-payload-sdk.zip
```

Fetch UFS2Tool at `b5307a60d5b4e3a68ba680e0e33cfadf05017c77` and,
after installing its locked .NET SDK and restoring its dependency cache, run:

```sh
python3 tools/prepare_ufs2tool.py --source /path/to/pinned/UFS2Tool
```

The native converter uses pinned PS5_Vulkan revision
`3f3ee69607013b345d2baa6d6a37c86745649a08` and host zlib 1.3.2.
The title's bundled elevation helper comes from the pinned ProsperoEden and
PS5-Lapy-JB-Daemon revisions in `elevation-inputs.json`, plus the checked
ps5log header. Launch/close controller source comes from the pinned
ps5-homebrew-dev-protocol revision in `controller-inputs.json`. These are
additional GitHub sources; access to `ps5-hip` alone does not include them.
Their existing preparation entry points are `tools/prepare_host_zlib.py`,
`tools/prepare_elevation.py`, and `tools/build_controllers.py`; see
`NATIVE_BUILD.md` and `CONSOLE_WORKFLOW.md` for the arguments and authority
checks. Do not substitute another version and retain the old evidence claim.

Fetch upstream ROCm/clr at
`65d174c3e35423bf25c9c369513780c9d9ca760c` into
`results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c`.
The exact upstream-file hashes in `ports/clr/manifest.json` and the public HIP
patch manifest reject an incorrect checkout. This is a **source tree** for the
build, not a previously qualified runtime binary.

## 3. Close the generated-prerequisite gap

Before the final bootstrap commands can run, a clean checkout must regenerate
at least these products at the paths the existing scripts consume:

| Generated input | Used by | Existing producer or source |
| --- | --- | --- |
| `.deps/sdk`, `.deps/ufs2tool-<revision>`, native converter and AGC stubs | compile, convert, package | Pinned preparation scripts above; converter/stubs also come through the native integration build. |
| `.local/g4-native-workers/results/workers-compiler-v2/ps5/threads.hsaco` | genuine ROCclr GPU title | Worker compiler experiment; its build receipt must be regenerated. |
| `.local/g4-clr-cpu-lifetime/results/native-title-lifetime-v1` and `results/clr-adapter-native-compile-v1` | `tools/build_clr_gpu_title.py` | CPU lifetime and adapter native builds. |
| `.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2` | AGC stubs, converter, elevation | Multi-reduction native integration build. |
| `.local/g5-clr-gpu-startup/results/source-v1/source`, `results/native-title-v1/{commands.json,receipt.json}` and its `dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42` package | public HIP source, 74-object compiler flags, package shared files | `tools/prepare_clr_gpu_source.py`, `tools/build_clr_gpu_title.py`, `tools/convert_clr_gpu_title.py`, `tools/package_clr_gpu_title.py` at the qualified GPU source state. |

`tools/build_clr_gpu_title.py` itself binds fixed hashes of the older CPU,
adapter, and worker receipts. The public HIP link then derives its compile/link
commands from the generated GPU `commands.json`, rather than a checked-in
toolchain preset. Reproducing those path-bearing receipts on another machine
may change receipt digests even if the code is equivalent. The source-only
rebuild task is to make this chain explicit and relocatable, rebuild every input
from pinned source, and replace old fixed receipt identities with freshly
reviewed identities. Keep the old evidence immutable; create new evidence for
the new build. A fresh clone is **not ready for step 4** until these inputs exist
and their producers have passed offline checks.

## 4. Build the public HIP bootstrap after step 3

The final-stage recipe is also in the published bootstrap branch's
`docs/HIP_NATIVE_BOOTSTRAP.md`. From the main repository root, after all paths
in step 3 have been regenerated and checked, use fresh result directories:

```sh
git -C .local/g5-hip-native-link checkout --detach \
  ea37055286903f680e43b049d70730743ffa2358
python3 .local/g5-hip-native-link/tools/prepare_hip_admission.py \
  --base-source .local/g5-clr-gpu-startup/results/source-v1/source \
  --hip-source results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c \
  --output .local/g5-hip-native-link/results/native-source-v1
(
  cd .local/g5-hip-native-link/results/native-source-v1
  patch --batch --forward -p1 -i ../../../../ports/clr/ps5-hip-correlation.patch
)
python3 .local/g5-hip-native-link/tools/guard_native_hip_gl.py \
  .local/g5-hip-native-link/results/native-source-v1
python3 .local/g5-hip-native-link/tools/probe_native_public_hip_link.py \
  --output .local/g5-hip-native-link/results/native-hip-compile-all-v2 --all
python3 .local/g5-hip-native-link/tools/link_native_public_hip_bootstrap.py \
  --output .local/g5-hip-native-link/results/native-hip-console-clean-link-v2 --console-title
python3 .local/g5-hip-native-link/tools/convert_native_public_hip_bootstrap.py \
  --link .local/g5-hip-native-link/results/native-hip-console-clean-link-v2 \
  --output .local/g5-hip-native-link/results/native-hip-console-clean-convert-v2 --console-title
python3 .local/g5-hip-native-link/tools/verify_native_public_hip_bootstrap.py --console-title
python3 .local/g5-hip-native-link/tools/package_native_public_hip_bootstrap.py \
  --results-dir-source .local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42/app/results-dir.txt
```

The strict link must show **74 freshly compiled objects and zero reused
objects**. Packaging consumes the reviewed GPU package's `results-dir.txt`,
verifies UFS2 extraction, and writes
`dist/PPSA99014/gfx1013-public-bootstrap-<eboot-hash>/PPSA99014.ffpkg`.
Run the focused bootstrap tests and obtain an independent package review.

Compare the rebuilt image SHA-256 with the reviewed image in `../HANDOFF.md`.
An exact match still does not recreate the historical D receipt. A different
hash is a **new** package, regardless of whether its source is similar. Neither
case authorizes execution E against the old D from source alone. Only a
separately reviewed, fresh console plan may change the installed title.

## What must be implemented for a true one-command source build

The next agent should add one offline build target that fetches/verifies the
pinned external sources, regenerates the table above in dependency order,
builds all 74 units, converts/packages the title, and records relocatable
source/tool/object hashes. Test it from a disposable fresh clone with empty
ignored directories and no original-machine files in `PATH`. The acceptance
test is a completed package, exact extracted-file inventory, zero reused
objects, and a compact independent review. Console traffic is outside this
build target and must remain a separately controlled milestone.
