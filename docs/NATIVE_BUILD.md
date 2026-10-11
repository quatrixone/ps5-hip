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
- `libps5hip_native_memory.a`: direct-memory allocation and ownership logic.
- `native-memory-smoke`: a separate, opt-in CPU direct-memory write/read test.

The smoke executable requires `--memory-only` to do anything. It contains no AGC
initialization or GPU submission. It was **not run on a console**. The current SDK
linker produces a payload-style ELF64 x86-64 shared object, not a packaged native
title. Static inspection showed 16 KiB PT_LOAD alignment and imports from
`libkernel_web.sprx`, `libSceLibcInternal.sprx` and `libSceNet.sprx`. The SDK's first
load segment is RWX. Dedicated-title linking/packaging and its startup contract
remain separate work before an AGC launch experiment.

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

## Dependency provenance

The SDK fork is `mihawk-99/PS5_PayloadSDK` at
`95c08f27386fc698f6bbe21dde3030140a41d10b`. Its default license and the platform
headers/library are GPL-3.0-or-later. FreeBSD headers and imported components have
their own notices (including BSD and MIT components). The ignored SDK export keeps
those notices. No SDK binary, proprietary library implementation or firmware dump
is added to Git. The project has not selected a distribution license or published
native binaries. This dependency selection must remain visible when that decision
is made; it must not be described as an entirely MIT-licensed SDK.
