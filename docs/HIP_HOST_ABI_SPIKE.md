# Ordinary HIP host-launch ABI spike

Recorded 2026-10-11. LLVM **18.1.8** now emits real Linux-host and PS5-host
relocatable objects for ordinary HIP launches with full **ROCm 6.2.4 headers**,
including linked indexed-vector device code for gfx1013 and gfx1030. This uses
an explicitly selected upstream workitem-library slice and device ABI flags;
the complete matching device-library build still fails. No runtime
implementation, native executable link or GPU execution is established.

The initial newer-TheRock-header experiment below remains useful failure and
ABI evidence. Its full-header/compiler mismatch is specific to that pair;
the later 6.2.4 selection supersedes it as the bounded fixture candidate.

A subsequent matching **TheRock Clang 24 / HIP 7.17** experiment emits the
full-header vector fixture for Linux and PS5 hosts with complete upstream
`ockl`/`ocml` inputs. PS5 needs an explicit `unknown` triple environment to
satisfy the newer bundler. The emitted objects still fail the current device
loader's target-string check. See the final section and
[`offline-therock-hip-host-abi.json`](../evidence/offline-therock-hip-host-abi.json).
LLVM 18 remains the selected native route.

Evidence: [`offline-hip-host-abi.json`](../evidence/offline-hip-host-abi.json).
Raw sources, exact command arrays, diagnostics, ELF inspection and objects are
preserved under ignored `results/hip-host-abi-spike/`.

## Inputs and header generation

- Compiler: Ubuntu clang/LLVM 18.1.8, package `20ubuntu8`.
- HIP/CLR: TheRock's `rocm-systems`
  `2f2a92736dbe3f65108539b6659d2c7289d300e8`.
- Native compiler: existing `tools/ps5-clangxx`, using SDK source pin
  `95c08f27386fc698f6bbe21dde3030140a41d10b`, target `x86_64-sie-ps5`,
  SDK libc++/system headers and emulated TLS.
- Device flags for combined objects: `gfx1013`, code-object version 5, wave32,
  `-B/usr/lib/llvm-18/bin`, `-nogpuinc -nogpulib`, C++17, `-O0`.
  Explicit include paths select real upstream HIP and AMD-detail headers;
  no declarations or successful runtime stubs were fabricated.

Both first attempts failed because `hip/hip_version.h` had not been generated.
The upstream generator is `projects/clr/hipamd/CMakeLists.txt:91-150,308-323`.
An exact prefix through the header write was copied into an isolated ignored
CMake project. Its version logic and header template were retained. The bounded
extraction changes were:

1. Declare CMake 3.27 and `project(hip LANGUAGES NONE)` for header-only generation.
2. Point upstream source-directory metadata queries to the real pinned hipamd
   directory, so Git reads actual commit identity and timestamp.
3. Omit upstream's unconditional `git fetch --unshallow`, preserving the pinned
   source history and avoiding network traffic.
4. Stop after writing the header, before any CLR runtime targets.

The authentic `VERSION` values and upstream Git-date calculation produced
**HIP 7.17.26405**, using Unix timestamp `1791535811`. Original extracted CMake,
modified CMake, command, output and generated-header hashes are retained. The
live TheRock build and its source checkout were not modified. This is a header
generation extraction, not successful configuration/build of the entire CLR.

## What actually compiled

The fixture contains one kernel writing `out[0] = 1` and one host function
calling `store_one<<<1,1>>>(out)`. No allocation, math-library call, thread-index
builtin or device library is needed by that body.

| Attempt | Linux host | PS5 host |
| --- | --- | --- |
| Full `hip_runtime.h`, missing generated version | Compilation fails | Compilation fails |
| Full `hip_runtime.h`, authentic generated version | Compilation fails on newer device-header builtin use | Same failure |
| Actual upstream `hip_runtime_api.h`, host-only | Object emitted; external `__hip_fatbin` remains | Object emitted; external `__hip_fatbin` remains |
| Actual upstream `hip_runtime_api.h`, combined host/device | Object emitted with embedded gfx1013 ELF | Object emitted with embedded gfx1013 ELF |

The full-header failure begins at
`amd_detail/amd_device_functions.h:199`:
`__builtin_amdgcn_is_invocable(__builtin_amdgcn_alignbit)` produces
`builtin functions must be directly called`; warp helper uses also fail.
This is a concrete mismatch between selected LLVM 18 and these newer full HIP
headers on **both** host targets. It is not evidence of a PS5-specific C++ ABI
failure. Replacing the fixture's include with the narrower public API header
isolates launch emission; it does not fix normal HIP client headers.

## Observed emitted ABI

Both combined objects have exactly these undefined symbols:

```text
__hipPopCallConfiguration
__hipPushCallConfiguration
__hipRegisterFatBinary
__hipRegisterFunction
__hipUnregisterFatBinary
atexit
hipLaunchKernel
```

ELF relocation/disassembly shows `__hip_module_ctor` registering the fat binary,
then its kernel, and registering `__hip_module_dtor` with `atexit`. The destructor
calls `__hipUnregisterFatBinary`. Linux puts the constructor in `.init_array`;
the PS5 host target puts it in **`.ctors`**. The pinned native integration already
provides a route: `ps5-pie.ld:47-51` keeps/sorts `.ctors` and `.ctors.*` into
`__init_array`, and `app_crt.cpp:61-64` executes that array. Both come from
native-tools `3f3ee69607013b345d2baa6d6a37c86745649a08`. This is existing
infrastructure; a linked HIP fixture and its registration/teardown lifecycle
remain unverified.
The device symbol is mangled `_Z9store_onePj`. Its host identity resolves through
the generated device stub, not directly to device instruction bytes.

The `.hipFatBinSegment` wrapper is 24 bytes: little-endian magic `0x48495046`,
version 1, an eight-byte pointer relocation at offset 8 to `.hip_fatbin`, and an
eight-byte zero reserved field. `.hip_fatbin` has 4,096-byte alignment and starts
with `__CLANG_OFFLOAD_BUNDLE__`. Both bundles contain:

- `host-x86_64-unknown-linux--`: zero-byte placeholder.
- `hipv4-amdgcn-amd-amdhsa--gfx1013`: the device ELF payload.

The empty Linux-labelled placeholder also occurs in the PS5-host bundle; it
must not be interpreted as the outer host ABI. The actual PS5 driver trace
selects `x86_64-sie-ps5`. Likewise, `hipv4` is the observed bundle target prefix,
not a reason to override the selected device code-object version.

The existing PS5 wrapper forwards its host library flags into device bundling:
the driver attempts to unbundle SDK libc/libc++/libc++abi/libunwind archives with
`-allow-missing-bundles`. This fixture succeeds, but a production split host/device
driver needs deliberate flag routing instead of assuming all SDK flags are
meaningful to the device compiler.

## Decision after the initial spike

The normal launch ABI is concrete enough to design registration and argument
packing tests. The later 6.2.4 experiment below selects a compatible full-header fixture
configuration; the newest fetched headers cannot simply be paired with LLVM 18.
Future emission fixtures should cover multiple translation units, templates,
device globals and the chosen default-stream mode. The current result does not cover their symbols.

Native linking/startup, module lifetime, asynchronous argument ownership and
actual dispatch remain separate work. No HIP runtime was linked, no console was
accessed, no dependencies or frozen candidates changed, and no hardware gate
passed.

## ROCm 6.2.4 compatibility experiment

A separate bounded experiment selects **ROCm 6.2.4** as a candidate for the
installed LLVM 18.1.8. AMD's [6.2.4 component table](https://rocm.docs.amd.com/en/docs-6.2.4/about/release-notes.html)
lists HIP 6.2.4 and LLVM 18.0.0. AMD's compiler is a
[fork of LLVM](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/),
so that release identity motivates this test; it does not certify Ubuntu LLVM
18 as interchangeable with AMD's compiler.

Exact release-tag source identities:

| Input | Commit | Download SHA-256 |
| --- | --- | --- |
| ROCm/HIP | `d3b75fbb82113374e2d6103baa96a9c61425da48` | `18315a21757f7302deace880fd5b6e75c46d0bc75870285e444e01ad0f502b3e` |
| ROCm/clr | `65d174c3e35423bf25c9c369513780c9d9ca760c` | `b36c0d7d7be2bda98cc62cc46cef2048fd723afa940cab79a33ea4b3558e2420` |
| ROCm/llvm-project, only `amd/device-libs` | `1e2c94795ee0d6ab8e2ff3035965a6b74e11b475` | Per-file hashes in the raw source manifest; 704 files, 1,588,598 bytes |

The two HIP/CLR archives total 3,563,943 bytes. The version generator uses
upstream CLR's actual `VERSION`, `hip_version.h.in`, exact Git commit metadata
and official-build logic. An isolated prefix of `hipamd/CMakeLists.txt` through
the generated header and `share/hip/version` writes is used, with only
`project(hip LANGUAGES NONE)` and the actual source-directory setting substituted.
This produces **HIP 6.2.41134**, Git hash `65d174c`; it satisfies the pinned ggml
client's HIP >= 6.1 check. No version macro was invented. The release version
6.2.4 and generated API/header version 6.2.41134 are different version formats.

Two driver details matter. `-nogpuinc` suppresses Clang's HIP runtime wrapper;
merely putting the real headers on `-I` paths then fails in complex-number
helpers. A directory with only headers and no authentic `share/hip/version`
also fails version-sensitive wrapper injection. With a correctly staged package,
Linux compilation injects `__clang_hip_runtime_wrapper.h`. The existing PS5 host
toolchain needs an explicit package include directory and explicit Clang wrapper
include. The SDK's standard-library arguments also need routing for device-only
IR emission (`-nostdlib -nostartfiles` for this compile-only experiment).
These are recorded driver adaptations, not changes to the SDK or dependency lock.

The newer HIP 7.17 headers were rechecked with the Clang wrapper explicitly
included: they still fail at the newer builtin use on both hosts. That original
failure remains evidence; it was not explained away by the 6.2 driver fix.

### Device-library boundary and explicit fixture inputs

The bounded 6.2.4 device-library build did **not** pass as a whole. Its original
standalone configuration encounters an absent installed `libclangBasic.a` through
Clang's CMake package. An ignored superproject imports the installed LLVM 18
executables instead; a separate source copy changes only CMake `CMP0053 OLD` to
`NEW` for the current CMake. The device helper sources are unchanged. With those
build-system adaptations, the build reaches `ockl/src/dm.cl:307,320` and fails:
`__builtin_amdgcn_ballot_w64` requires `wavefrontsize64` for the generic target.
No builtin implementation or device-library semantic repair was substituted.

To answer the launch-ABI question, a **workitem-only source slice** is built from
unchanged `ockl/src/workitem.cl` with its real upstream OpenCL compile command,
then processed by the actual upstream `prepare-builtins` utility. It is named
`ockl-workitem-only.bc`, never installed or advertised as complete `ockl.bc`.
Unchanged `oclc/src/abi_version_500.cl`, `isa_version_1013.cl` and
`isa_version_1030.cl` produce the authentic ABI and target constants. The fixture
selects exactly three inputs through LLVM 18's driver:

```text
--rocm-device-lib-path=<ignored-workitem-slice-directory>
--hip-device-lib=ockl-workitem-only.bc
--hip-device-lib=oclc_abi_version_500.bc
--hip-device-lib=oclc_isa_version_1013.bc  # gfx1013 only
```

The gfx1030 invocation selects `oclc_isa_version_1030.bc` instead. Driver traces
record the real `-mlink-builtin-bitcode` inputs. No gfx1030 library is silently
substituted for gfx1013. The initial absolute-path attempt without a search
path failed: LLVM 18 `HIPAMD.cpp:359-379` resolves explicit device-library names
through its configured library search paths. That diagnostic is retained.

PS5 compilation reveals another concrete ABI issue: its auxiliary host triple
makes device IR default to `wchar_size=2`, whereas these Linux-built upstream
bitcode inputs have `wchar_size=4`. The original link fails with conflicting
module flags. The bounded successful route explicitly uses
`-Xarch_device -fno-short-wchar`, keeping the PS5 host target and host ABI
unchanged. This chooses a four-byte device `wchar_t`; host/device structures or
kernel arguments containing `wchar_t` would require their own layout contract.
It is not a general cross-platform C++ ABI compatibility claim.

### Emitted results and the concrete launch extension

After the explicit driver choices above, **12 combined-object compilations pass**:
Linux host and PS5 host, each targeting gfx1013 and gfx1030, for the full-header
store kernel, ordinary indexed vector addition, and a `blockDim`/`gridDim`
query kernel. These contain actual linked device ELFs. All retain the same seven
host registration/launch undefined symbols and fat-binary wrapper described
above; Linux uses `.init_array`, PS5 uses `.ctors`. This is compiler and object
emission evidence only. The complete OCKL build still fails, and no native HIP
executable or HIP runtime has been linked or run.

The actual vector descriptor on both targets and hosts has properties `0x409`,
RSRC2 `0x8c`, **six user SGPRs**, wave32, zero private/LDS storage and no preload.
The kernarg pointer is still `s[4:5]`; the workgroup ID is `s6`, and the local
thread ID is `v0`. Neither a dispatch-packet pointer nor a queue pointer is
requested by this fixture. The COv5 implicit-argument pointer is lowered into
the existing kernarg allocation, rather than requiring a new pointer SGPR.

The vector's allocation is nevertheless **288 bytes, aligned to 8**:

| Metadata entries | Offsets from kernarg base | Sizes |
| --- | --- | --- |
| Explicit `a`, `b`, `out`, `n` | 0, 8, 16, 24 | 8, 8, 8, 4 |
| `hidden_block_count_{x,y,z}` | 32, 36, 40 | 4 each |
| `hidden_group_size_{x,y,z}` | 44, 46, 48 | 2 each |
| `hidden_remainder_{x,y,z}` | 50, 52, 54 | 2 each |
| `hidden_global_offset_{x,y,z}` | 72, 80, 88 | 8 each |
| `hidden_grid_dims` | 96 | 2 |

The implicit area begins at offset 32 and reserves 256 bytes. The emitted vector
ISA really loads `hidden_group_size_x` at **kernarg + 44**; it multiplies that
value by `s6` and adds `v0`. These are not merely unused metadata declarations.
The separate dimensions kernel has an eight-byte explicit pointer and a
264-byte total allocation; its ISA reads group size at +20 and block count at
+8. Under the compiler's recorded uniform-workgroup assumption, `gridDim.x`
reduces to the hidden block count. Other listed fields are not read by these
optimized fixtures, but their metadata cannot be accepted as arbitrary user
arguments or ignored silently.

This differs from `kernels/canary.hip:8-20`, which hard-codes block width 64 and
therefore avoids the implicit workgroup-size record. Reusing the six-SGPR
descriptor contract without initializing the hidden fields would leave block
size zero and produce wrong indexing.

### Implemented offline geometry packing

`Kernel.arguments` now contains only explicit arguments, while
`Kernel.hidden_arguments` retains the hidden metadata. The inspector displays
their separate counts and every hidden field. Admission requires all 13 fields
above, with their exact ABI offsets and widths, in an eight-byte-aligned
256-byte implicit area. Unknown kinds, missing/duplicate fields, overlap,
wrong storage size or alignment are rejected. The caller cannot supply hidden
values as additional scalar/pointer arguments.

`prepare_initial_launch` checks the existing one-dimensional 64x1x1 shape and
nonzero group count (at most `UINT32_MAX / 64`) before generating block counts
`(groups_x,1,1)`, group sizes `(64,1,1)`, zero remainders/global offsets and grid
dimensions `1`. The complete explicit, alignment-padding and implicit storage
is zero-initialized before little-endian packing, including two-byte fields.

The size rule follows LLVM 18.1.8's
[`AMDGPUSubtarget::getKernArgSegmentSize`](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp):
explicit storage follows argument alignment, implicit storage starts on its
required boundary, and the total rounds to four bytes. Our supported explicit
4/8-byte arguments already end on that boundary, so explicit-only profiles
reject any unaccounted trailing storage. This limits admitted layouts; it does
not verify that arbitrary instructions match metadata. Existing arithmetic,
resource, wave and full-header store fixtures retain their sizes and behavior.

The normal sanitizer CTest suite includes `hidden_geometry_model`, using
synthetic vector/dimensions objects with the same byte oracle and admission
checks; all 25 registered tests pass without needing hidden-geometry ELF files.
Additional host sanitizer checks consume eight real linked device ELFs through the
`hidden_geometry_test <fixture.hsaco>` command: vector/dimensions kernels for
both host triples and GPU targets. They compare every kernarg byte, exercise
count/shape limits, pointer/scalar validation and malformed layouts. The PS5
libraries also cross-build. Evidence is
[`offline-hidden-geometry.json`](../evidence/offline-hidden-geometry.json).
No native title or HIP runtime uses these fixtures yet; this closes a bounded
host-packing gap, with no GPU execution or new native capability claim.

The upstream source basis is `ockl/src/workitem.cl`: ABI >= 500 reads the
implicit argument record; its older-ABI branch reads a dispatch packet.
This result applies to the tested code-object-v5 uniform, one-dimensional
fixtures. Nonuniform groups, other builtins, device calls, allocation, atomics,
math libraries and different code-object/compiler versions need separate
contracts. Neither gfx1013 nor gfx1030 emission establishes the PS5 GPU family
or permission to enable ggml's RDNA2 SDOT4 path.

Evidence: [`offline-compatible-hip-headers.json`](../evidence/offline-compatible-hip-headers.json).
Exact sources, commands, failed attempts, driver traces, bitcode identities,
metadata, disassembly and object hashes are under ignored
`results/hip-host-abi-spike/rocm-6.2.4/`. No console, dependency-lock change,
commit or push was performed for this experiment.

## Matching TheRock compiler and HIP 7 headers

The completed compiler component at LLVM revision
`6da731510c233517c1eeb691d3446d80e9e32525` was tested with the unchanged
HIP/CLR headers from `rocm-systems` revision
`2f2a92736dbe3f65108539b6659d2c7289d300e8`, including the authentic generated
**HIP 7.17.26405** version header described above. The existing `vector-add.hip`
fixture was copied byte-for-byte into ignored `results/therock-hip-abi-spike/`.
This is the smallest full-header fixture using ordinary launch syntax and
`blockIdx`, `blockDim` and `threadIdx`.

Both hosts use the completed component's real, complete `ockl.bc` and `ocml.bc`,
plus its ABI-500, gfx1013, wave32, finite-only-off and unsafe-math-off bitcode.
Driver traces identify all seven identical inputs. No workitem slice, header
patch, invented version macro, builtin replacement or runtime stub is used.
Explicit Clang runtime-wrapper inclusion and pinned header directories select
the real headers; `-nogpuinc` disables automatic header discovery. The selected
code object is V5, wave32, compiled at `-O2`.

| Host invocation | Combined compile result |
| --- | --- |
| Linux, matching Clang 24 | Pass |
| Unmodified SDK target `x86_64-sie-ps5` | Device compilation reaches bundling; bundle target format is rejected |
| Same SDK with `--target=x86_64-sie-ps5-unknown` | Pass |

The SDK wrapper is unchanged and selects this compiler through `LLVM_CONFIG`.
Compile-only `-nostdlib -nostartfiles` avoids forwarding host archives into the
device link. `-Xarch_device -fno-short-wchar` remains device-only, preserving
the existing PS5 host flags. The original failure and successful command arrays
are retained separately.

The pinned `clang/lib/Driver/OffloadBundler.cpp:1748-1753` requires five or six
hyphen-separated bundle-ID components, including offload kind and environment.
`clang-linker-wrapper` constructs the host ID from the normalized host triple;
the three-component PS5 triple lacks that environment. Explicit `unknown`
supplies the required field without selecting another host OS. This conclusion
is checked beyond spelling: separate host-only LLVM IR for the original and
expanded triples is identical after replacing **only** the triple line and
compiler-generated HIP CUID symbol suffixes. Every function, argument,
attribute, data layout, module flag and registration operation otherwise
matches. The complete diff and normalization hash are retained. This proves
the tested fixture's ABI properties, not every possible PS5 ABI feature.

The combined objects retain the seven undefined symbols listed earlier and the
24-byte version-1 `0x48495046` wrapper. Linux uses `.init_array`; PS5 uses
`.ctors`. The empty host bundle entries now reflect their actual host triples;
the device bundle ID is `hip-amdgcn-amd-amdhsa--gfx1013`. Each extracted device
ELF is 4,504 bytes, with zero private/group storage, six user SGPRs, 288-byte
kernarg storage and the same 13 hidden geometry kinds described above.

The metadata target is `amdgpu10.13-amd-amdhsa-unknown-gfx1013`, and the current
loader rejects both objects with `metadata target mismatch`. Descriptor RSRC1
is `0xe0af0000`, including the newer forward-progress setting discussed in
[the compiler comparison](THEROCK_BINARY_COMPARISON.md). Matching argument
layouts do not qualify those descriptor bits or instructions on the console.

Evidence: [`offline-therock-hip-host-abi.json`](../evidence/offline-therock-hip-host-abi.json).
It records the original failure, source references, host-IR comparison,
compiler/shared-library and bitcode hashes, complete header-inventory digest,
output identities and driver/diagnostic provenance. This is combined-object and
fatbinary emission only. There is no final native executable link, conversion,
registration runtime, native load or GPU execution, and supplying full device
libraries does not prove all their functions or framework compatibility.
