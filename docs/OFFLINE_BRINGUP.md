# Offline compiler and object loading

Status: G1 work in progress. Nothing in this document establishes GPU execution.
No console connection is needed for these checks.

## Reproduction

On the WSL host, install/use Clang, LLD and LLVM tools version 18, CMake, Ninja
and Python 3. The observed compiler is Ubuntu Clang 18.1.8 (20ubuntu8).

From the repository root:

```sh
python3 tools/build_canaries.py
python3 tools/build_canaries.py --fixture resources
cmake -S . -B build/host -G Ninja -DCMAKE_CXX_COMPILER=clang++-18 -DPS5HIP_SANITIZE=ON
cmake --build build/host
ctest --test-dir build/host --output-on-failure
build/host/ps5hip-inspect build/canaries/gfx1013/canary.hsaco
```

The compiler recipe deliberately uses HIP device language without HIP headers or
device libraries. This establishes compilation of these particular kernels;
normal HIP source registration, host APIs and framework compatibility remain G4+
work. `--no-gpu-bundle-output` produces a linked ELF rather than a bundled device
container. The recipe explicitly selects LLD 18, code-object V5 and wave32.
`build/canaries/build.json` records compiler commands, versions and artifact hashes.
LLVM readobj and disassembly outputs remain beside each artifact for inspection.
The recipe rejects a compiler, linker or inspector version that differs from the
exact LLVM version in `dependencies.json`, before creating output. Same-host
rebuilds in a separate directory reproduced all four arithmetic/resource code
objects byte for byte; see `evidence/offline-compiler-reproduction.json`. This
does not replace the planned clean-machine reproduction requirement.

## Observed contract

Both gfx1013 and gfx1030 outputs contain `indexed_store` and `vector_add`. Each
fixture computes its global index using an explicitly fixed 64-thread workgroup.
Launching with another workgroup size would give incorrect indexing.

The indexed-store kernarg has an eight-byte output pointer followed by a four-byte
count (12 bytes). Vector addition has three eight-byte pointers followed by a
four-byte count (28 bytes). Both kernels use wave32 and report zero private and
group storage. Their descriptors reserve four private-buffer user SGPRs followed
by the two kernarg SGPRs: the kernarg address belongs in s4:s5, with group ID X
in s6. Zero scratch usage does not remove those reserved slots. This is a compiler
observation, not confirmation of the native PS5 launch ABI.

## Loader boundary

`ps5hip_object` structurally validates linked ELF64 AMDHSA V5 files for the two
candidate targets. It bounds ELF ranges and metadata complexity; checks load
permissions, section/symbol mappings, descriptor fields and metadata agreement;
and rejects relocations, unresolved symbols, unsupported dynamic linking, TLS,
XNACK-ON and unsupported metadata encodings. Materialization copies load segments
to their relative virtual addresses and zero-fills gaps and BSS.

`initial_launch_rejection` further restricts the first launch profile to wave32,
scratch-free and LDS-free kernels with the inspected argument and SGPR layout.
Structural acceptance is not an instruction validator. The code object still
requires ISA review and a qualified native submission path. The parser is a
deliberate initial subset, not a general AMDHSA implementation.

## Validation performed

Both real compiler fixtures passed with AddressSanitizer and UndefinedBehaviorSanitizer.
Tests cover metadata inventory, argument sizes, profile rejection, exact image
materialization including zero-fill, every truncated prefix, targeted malformed
headers/descriptors, and 5,000 deterministic byte-mutation inputs per target.
Mutations may legitimately remain structurally valid; the tests check safe parsing,
not blanket rejection. This finite corpus does not prove robustness for all inputs.

## Resource probes and unresolved scratch lowering

`kernels/resources.hip` adds a serial 16-element integer reduction, a 64-thread
LDS reduction, a relaxed global atomic increment, two private-address probes and
a register-pressure kernel. Both target builds preserve all six kernels. The LDS reduction reports 256 bytes
of group storage and emits `ds_*` operations and `s_barrier`. The private array
reports 144 bytes of private storage and emits `HW_REG_FLAT_SCR_LO/HI` setup.
The private array reports zero register spills: it is not a substitute for the
deliberate register-spill test required by G3. `private_explicit` performs the same
array operation through a pointer explicitly in address space 5. It emits buffer
loads/stores without flat-scratch setup on both targets, with 144 private bytes.
`register_spill` caps allocation to 16 VGPRs and orders 64 volatile loads before
the corresponding stores. Both target outputs report **52 VGPR spills and 212
private bytes**, with buffer instructions for spill storage and no flat-scratch
setup. Generic global accesses in that kernel still use `flat_load/store`;
this is a separate native execution requirement.

The resource tests verify descriptor requirements and initial-profile rejection
for LDS/private storage. Atomic metadata can pass the initial resource filter;
this does not qualify its `global_atomic_add` instruction for native execution.

These useful buffer paths are present with the **unmodified compiler recipe**.
An experiment through LLVM bitcode with `-enable-flat-scratch` disabled did not
fix generic private accesses, and its extra build route was removed as unnecessary.
Disabling `flat-scratch-insts` additionally produced an empty kernel inventory
despite tool exit success; that configuration was discarded. The build checks
both metadata and disassembly kernel inventories and verifies buffer load/store
presence and absence of flat-scratch setup in the explicit-private and spill
fixtures. It records per-kernel opcodes and displays compiler diagnostics.
This text-based audit is a regression check, not a complete binary verifier.

The generic volatile private-memory path remains unresolved. An isolated contrast
experiment removed only `volatile` from the same array declaration: both targets
then retained 144 private bytes and emitted buffer loads/stores without flat-scratch
setup. Optimized IR contains 32 stores and one load in private address space 5;
the volatile variant retains those accesses through generic pointers. Thus this
ordinary fixed-size array does not require an explicit source address-space cast.
LLVM 18's `InferAddressSpaces` pass only rewrites volatile accesses when target TTI
permits the volatile variant; see the pinned source reference and reproducible
source/arguments in `evidence/offline-private-inference.json`.

Removing `volatile` is a contrast experiment, not a valid fix for a program that
requires volatile semantics. Arbitrary generic private pointers are not qualified.
The explicit address-space cast remains a diagnostic, not a public source
requirement. Buffer-scratch allocation, descriptor setup, per-wave offsets and
actual spill correctness also remain unimplemented/unverified.

Native launch-state construction, cross-build and packaging now exist; see
`STATE.md` for their separate CPU lifecycle results and limitations. Remaining
offline qualification includes clean-machine reproduction, final dependency/license
review and the explicit toolchain-route comparison. Generic private-address
lowering remains a G3 issue. G1 remains open; these updates do not certify native
execution. Scratch, LDS, atomics, synchronization and spill behavior require dedicated
G3 coverage; the arithmetic canaries cannot qualify them.

## Isolated compiler reproduction and linker correction

The first fresh-userspace run found that Clang 18's HIP device driver ignored
`-fuse-ld` and selected `/usr/bin/lld` (21.1.8 on the development host). The earlier
version check tested an executable the driver did not use. Earlier same-host
receipts remain historical evidence, but do not establish an all-LLVM-18 build.
The corrected recipe uses `-B` with the resolved pinned linker's directory and
rejects a `-###` device-link command selecting any other executable. The driver
trace is retained beside every generated object. Regression checks reject absent,
wrong and duplicated device-link commands.

A fresh Ubuntu package root reproduced all four corrected device objects exactly.
The run shares the WSL kernel and CPU, but its compiler, libraries and Python came
from authenticated Ubuntu packages. Only four selected project source files enter
the build directory; bubblewrap hides the development filesystem and disables
networking. No physical second-machine result is claimed.

To prepare a disposable root (mmdebstrap may itself be extracted from its package):

```sh
mmdebstrap --mode=chrootless --variant=extract --components=main,universe \
  --include=clang-18=1:18.1.8-20ubuntu8,lld-18=1:18.1.8-20ubuntu8,llvm-18=1:18.1.8-20ubuntu8,python3,coreutils,bash \
  --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg \
  resolute /tmp/ps5hip-fresh-root http://archive.ubuntu.com/ubuntu
mkdir /tmp/ps5hip-fresh-root/work
python3 tools/reproduce_isolated_canaries.py \
  --rootfs /tmp/ps5hip-fresh-root --output /tmp/ps5hip-fresh-build
```

The script rejects an existing output and checks all four complete object hashes;
it does not normalize away linker differences. Evidence is in
`evidence/offline-isolated-compiler.json`. Rebuild native candidates embedding old
objects; preserve their frozen images as superseded history. TheRock's built
compiler comparison remains outstanding.

## Wave32 probe preparation for G3

`kernels/wave.hip` separately exercises first-active-lane broadcast, XOR-one
shuffle, odd-value ballot and integer butterfly reduction. `wave_full` uses all
32 lanes; `wave_partial` activates only lanes 4 through 19, including a case where
lane zero is inactive. Every shuffle source is an active lane. These probes do
not change the arithmetic/resource fixtures or the frozen first-launch image.

Run the compile-only checks with a fresh output directory:

```sh
python3 tools/probe_wave_compiler.py --output results/wave-compiler-qualification-new
```

The tool checks the pinned compiler and actual device linker, compiles both
targets at O1 and O2, and retains metadata/disassembly plus independent scalar
expected-output files. All four profiles preserve `v_readfirstlane_b32` and
`ds_bpermute_b32`, wave32, and zero private/LDS bytes. The partial-wave branch uses
`s_and_saveexec_b32` on gfx1013 and `v_cmpx_gt_u32_e32` on gfx1030. Ballot values
come from the active-lane comparison mask; they need no standalone ballot opcode.
See `evidence/offline-wave-probes.json` for source/recipe hashes and inventories.

The future native test must launch exactly one 64-thread workgroup and initialize
256 output words to `0xdeadbeef`. Each invocation writes four words: broadcast,
shuffle, ballot, sum. The two full waves expect sums 528 and 1552 and ballot
`0x55555555`; the two partial waves expect sums 200 and 712 and ballot `0x55550`.
Inactive invocation records must stay unchanged. Use guarded allocations and the
same completion/teardown requirements as other native cases.

There is no native harness or execution result for these probes yet. Compilation
and scalar oracle self-checks do not prove PS5 wave semantics, wave64 support or
all inactive-lane patterns. They prepare required G3 cases without advancing G3.
