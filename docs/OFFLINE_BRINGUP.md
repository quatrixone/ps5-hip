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

Ordinary generic-pointer private-memory support remains unresolved. The explicit
address-space cast is a diagnostic probe, not a requirement we intend to impose
on normal HIP source. Buffer-scratch allocation, descriptor setup, per-wave offsets
and actual spill correctness also remain unimplemented/unverified.

Still outstanding offline: generic private-address lowering, expanded semantic
negative fixtures, dependency/license lock, native launch-state construction,
native cross-build and the first bounded hardware experiment package. G1 remains
open. Scratch, LDS, atomics, synchronization and spill behavior require dedicated
G3 coverage; the arithmetic canaries cannot qualify them.
