# TheRock compiler comparison

Status: **build-tree compiler binaries only; full component completion and staged
installation remain pending.** The four frozen arithmetic/resource fixtures
compile with AMD Clang/LLD 24 from pinned LLVM revision `6da731510c233517c1eeb691d3446d80e9e32525`.
All four fail the current native loader's metadata target-string check. Retain
the selected LLVM 18.1.8 route; no compiler recipe, loader or frozen title changed.

`evidence/offline-therock-binary-comparison.json` records source/recipe/driver/tool
identities, object hashes, metadata, descriptor fields and opcode inventories.
It also records the resolved shared-library hashes, including libclang-cpp,
libLLVM and TheRock's zlib/zstd: executable hashes alone do not identify this
compiler. Library hashes were observed during analysis, not captured at the
original compile instant. Raw outputs and analysis scripts are in
`results/therock-buildtree-comparison`. Tool output readability is not native
loader acceptance or hardware compatibility.

## Target identity: new spelling, matching processor

| Target | LLVM 18 metadata target | TheRock metadata target | ELF flags, both |
| --- | --- | --- | --- |
| gfx1013 | amdgcn-amd-amdhsa--gfx1013 | amdgpu10.13-amd-amdhsa-unknown-gfx1013 | 0x142 |
| gfx1030 | amdgcn-amd-amdhsa--gfx1030 | amdgpu10.30-amd-amdhsa-unknown-gfx1030 | 0x36 |

Both routes emit AMDHSA V5 (ELF ABI version 3), metadata version 1.2 and the
requested ELF processor. gfx1013 retains XNACK ANY; gfx1030 retains the
unsupported-XNACK encoding. The new strings are documented by the pinned
[AMDGPU architecture tables](https://github.com/ROCm/llvm-project/blob/6da731510c233517c1eeb691d3446d80e9e32525/llvm/docs/AMDGPUUsage.rst#L50):
`amdgcn` is a legacy alias of `amdgpu`, and the subarchitecture table maps
`amdgpu10.13` to gfx1013 and `amdgpu10.30` to gfx1030. The metadata emitter
serializes `TargetID.toString()`, which appends the processor to the target
triple. `unknown` here is the triple environment spelling, not an unknown GPU.

The current loader accepts only the old exact string. Its rejection does not
show that the compiler targeted the wrong processor, but later loader checks
have not run on these objects. A future compatibility change must explicitly
validate architecture/processor correspondence, vendor/OS/environment and
features against ELF flags, with mismatch rejection tests. Arbitrary prefix
stripping or suffix matching would weaken validation and is not proposed here.

## Initial canaries

All kernel argument layouts match their LLVM 18 counterparts. Indexed store has
12 argument bytes and vector add has 28; alignment remains eight, wave size 32,
and private/LDS requirements zero. Descriptor properties remain `0x409`, RSRC2
`0x8c`, RSRC3/preload zero. Entries move from `0x1800/0x1900` to `0x1900/0x1a00`;
the loader must continue using actual entry offsets. Object sizes increase from
4,096 to 4,784 bytes, and all compared object hashes differ.

Canary RSRC1 changes from `0x60af0040` to `0xe0af0000`. The pinned
[descriptor definition](https://github.com/ROCm/llvm-project/blob/6da731510c233517c1eeb691d3446d80e9e32525/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h#L107)
identifies the changes as GFX10 `FWD_PROGRESS` bit 31 becoming one and the
encoded SGPR-count field at bits 6-9 becoming zero. Metadata register counts
remain unchanged. This is a changed native register configuration requiring
review and hardware qualification; it is not automatically accepted because
the canary arithmetic looks familiar.

Disassembling **both** gfx1013 canary objects with the same LLVM 24 tool produces
identical instruction/operand text after excluding padding, despite changed
machine words. The changed second dwords are:

| Kernel/instruction | LLVM 18 | TheRock |
| --- | --- | --- |
| store: v_lshlrev_b64 | 0x00020082 | 0x02020082 |
| store: v_add_co_u32 | 0x00020200 | 0x02020200 |
| add: v_lshlrev_b64 | 0x00020082 | 0x02020082 |
| add: three v_add_co_u32 instructions | 0x00020000 / 0x00020002 / 0x00020004 | 0x02020000 / 0x02020002 / 0x02020004 |

Each difference is second-dword bit 25, within the unused VOP3 third-source
field. The pinned
[postEncodeVOP3 implementation](https://github.com/ROCm/llvm-project/blob/6da731510c233517c1eeb691d3446d80e9e32525/llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUMCCodeEmitter.cpp#L763)
sets unused GFX10+ source fields to inline immediate zero (`0x80`), avoiding
conservative SGPR-read accounting. This explains the bytes without attributing
the difference to disassembler versions; it does not qualify PS5 execution.

gfx1030 also changes carry-add selection from an e32 instruction writing VCC
to `v_add_co_ci_u32_e64` writing `null`, with corresponding branch-distance
changes. All changed code requires its own instruction review and hardware
qualification before replacing the frozen LLVM 18 candidate.

## Resource probes and private lowering

The reduction, LDS and atomic probes retain their argument layouts, wave32 and
private/LDS sizes. RSRC1 has the new forward-progress setting throughout.
Scheduling and encodings differ; matching metadata alone is insufficient.

| Probe | Concrete TheRock result on both targets | Implication |
| --- | --- | --- |
| private_array | 144 private bytes; 32 flat stores + one flat load; explicit HW_REG_FLAT_SCR setup remains | The volatile generic-pointer issue is not fixed |
| private_explicit | 144 private bytes; 32 buffer stores + one buffer load; no hardware flat-scratch setup | Buffer route remains a candidate for G3 scratch work |
| register_spill | 212 private bytes; 52 VGPR spills, 52 buffer stores/loads; no flat-scratch setup | Deliberate buffer spilling remains represented |

The generic private-array code is substantially smaller in register usage:
gfx1013 SGPR/VGPR counts fall from 42/53 to 14/9; gfx1030 falls from 38/60 to
14/9. This is compiler output evidence, not a speed measurement or removal of
the flat-scratch dependency.

Unlike LLVM 18, TheRock's explicit-private kernel drops the unused flat-init
SGPR pair: properties `0x429` becomes `0x409`, RSRC2 `0x91` becomes `0x8d`, and
user SGPR count falls from eight to six. Group X moves from s8 to s6 and the
private wave offset from s9 to s7. Kernarg remains s4:s5; the private-buffer
descriptor remains s0:s3. Private bytes remain 144. This is a compiler-produced
ABI change, not a descriptor modification by this project.

The spill fixture's 64 flat global-input loads and 64 flat global-output stores
remain distinct from its buffer-based scratch traffic. Neither fixture supplies
the missing native scratch allocation, wave offsets, descriptor, ring state or
hardware validation described in `VOLATILE_PRIVATE_LOWERING.md`.

## Next decision

Keep LLVM 18 for initial execution. Future compatibility work includes explicit
new target-ID validation, changed register settings/encodings, and independent
G3 private-memory implementation. The full `amd-llvm` component build and staged
installation remain unverified; this comparison qualifies neither that installation
nor PS5 execution.

## Identity-verified replay

`evidence/offline-therock-verified-comparison.json` supplements the original
preliminary receipt without replacing it. All four compiler runs were bracketed
by executable, resolved shared-library, source/recipe and host-inspector hashes.
Those identities were stable and matched the first comparison's compiler identities.
The current loader again rejected all four objects for the documented target spelling.

The fresh output directory changed `__hip_cuid_*` and its symbol/string/hash tables;
whole-object equality is **false**. All four `.text`, `.rodata`, `.note`, `.eh_frame`
and `.AMDGPU.gpr_maximums` sections are byte-identical to the first comparison.
The pinned driver hashes non-input arguments, including `-o` and `-###`, into its
compilation ID; AST/code generation hashes that ID again for the exported symbol.
Consequently a separate driver trace is not the original compilation's ID.
The receipt records changed sections and corresponding ELF size/offset effects.

A diagnostic passing the exported symbol suffix as `-cuid` did not reproduce the
original bytes: it hashes that suffix again. A source-derived driver job-printer
also failed to recover the original ID and executed no compiler jobs after its
failed precheck. These attempts are retained as failed diagnostics, not normalized
into a success claim. GNU/SysV hash-table contents were not individually normalized.

G1 explicitly permits explained metadata differences. The comparison supports
selecting the independently reproducible LLVM 18 route; it does not require the
newer compiler to produce identical binaries. Full Flang/component completion and
staged installation are distinct from this completed comparison of the relevant tools.
