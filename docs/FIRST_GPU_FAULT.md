# First indexed-store fault: frozen-artifact analysis

The first indexed-store run failed with a CPU jump to address zero. Offline
inspection found a concrete import-provider mismatch for `sceAgcSuspendPoint`,
and the frozen call sequence makes that unresolved import the strongest
explanation of the fault. The runtime GOT value was not captured, so this is a
high-confidence diagnosis of the failing boundary, not proof of the exact
pointer value at the instant of failure. GPU execution remains unproven.

The one-run observation is recorded in
[`native-indexed-store-first.json`](../evidence/native-indexed-store-first.json).
The application reached `START`, CPU runtime checks and `SUBMIT`; that last log
is emitted before entering `submit_and_wait`. There was no completion marker,
output oracle result or normal application finish. No automatic retry followed.

## Artifact identity

The analyzed image is the preserved
`dist/PPSA99014/gfx1013-indexed_store-3d295773d2fb` candidate, source
`15bfc358f014b4231fcf9f84918b79b1b301008f`. Its package SHA256 is
`fb634e34a1a6420fbd257e3929a3fbd37b5be8f6121db1eb6ee43362015febbb`.

| Artifact | SHA256 |
| --- | --- |
| Staged `app/eboot.bin` | `30902a8a8170319dcd3607f2ebf3dcb371ba7d32c28f539cf461876ab1954788` |
| Retained native `eboot.elf` | `d9878bce81b1c106b56a27a5ab86b9e4f0b55e8876be4f610bfa50b55068cf18` |
| Retained symbol-bearing `llvm-pie.elf` | `a0c26648e3748a13900444ab06933a10011112cc678835f588c5e2fac25e4bce` |
| ELF extracted from staged FSELF | `3ce7f4fe4930ff43e685820f3e5aa8be5900d77a235602c1976ab81558acf0f0` |

FSELF integrity passes. Every extracted `PT_LOAD` byte equals the corresponding
retained native ELF segment; its executable segment also equals the original
LLVM PIE executable segment. This establishes that the symbol/disassembly
mapping describes the frozen signed image. Extraction wrote a new ignored
diagnostic file; no frozen artifact was rebuilt or overwritten.

Before any replacement build, all 38 files from the failed native build and
staged candidate were copied into
`results/first-gpu-fault-analysis/frozen-native-build/` and
`results/first-gpu-fault-analysis/frozen-staged-candidate/`. Source hashes before
and after copying equal the snapshots, and every staged manifest/build-receipt
artifact hash matches. The local `preservation.json` SHA256 is
`8cd96593be57ff395256daeedb009d349c41caa2c3e26fe32f538a996bbf9447`.

## CPU call chain

The kernel reports executable base `0x400000`, RIP zero and saved return
addresses `0x400c4e`, `0x40016a`, zero.

| Relative address | Frozen instruction or symbol |
| --- | --- |
| `0xc48` | `main` calls `SubmissionQueue::submit_and_wait` at `0xb230` |
| `0xc4e` | Return from that call, matching the first saved frame |
| `0x16a` | `_start` return from `main`, matching the next frame |
| `0xac090` | `AgcBackend` vtable; its address point is `0xac0a0` |
| `0xb651` | Queue calls virtual `kick()` through vtable offset `0x20` |
| `0xbf20` | `AgcBackend::kick()` consists of `jmpq *GOT[0xb33c0]` |
| `0xb754` | Queue's first virtual `now_ns()` call, after `kick()` |

Fault RAX is `0x4ac0a0`, exactly the relocated backend vtable address point.
RDI is the stack backend object, RBX has the queue address, R13/R15 are zero,
and R12 retains a heap-like pending-vector end. The kick tail jump preserves
these registers and introduces no frame. A null `sceAgcSuspendPoint` GOT slot
therefore explains both RIP zero and the absence of a backend frame.

The captured state does not match the initial cache-flush call under ordinary,
uncorrupted execution: that call requires a nonzero buffer in RSI/R15 and an
allocation byte count in RDX. Nor does it match a null `clock_gettime` import:
`now_ns()` first creates another frame and sets RDI to clock ID 4 and RSI to its
stack timespec. A null initial `now_ns()` vtable slot remains a formal
alternative because the raw return pointer at RSP and runtime vtable/GOT memory
were not captured. All eight backend `R_X86_64_RELATIVE` relocation records
survive conversion into the native ELF; a missing relocation record is not the
observed static defect.

The control-flow inference is consistent with reaching the kick boundary after
a zero submit return. It is not a captured driver result and must not be recorded
as proof of accepted commands, GPU execution or completed work.

## Confirmed provider mismatch

The frozen native import table requests:

| Symbol | NID | Frozen module / library | Pinned reference provider |
| --- | --- | --- | --- |
| `sceAgcInit` | `kW3GLb7QfPg` | `libSceAgc` / `libSceAgc` | `libSceAgc` |
| `sceAgcDriverSubmitDcb` | `UglJIZjGssM` | `libSceAgcDriver` / `libSceAgcDriver` | `libSceAgcDriver` |
| `sceAgcSuspendPoint` | `h9z6+0hEydk` | `libSceAgcDriver` / `libSceAgcDriver` | **`libSceAgc`** |

The exact locked PS5_Vulkan revision
`3f3ee69607013b345d2baa6d6a37c86745649a08` records this split in
[`probes/SHADER_PACKAGES.md:397-402`](https://github.com/mihawk-99/PS5_Vulkan/blob/3f3ee69607013b345d2baa6d6a37c86745649a08/probes/SHADER_PACKAGES.md#L397-L402)
and checks linked addresses before submission in
[`src/diagnostics.cpp:1733-1742`](https://github.com/mihawk-99/PS5_Vulkan/blob/3f3ee69607013b345d2baa6d6a37c86745649a08/src/diagnostics.cpp#L1733-L1742).
The frozen ps5-hip builder instead put both submission functions in its
`libSceAgcDriver` link stub. Previous offline checks asserted that same incorrect
binding; passing them did not establish the correct provider.

## Smallest corrective experiment

Correct the link stub so `sceAgcSuspendPoint` comes from `libSceAgc`, and add an
offline regression that inspects the converted native import table against the
independently pinned split. Preserve the failed image and its receipts.

The corrective candidate at `b8bc2b0` starts from the failed source revision and
changes only this import provider. Newer geometry/wave code stays outside that
isolated build. All nine application objects and the complete LLVM PIE are
byte-identical; the native ELF changes only 28 provider/hash/build-ID bytes.
FSELF and package checks pass. See
[`offline-agc-provider-fix.json`](../evidence/offline-agc-provider-fix.json).
Retain existing ownership/quarantine and
one-run rules. If the corrected provider does not resolve the fault, linked-address
checks and separate submit/kick checkpoints are the next diagnostic step; do not
infer a GPU failure from the same partial application log.

This correction addresses a CPU import boundary. It does not justify changing
GPU family selection, shader ISA, kernarg layout, cache packets or completion
rules based on this run.

The corrected image subsequently passed once on idle .30, firmware 6.02: all
73 indexed-store values matched, full guards were unchanged, allocations closed,
and the application completed its shell-managed lifecycle before host cleanup.
See [the hardware receipt](../evidence/native-provider-fixed-store.json).
This controlled outcome supports the provider diagnosis and proves this device
kernel executed; vector-add and repeated-dispatch/lifecycle coverage remain open.

Exact commands, disassembly, relocation tables, pinned source snapshots and the
mapping receipt are preserved locally under
`results/first-gpu-fault-analysis/frozen-mapping/`. Receipt SHA256:
`74b208c22ada983788bf82df893056fed8c0151619f85f14b2eab9b6559d7dda`.
No console access, rebuilding, packaging, commit or push was performed during
this analysis.
