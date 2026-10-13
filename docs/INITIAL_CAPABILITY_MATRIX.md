# Initial compute capability contract

Historical contract for the first arithmetic experiments. Later G2 and G3 results
are recorded in [the current qualification map](G3_QUALIFICATION.md) and `../STATE.md`.
The pending-gate language below describes that earlier experiment state.

Scope: corrected LLVM/LLD 18.1.8 arithmetic fixtures, AMDHSA V5, little-endian
ELF64, targets gfx1013 and gfx1030. Hardware evidence now covers the frozen
gfx1013 indexed-store once and vector-add twice in one later title cycle on
firmware 6.02. Both title cycles completed cleanly. G2 still requires one frozen
application running indexed-store followed by vector-add, with repeated clean
cycles of that combined image; separate profile images do not satisfy item 5's
literal requirement. gfx1030 is offline-only. These results qualify
the named arithmetic fixtures and launch shape, not arbitrary structurally
eligible objects, later hidden-geometry support or a general ISA/runtime.

| Requirement | Observed or implemented offline | Hardware status / rejection boundary |
| --- | --- | --- |
| CPU application lifecycle | CPU control, indexed-store and first two-dispatch vector title cycle passed on FW 6.02 | Ordinary C exit still faults; no threaded/framework qualification |
| Code object | Bounded loader, relative PT_LOAD image, kernel symbols/descriptors and metadata agreement | Reject relocations, unresolved symbols and unsupported layouts; no general binary loader |
| Entry address | Module base plus entry 0x1800 (store) or 0x1900 (add); 256-byte alignment | Both frozen entries executed within native shader window [0x200000000, 0x300000000) |
| Argument storage | 8-byte alignment; 12 bytes for store, 28 for add; explicit pointer/count offsets | No implicit dispatch/queue pointer, device calls or dynamic allocation |
| User SGPRs | Descriptor properties 0x409; six user slots: zero s0:s3, kernarg address s4:s5 | Private-buffer reservation is present but no scratch use is allowed |
| System registers | RSRC2 0x8c requests group X at s6 and local X at v0 | Exact 73-element results across two groups exercise X indexing; other layouts remain outside this hardware proof |
| Wave and launch shape | Wave32; harness sets 64x1x1 threads and 2x1x1 groups | No wave64 or multidimensional acceptance; tail test uses 73 elements |
| Program resource registers | Both kernels/targets emit RSRC1 0x60af0040, RSRC2 0x8c, RSRC3 0; preload 0 | Descriptor bits are copied into compute registers; privilege/debug, LDS, scratch and unsupported RSRC3/preload requirements rejected |
| FP execution mode | Compiler RSRC1 is preserved and compilation disables fast math | Two vector dispatches produced exact binary32 quarter-increment sums; no general IEEE claim |
| State initialization | Destination masks, accumulator zeros, program high bits, resource limits and thread dimensions explicitly emitted | Frozen store/vector fixtures passed with explicit state; other shader/resource configurations unqualified |
| Visibility and completion | CPU flushes; ACQUIRE_MEM before dispatch; compute drain then RELEASE_MEM marker | Exact host readback and ownership release passed for these serialized fixtures; no general coherence or GPU producer/consumer claim |
| Output | Exact indexed XOR oracle, full allocation guards and read-only input checks | A completion marker alone is insufficient; all expected output must match |
| Resource lifetime | Checked command/marker and harness allocation closure after completion | Any uncertain submission parks and retains allocations; no cancellation or retry claimed |
| Shutdown | Explicit finite-job shell barrier after successful main; strict finalization/request log | Must self-stop before host cleanup, release sandbox/image and leave services healthy |
| General compute | Resource probes compile for reductions, LDS, atomics and spilling | G3 work only; rejected by the initial launch profile, not advertised as supported |

## Instructions admitted to the first experiment

The frozen corrected indexed-store object has SHA-256
`d3962dfa317c6d494b2e81e2c987b4dd3ba238708d61130e5111c15f9a09b49e`.
Its inspected store entry uses scalar loads/waits and conditional execution,
integer index/pointer arithmetic and a global store:

`s_load_dword`, `s_load_dwordx2`, `s_waitcnt`, `s_and_saveexec_b32`,
`s_cbranch_execz`, `s_endpgm`, `v_lshl_add_u32`, `v_cmp_gt_u32_e32`,
`v_mov_b32_e32`, `v_lshlrev_b64`, `v_xor_b32_e32`, `v_add_co_u32`,
`v_add_co_ci_u32_e32`, `global_store_dword`. Disassembly also includes `s_nop`
alignment padding. The object contains a separate vector-add entry, which the
first store experiment does not dispatch.

This list describes the inspected frozen entry, not a general ISA whitelist
implemented by the loader. Structural eligibility does not inspect arbitrary
instructions. A changed object needs fresh disassembly and evidence before a
console experiment. Unknown instructions, scratch setup, LDS use, device calls,
matrix instructions and privileged/debug state are outside this first experiment.

The authoritative package/controller hashes are in
`evidence/native-provider-fixed-store.json`, `evidence/native-vector-twice-cycle1.json`
and the package catalog. The failed first image remains revoked. Runtime
acceptance and timeouts remain those in `FIRST_NATIVE_EXPERIMENT.md`.
