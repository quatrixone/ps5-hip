# Genuine tiled matmul compiler measurement

This unit measures two ordinary full-header HIP clients against pinned LLVM
18.1.8, HIP 6.2.4 headers, and the existing work-item/ABI500/ISA1013 device
library slice. It does not admit a runtime route or execute a GPU workload.

`tests/hip_tiled_matmul.hip` uses fixed tile width 8 in its row/column
calculation. `tests/hip_tiled_matmul_blockdim.hip` instead uses the natural
`blockIdx.y*blockDim.y+threadIdx.y` and X equivalent. Both declare two
8x8 static shared float arrays, launch 8x8x1 threads, and never read a dummy
Z coordinate. They use genuine compiler-generated registration, an ordinary
`<<<>>>` call, and compiler-generated unregister.

Run from this worktree with the main checkout as `--reference-root`:

```
python3 -B tests/geometry_lds_recipe_test.py
python3 -B tools/probe_geometry_lds.py --reference-root <main> --output results/compiler-fixed-v2
python3 -B tools/probe_geometry_lds.py --reference-root <main> --output results/compiler-blockdim-v1 --variant blockdim
```

The recipe requires a fresh output. Both completed receipts bind four actual
Linux/PS5 O1/O2 combined objects, selected gfx1013 device objects, metadata,
ISA, host lifecycle relocations, caller IR/CFG, commands, tools, and inputs.
The original `results/compiler-v1` is a preserved failure: the PS5 diagnostic
IR command removed `-c`, causing the SDK wrapper to append a CRT input.
Retaining `-c` fixes that diagnostic invocation; it does not change the combined
object compilation. Original source copies are retained next to that failure.

## Actual profile

| Property | Fixed tile calculation | `blockDim` calculation |
| --- | --- | --- |
| Static LDS / private bytes | 512 / 0 | 512 / 0 |
| Kernarg bytes / alignment | 36 / 8 | 296 / 8 |
| Explicit arguments | global pointers at 0,8,16; u32 at 24,28,32 | Same |
| Hidden arguments | None | Exact 13-field COv5 block starting at 40 |
| RSRC2 | `0x98c` | `0x98c` |
| Properties / RSRC3 / preload | `0x409` / 0 / 0 | Same |
| Wave / max threads / spills | 32 / 64 / zero SGPR and VGPR spills | Same |
| RSRC1 O1 / O2 | `0x60af0082` / `0x60af0083` | Same |

The hidden block has counts at 40/44/48, group sizes at 52/54/56,
remainders at 58/60/62, global offsets at 80/88/96, and grid dimensions
at 104, with the existing COv5 field widths. The final 256-byte implicit
area begins at aligned explicit size 40. There is no private scratch or
dynamic stack. The descriptor requests natural X/Y workgroup IDs and
X/Y work-item IDs: **`0x98c`, not `0x138c`**.

Linux and PS5 whole ELF hashes differ. For each variant and optimization,
their `.text`, `.rodata` descriptors, and `.note` metadata are byte-identical;
`results/target-comparison.json` records those exact comparisons. Whole-object
identity is not claimed. Both actual ISAs contain two barrier sites, LDS
reads/writes, guarded global loads/stores, and separate FP32 multiply/add.

## Barrier and oracle evidence

The K tile count derives solely from the uniform scalar K argument. Every
thread writes either a guarded valid value or zero to each shared tile before
the first barrier. Divergent load paths merge before that barrier. The second
barrier precedes reuse of shared storage and the uniform K-loop decision.
Only the final output store has a row/column predicate; edge threads do not
return early from a tile iteration.

In the blockDim O2 caller CFG the two load diamonds merge into block 83,
which contains both barriers and branches either to the next tile (62) or
the final store predicate (57). The O1 caller CFG retains an inner loop:
barriers are in blocks 73 and 75; the latter branches on the uniform tile
counter to 52 or the final-store path 47. Final linked ISA is retained
separately. This is manual source/CFG/ISA evidence, not an automated barrier
uniformity proof; caller IR omits device-library linkage.

The independent integer-valued FP32 oracle covers M/N/K = 8/8/8,
13/21/19, 1/1/17, and 17/9/1, each twice with both inputs changed.
The main edge case has six workgroups, 273 logical outputs, 111 inactive
threads, and three K tiles. Products and running sums remain exactly
representable in FP32. The client checks all output bits, entire unchanged
A/B allocations, and 16-word prefix/suffix poison guards. These fixtures do
not qualify arbitrary FP32 distributions or fast-math behavior.

## Prospective route boundary

The existing separate LDS and geometry routes reject these combined objects.
A new explicit geometry+LDS route should reuse checked capture, the 13-field
hidden packer for the blockDim variant, actual X/Y entry admission, and the
LDS planner. With the current 1KiB allocation rounding, static 512/dynamic 0
requires 1024 allocated LDS bytes and programmed RSRC2 **`0x1098c`**.
Keep old routes unchanged. Initially admit only measured wave32, private0,
spill0, static512/dynamic0, 8x8x1/2D profiles and exact metadata/descriptor
consistency. Reject malformed hidden fields, unsupported axes, bad shapes,
overflow, insufficient pointer extents, ambiguous backing ownership, dynamic
LDS, and scratch combinations before backend initialization.

This measurement is sufficient to design that bounded route. It is not a
native link, runtime callback model, package, hardware result, or broad G3
completion. Installed SDK/compiler support trees are not fully inventoried;
the receipts make no hermetic-build claim.
