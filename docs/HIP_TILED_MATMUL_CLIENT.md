# Ordinary tiled matrix client

The genuine HIP client uses the measured `tiled_matmul` shader unchanged, selected
by the reusable metadata-driven geometry/LDS route. One compiler constructor
registers one kernel; eight ordinary blocking launches cover shapes (M,N,K)
8/8/8, 13/21/19, 1/1/17 and 17/9/1, with two changed-input rounds each. The client
checks actual runtime observations after every completed launch. Diagnostic
counts do not control runtime admission.

Inputs are small signed integer-valued FP32 numbers. An independent host integer
dot product supplies the exact binary32 output oracle: at most 19 products of
magnitude six, so every product and partial sum is exactly representable. There
are 982 checked active results across eight launches. Inactive threads have no
separate output slots; surrounding 16-float bit guards detect writes outside the
logical result. Both complete input allocations, including their guards, are
compared byte-for-byte after every launch. Device allocations persist across the
two rounds of each shape, then all three are explicitly freed.

Application failures attempt each remaining free and report failure independently
of safe runtime finalization. The model exercises rejected shape/dynamic LDS,
allocation/map failures, initialization, pre-submit failure, missing/corrupt
results, either input mutation, prefix/suffix corruption, late registration
failure and unresolved submission errors. Failed completion or cleanup retains
the established terminal ownership rules. Genuine compiler unregister and State
destruction are observed; no handwritten destructor substitutes for them.

`probe_hip_matmul.py --reference-root PATH --output FRESH_DIRECTORY` builds five
real combined objects (Linux/PS5 O1/O2 and Linux ASan model O2), extracts the exact
bundle, checks descriptor/ISA and constructor evidence, and runs the linked real
client against synthetic transport callbacks. Those callbacks validate packets,
kernargs and ownership; they do not execute or emulate AMDGPU instructions. The
recipe rejects any change to the pinned measured shader body and preserves caller
IR separately from linked device ISA. Run `tests/hip_matmul_recipe_test.py --check
OUTPUT` to verify the saved source/input/artifact bindings and transcripts.

This step does not package or run a title, prove arbitrary shader indexing safety,
or qualify general floating-point distributions. The caller remains responsible
for indexing bounds. Existing single-module, single-thread, blocking runtime
limits remain. The sanitizer model uses deliberate mappings in the 8 to 12 GiB
shader window with `protect_shadow_gap=0`; those are not heap redzone allocations,
so explicit guards are necessary and this does not prove native allocator safety.
The installed SDK/resource/static support trees are not fully inventoried: this
is a reproducible bounded probe, not a hermetic toolchain package.
