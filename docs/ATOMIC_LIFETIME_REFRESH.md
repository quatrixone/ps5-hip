# Atomic candidate lifetime refresh

This isolated refresh starts from frozen atomic candidate `2b863a7`. It copies
only `include/ps5hip/native_memory.h`, `src/native_memory.cpp` and
`src/submission.cpp` from `f17adc3`. The memory setup exception records whether
cleanup retained backing. Submission now explicitly unwinds command and marker
temporaries on pre-submit failure and distinguishes cleanup failure from uncertain
GPU ownership. The atomic harness, loader, encoder and device code stay unchanged.

Focused syscall models check clean versus retained setup failure, GPU-use state,
marker allocation failure after command allocation, upload-flush failure and typed
pre-submit/completed cleanup failure. Existing atomic corruption/quarantine models
continue to exercise six ordered dispatches and 98,694 exact values per suite.

The original worktree, four O1/O2 device objects and old native package remain
frozen. The old gfx1013 package is `gfx1013-atomic_suite-da51f067626d`, image
`2696bbfdbd1510161dbb05625c1870c0e1b625ca3f94008b5f659ef78f1cb845`
(3,637,248 bytes). Its offline package qualification supersedes the historical
"no native image" wording in `ATOMIC_DIAGNOSTIC.md`; it was not a GPU atomic pass.

Fresh qualification requires all 29 original sanitizer CTests, both target native
relocatable compiles, and exact four-object/source/driver preservation. No device
recompile is needed. A new native package, independent artifact review and guarded
console cycle remain prerequisites for hardware qualification. This refresh does
not integrate the later general planners or advertise new HIP APIs or memory orders.
