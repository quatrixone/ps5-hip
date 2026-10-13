# Native device-copy fixture

`hip_device_copy` consumes the frozen synchronous tracked D2D runtime. The
experimental host client includes the same two generic device bodies; only
`affine_words` launches. Production runtime, registry and backend remain unchanged.

For N=1,73,257, distinct allocations and disjoint regions of one allocation each
run two rounds. A GPU producer writes the source, the real D2D API copies it, and
a GPU consumer transforms the destination. Only seed receives H2D writes. All
allocations receive one initial byte poison; destination is never reset between
rounds. Changed seeds and salts, with explicitly checked changed producer values,
make stale second-round data detectable.

Every allocation is read back after each pipeline. Independent host calculations
check seed preservation, source preservation, copied destination, consumer output
and all prefix/middle/suffix guards. Exact totals: 12 pipelines,24 dispatches,
12 copies,1324 copied words/5296 bytes,2648 GPU output values,5926 readback word
comparisons,630 guard comparisons and21 allocations/frees. There is no extra
host-template comparison claim. Actual Linux/PS5 compiler objects bind the
32-bit unsigned argument/data ABI and unchanged device metadata/ISA.

The transport model consumes actual mapped input storage and packet pointers;
it never uploads CPU oracle outputs. A test-only `hipMemcpy` interposer forwards
normal calls unchanged and injects skipped first/second copy, shifted source or
destination, truncation, destination corruption and post-copy source mutation.
Fault offsets stay within guard storage. The second-round skip retains the first
round's data. Missing/corrupt GPU output and guard faults also fail the client.
Uncertain submission, kick, timeout, exception and postflush faults retain exactly
eight backings with zero releases; cleanup failure retains five with three releases.
These transport faults occur on the first producer, before D2D is attempted.

Compiler reproduction:
`tools/probe_hip_generic.py --compiler-only --title-entry --device-copy`.
Actual-runtime ASan/UBSan lifecycle reproduction:
`tests/hip_generic_title_test.py --device-copy`.
Both accept the existing reference/probe/output arguments. Profile checks use
`tests/hip_generic_title_profile_test.py --device-copy --compiler DIR`.
CMake cache entries `PS5HIP_HIP_DEVICE_COPY_MODEL` and
`PS5HIP_HIP_DEVICE_COPY_COMPILER` replay source-bound receipts.

Native execution remains unqualified until an independently reviewed frozen
build/package and bounded console cycle. Use the existing 35-second external
observation; failures stop immediately rather than consuming 24 timeout budgets.
No DMA, bandwidth, asynchronous transfer, concurrent GPU execution, general HIP,
framework readiness or full G4 completion claim follows from this fixture.
