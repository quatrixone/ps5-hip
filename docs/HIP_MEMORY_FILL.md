# Tracked synchronous memory fill

`hipMemset(void *, int, size_t)` fills a live logical allocation, including an
interior range, with the low byte of its value. It uses the existing runtime
gate and CPU mapping. Supported launches finish before returning, so a fill
waits for an earlier caller's submission and completes before returning itself.
This is a blocking subset; it does not implement `hipMemsetAsync`, streams,
pitched fills, or GPU-accelerated fill kernels.

Nonzero null, unregistered, freed and out-of-range destinations return
`hipErrorInvalidValue`. Zero bytes require no valid destination, but still obey
runtime registration, terminal and shutdown checks. Padding beyond the logical
allocation is never admitted. GPU-owned or poisoned backing remains inaccessible;
the shared registry owns these checks. No allocation or backend initialization is
needed for the fill itself.

The semantic reference is pinned ROCm 6.2.4: `hip_runtime_api.h`'s byte-fill
declaration and CLR `hip_memory.cpp`'s `hipMemset`/`ihipMemset` implementation.
Upstream can enqueue some device fills asynchronously; this implementation makes
the stronger blocking guarantee within its currently supported launch model.

Offline coverage extends the native allocation model and genuine compiler/runtime
thread model: interior writes, signed/truncated values, guards, invalid extents,
ownership/poison, zero sizes, worker calls, failed calls, closed lifecycle, and
a fill contending with an unfinished synthetic submission. Synthetic transport
does not validate PS5 cache visibility; a native fill-and-kernel client is still
required before hardware qualification.
