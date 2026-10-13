# Serialized native vector control (experimental)

This unit connects the pinned single-translation-unit HIP constructor and genuine
`<<<>>>` host stub to the owned native data registry, materialized code, 288-byte
kernargs and existing submission queue. It is a narrow executable control, not a
production HIP runtime or evidence of GPU execution.

Only the frozen ROCm 6.2.4 / LLVM18 gfx1013 vector object is selected. The exact
profile is 73 FP32 elements, two 64x1x1 groups, wave32, null stream, zero dynamic
LDS, no scratch and the measured 13 COv5 geometry fields. Registration consumes
the existing authoritative linker spans; constructors allocate no native memory
and never initialize AGC. Argument storage is trusted readable storage from this
exact compiler stub. Its copied pointers must resolve inside current logical
registry allocations. Raw pointers do not solve address-reuse ABA.

Preparation reinspects owned object bytes, validates the selected kernel and all
logical extents, and owns code/kernarg storage plus unique data owners. Entire
native backing ranges must be disjoint from launch storage. AGC and its single
queue are initialized lazily only after preparation succeeds. Initialization is
attempted at most once. `hipLaunchKernel` blocks on the existing bounded completion
protocol; D2H then copies completed, cache-flushed storage. This deliberately
provides no asynchronous/default-stream or multi-thread scheduling semantics.
`hipDeviceSynchronize` succeeds only in the nonterminal idle/completed state;
it returns NotReady for an entered queue operation and rejects terminal failure.
It creates no backend and does not pretend to drain unimplemented stream work.

Code and kernarg allocations are explicitly closed after completion and on safe
pre-submit errors. Queue command/marker allocations also have explicit cleanup
on setup/preflush failure. Typed cleanup errors distinguish failed native teardown
from unresolved GPU ownership: both fail the operation and terminate reuse, but
only the latter retains GPU-use references. A successful destructor retry cannot
turn an explicitly failed operation into success. Clean setup/preflush errors can
unwind without poisoning. Initialization errors cannot trigger another AGC init.

Once submission may have consumed work, submit/kick errors, timeout, clock errors
or exceptions poison the control and data registry. D2H, free, even supported
zero-size operations, relaunch and successful unregister are then refused. Queue
owners retain code, kernargs, all data and completion storage. The native client
quarantines rather than attempting title-local teardown on terminal failure.
There is no late-marker recovery. Normal compiler unregister remains genuine;
partial function registration releases the metadata-only module.

The full-header client performs two launches through the original compiled stub.
Three 16KiB allocations contain guarded interior vector data; H2D/D2H checks cover
all rounded backing, 73 exact output values per launch, all inactive-tail bytes,
and unchanged inputs. Host syscall/backend doubles exercise actual allocator,
registry, preparation and queue code; their callback writes arithmetic results
only to test transport and validation. It is not AMD ISA emulation. Dedicated
cases reject missing/corrupt results, changed input, guard writes, foreign/freed/
short pointers and unsupported shapes before backend initialization. Failure
cases cover clean and retained setup, preflush, initialization, completed cleanup
and uncertain GPU ownership; standard and unknown exceptions cannot cross C ABI.

`tools/probe_native_vector.py --reference-root <checkout> --output <fresh-directory>`
reuses frozen combined objects and full HIP headers, preserving their hashes. Its
`--host-only` checkpoint supports source review before PS5 compilation/linking.
Native linking uses a private copy of the pinned linker export, reviewed AGC
provider stubs and raw/converted wrapper/bundle/constructor/RELRO checks. Linker
stubs are import declarations, never staged GPU implementations. The old
registration/capture/memory controls remain separate and unmodified.

No native execution, package/stage selection, shader hardware qualification,
all-HIP compatibility, throughput or framework claim follows. The raw executable
also lacks the packaged title's Eden and shell-managed startup/teardown integration;
it must not be deployed as a qualified title. Installed compiler resource headers
and static host archives are not fully inventoried, so the recipe is not a
hermetic toolchain package.

The host models deliberately map their private backing in the production shader
address window (8 to 12 GiB). ASan normally reserves this range as its shadow gap.
These tests use `ASAN_OPTIONS=protect_shadow_gap=0` and bounded `MAP_FIXED`
model mappings; ordinary `MAP_FIXED_NOREPLACE` was experimentally rejected by the
ASan reservation. This special host address-layout accommodation is not a native
allocator change or evidence about console address mapping. Explicit byte guards
and ownership/extent checks remain required alongside sanitizers.

Custom mmap-backed model ranges are not heap-redzone allocations. Explicit full
extent guards detect buffer overruns; the ASan address accommodation and passing
host checks do not certify the native allocator or console mapping safety.
