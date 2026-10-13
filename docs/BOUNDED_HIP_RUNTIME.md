# Bounded reusable HIP runtime

The default policy in `src/hip_launch_profile.cpp` admits structural launch
classes rather than experiment names, expected answers, exact allocation sizes,
or fixed completion counts. The separately linked vector and routes diagnostic
adapters retain their existing fixed contracts. The same runtime, ownership,
argument capture, submission and teardown code serves both modes.

The reusable profile is one trusted compiler-generated module on one serialized
thread. It supports up to 256 registered functions, unused functions, and zero or
more blocking launches. The bound limits resources; it is not an expected count.
The accepted classes are the existing 64-thread 1D initial, LDS and buffer-scratch
routes, plus the measured 8x8x1 and 8x4x2 geometry routes. The separate
`GeometryLds` profile combines natural XY entry, an 8x8x1 group, static512-byte
LDS and the complete COv5 geometry area; see `GEOMETRY_LDS_RUNTIME.md`.
Combined scratch/LDS and other geometry/resource combinations remain rejected. Private-call entry layouts,
including the eight-SGPR 304-byte candidate, are not admitted by descriptors alone.

Capture validates the entire argument layout before reading compiler argument
storage. Pointer arguments must resolve into live owned allocations; the native
preparer rechecks the selected owned code object, backing overlaps and captured
extents. **The caller is responsible for the kernel's indexing bounds.** Neither
scalar inspection nor a live pointer extent proves an arbitrary shader safe.
Compiler argument pointers and linked names are trusted ABI inputs inside their
documented readable spans; this is not an untrusted host-pointer API.

Any failed module or function registration latches partial state in reusable mode
and retires prior bindings when GPU ownership is safe. Reading the last error does
not clear that latch. Uncertain submissions retain resources and poison subsequent
operations. Known cleanup failure remains terminal even when GPU completion is
known. Constructors perform no native allocation or AGC initialization. A zero-work
client can unregister and finish without creating a backend.

Lifecycle finalization means safe unregister/destruction and empty owned storage;
it does not certify application results. A clean numerical failure may finalize
its ownership while retaining a failed application result. Any future native title
acceptance must require both the application's checks and full lifecycle proof.

`tests/hip_generic.hip` exercises different names, two or three functions, unused
functions, changing launch counts, interior offsets, allocation extents, scalar
high words and 1D tail sizes. Its synthetic transport validates actual emitted
packets and computes independent host references; it does not emulate GPU ISA.
ASan models use explicitly bounded fixed mappings in the shader address window
with `protect_shadow_gap=0`. These mappings do not have heap redzones; explicit
guards detect the tested bounds failures. No native allocator safety follows
from that accommodation.

The compiler proof uses pinned ROCm 6.2.4 public headers and the existing LLVM 18
workitem/constants slice. Native proof is compile/link/import/RELRO inspection,
not native execution or a package. Installed compiler resource trees and static
host support dependencies are not completely inventoried: this is not a hermetic
toolchain package. Device discovery, streams, concurrency, broad scalar/pointer
ABIs, arbitrary modules, libraries and full G4 compatibility remain open.
