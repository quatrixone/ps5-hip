# Owned native data boundary (experimental, offline)

`NativeDataRegistry` owns actual `NativeMemory` data allocations. It exposes the
requested logical extent, never the rounded 16 KiB backing padding. Interior
pointers resolve to shared backing holds plus offset and remaining logical bytes.
The complete mapped backing extents must be disjoint. Code, kernarg, packet and
completion allocations belong to the existing launch/queue owners instead.

Copies validate device extents before touching memory. Host storage must already
be valid, readable/writable and stable for the whole call; numeric checks cannot
prove arbitrary host memory is mapped. Host ranges overlapping any registered
backing (including padding) are rejected. Zero-byte core operations are rejected;
this is not yet a HIP ABI implementation or a declaration of HIP zero-size rules.
The CPU copies do not submit GPU work or perform the existing queue's cache flush.

All calls, holds and submission transitions require one serialized owning caller.
A shared hold keeps the C++ owner alive but is not a GPU-use lease. Explicit free
may close an ordinary held backing; later `acquire_gpu_use` rejects that closed
owner. The caller must acquire its GPU use before any free can intervene, and
retain/release it through the existing queue completion protocol. A raw pointer
cannot distinguish address reuse (ABA); this inventory certifies only current
allocations. Do not mutate or close the held NativeMemory outside its registry
and queue ownership protocol.

Free requires a current exact base and refuses GPU-owned buffers. Cleanup failure
retains the owner, retires the record and poisons the registry. Poison rejects all
further registry work. Native setup errors remain `NativeFailure` with the native
detail; the registry does not guess out-of-memory from message text. The small
`NativeMemorySetupError` subtype reports whether setup cleanup retained backing;
only failed cleanup poisons, so clean setup failure leaves existing buffers usable.
Allocation of C++ bookkeeping may throw `std::bad_alloc` before mapping.

Keep a poisoned registry and queue alive until process termination: there is no
reset, retry or recovery API. At destruction ordinary non-GPU backing may close
(and retry failed kernel cleanup); `NativeMemory` refuses to unmap or release any
unresolved GPU backing even when the final shared owner is destroyed. That kernel
allocation is deliberately retained, but its C++ bookkeeping cannot be recovered.
This matches existing queue failure semantics without a global graveyard. Tests
isolate unresolved GPU ownership in a child that exits; completed cases clean up.

Host syscall doubles exercise the real NativeMemory implementation with aligned
host backing, including allocation/map/unmap/release failures and a faulty provider
returning overlapping mappings. These checks do not qualify native GPU mapping,
cache visibility or runtime lifetime. A PS5 cross-compile only checks declarations
and emitted object compatibility. Existing compiler registration and capture
controls remain unchanged; the core registry adds no GPU submission or success claim.

The separate `tools/hip_memory_control.cpp` does define seven experimental C ABI
symbols against pinned ROCm 6.2.4 full headers: malloc, free, H2D/D2H memcpy,
last-error get/peek, error string and unsupported launch. The ordinary full-header
client uses the upstream typed malloc overload and checks real owned copies.
Supported zero-size operations first check/initialize the memory registry and
reject terminal poison. Other copy directions and all launch requests return
NotSupported. Only C++ allocation failure maps to OutOfMemory; opaque native
failure maps to Unknown with its detail printed. Exceptions never cross these
C ABI definitions. Last-error state is one serialized control caller, without
per-thread qualification. The output pointer to malloc must be valid writable
host storage. This control must never be linked together with the separate
registration control's overlapping ABI symbols; neither is a production runtime.
