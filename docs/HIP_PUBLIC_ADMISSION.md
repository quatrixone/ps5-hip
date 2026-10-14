# Upstream public HIP admission (offline)

This opt-in slice connects the PS5 ROCclr adapter to the genuine upstream HIP
runtime on Linux. It does not replace HIP streams with a project scheduler.
`hipInit` registers the actual adapter `amd::Device` and its eternal context;
upstream HIP creates its `hip::Device`, default/graph pools, streams, events,
dispatch table and thread-local bookkeeping.

The bounded model uses two public streams. A host callback holds stream A while
an event recorded behind it becomes a dependency of stream B. B's event must
remain not ready until A is released. Synchronization and destruction then run
through actual HIP/ROCclr code. This is a software dependency and lifetime test;
the synthetic transport is never called and no GPU result is claimed.

The profile rejects COMGR-only operations when COMGR is absent, and graphics
entry points under `PS5HIP_CLR_NO_GRAPHICS`. It retains the real full dispatch
table. Other reachable HIP APIs are not thereby qualified. Public kernel/module
launch, graphs, asynchronous allocation, timing, IPC, native HIP TLS and native
startup require their own admission. This is not G5 acceptance.

## Lifetime contract

An external owner must not concurrently destroy or reuse the same stream.
Checked stream shutdown establishes the actual queue join before permitting
owner destruction. Successful shutdown is cached; failed or recursive shutdown
is permanent. The profile rejects subsequent ordinary HIP operations and retains
runtime/device/queue owners after a failed join. Device retirement preflights all
its streams before freeing pools. A retained worker's interaction with remaining
process global destructors requires independent review before native use.

The pthread TLS record explicitly owns dynamically created `HostThread` objects;
borrowed command-queue thread objects remain owned by their queues. Its destructor
does not call TLS accessors. Main-thread bookkeeping remains process-lived.
The last queue command is released only after a successful join.

The model checks normal, default-stream and thread-exit cleanup, sticky init
failure, actual pthread creation failure, an unconsumed failed join, and a worker
attempting to destroy its own stream. A final observer checks exact queue and
gate ownership after runtime teardown. Intentional failed joins retain the
ThreadSanitizer thread-leak diagnostic; that finding is not suppressed or
described as successful cleanup. Ordinary modes require leak detection.

## Reproduction

Apply `ports/clr/ps5-public-hip.patch` with `tools/prepare_hip_admission.py` to the
qualified adapter CLR source tree and pinned HIP source. The manifest requires
exact old and new bytes for all thirteen changed upstream files. Adapter and TLS
support sources remain in `ports/clr`.

`tools/qualify_hip_admission.py` rebuilds the complete real closure with source,
compiler-discovered header, tool and selected link-library bindings. ASan/UBSan
and TSan use separate output directories. These are Linux model proofs, not a
hermetic operating-system build or a native package qualification.

Device reset is explicitly unsupported in this bounded profile. The public dispatch returns hipErrorNotSupported before Device::Reset can release pools or destroy streams. Repeated reset while a host callback holds a cross-stream event dependency preserves the pool identity, live queues and pending event; dependency completion and normal/failed-join shutdown are tested separately. This does not establish process-global destructor safety for retained live workers.
