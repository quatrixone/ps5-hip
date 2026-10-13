# Single-device discovery

The runtime exposes its configured PS5 compute route as device 0 through
`hipInit`, `hipGetDeviceCount`, `hipGetDevice`, `hipSetDevice`, and
`hipDeviceGetAttribute`. Discovery creates only CPU thread/error bookkeeping;
it can precede compiler registration and never creates the GPU backend.
The count is one configured route, not a fresh hardware health probe.

`hipInit(0)` is idempotent; other flags return `hipErrorInvalidValue`.
The only accepted device ordinal is zero. All threads therefore select zero;
an invalid selection returns `hipErrorInvalidDevice` without changing it.
Null outputs return `hipErrorInvalidValue`. Only `hipDeviceAttributeWarpSize`
is supported and returns 32, matching the admitted wave32 execution ABI.
Other selectors return `hipErrorInvalidValue`, as the pinned CLR fallback does.
No physical memory capacity, PCI identity or unqualified capability is reported.

An output remains unchanged on every error, including mutex-unlock failure.
Lifecycle and infrastructure failures take precedence over invalid arguments.
Discovery preserves registration failure and terminal-state rejection; success
updates the calling thread's last error just like other APIs. Calls after the
internal finish boundary return `hipErrorDeinitialized` and cannot resurrect
the runtime.

The semantic reference is the repository's pinned ROCm 6.2.4 HIP headers and
CLR device/context implementation. Discovery does not require explicit hipInit
before a query. It does not provide device reset, properties, allocation before
module registration, or clean discovery-only internal finalization. A program
that only queries devices retains the existing process-lifetime key/context
policy; successful compiler unregister/State destruction is not claimed for it.

Offline tests run queries from a worker and main before genuine compiler
registration, then verify normal registration and finalization. Additional
cases exercise implicit initialization, repeated calls, invalid flags/ordinals,
unknown and unsupported attributes, unchanged error outputs, thread error
isolation, context OOM recovery, failed registration, terminal transport,
abandoned callers and shutdown. A failed first test is retained: INT_MAX was
outside the C++ enum's representable range and triggered UBSan. The unknown
selector test now uses representable value 30000, without disabling sanitizers.

These are host-model results. Native symbol/provider checks and a PS5 client
must pass before a hardware qualification claim.
