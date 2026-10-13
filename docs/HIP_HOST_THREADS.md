# Checked host threading

The runtime serializes complete blocking calls with one checked process-lifetime
pthread mutex. Errors and launch configuration stacks belong to checked pthread
key contexts. No runtime-owned C++ thread-local allocation is needed. Key or
mutex failure is sticky process failure; get-last-error cannot repair it.

State is allocated only during genuine compiler module registration. Failure
latches registration failure: later APIs and registration callbacks reject work
without retrying State allocation, including after memory becomes available.
If a worker exits with pending launch configurations, the open runtime becomes
unusable. Both peek/get-last-error return sticky hipErrorUnknown before looking
up or allocating a thread context. Finish then transitions error queries to the
closed-lifecycle hipErrorDeinitialized contract.

Join every caller before finish. Finish closes ordinary APIs, which subsequently
return hipErrorDeinitialized. Genuine compiler unregister remains admitted for
teardown. Pending configurations, live workers, poisoned GPU ownership or broken
synchronization prevent successful finalization and retain unsafe State storage.
Cancellation, recursive callbacks, signal handlers and detached callers are unsupported.

`tools/probe_hip_threads.py --reference-root ../.. --output results/threads-v1`
compiles a genuine HIP kernel and host launch stub, then tests the actual runtime
against synthetic packet transport under ASan/UBSan. Coordination uses condition
variables and runtime-only checked-lock interception, never timing sleeps.
Fault wrappers apply only to the runtime translation unit. An unlock-failure
wrapper releases the real mutex and reports failure; it does not model recovery
from a truly stuck native lock. Each mode has a bounded separate process.

This host proof does not establish native AGC thread affinity or concurrent GPU
execution. A separate frozen native experiment must qualify those properties.
