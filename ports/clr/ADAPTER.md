# Bounded PS5 ROCclr device adapter

This slice connects genuine ROCclr `Device`, device `Memory`, `Program`, `Kernel`
and `VirtualDevice` classes to the existing checked native memory registry,
argument capture, launch preparation and submission queue. It has no replacement
host scheduler and exports no new public HIP success path.

Reproduction starts with the pinned CLR checkout and existing `ps5-platform.patch`
documented in `README.md`. Apply `ps5-lifetime.patch`, `ps5-device-aot.patch`,
`ps5-affinity.patch`, then `ps5-executable-path.patch`,
using `git apply --check` before each application. `adapter-manifest.json` records
the order and exact original/adapted hashes. The standalone lifetime patch also
serves the CPU diagnostic; it does not depend on the AOT adapter or ignored
preparation scripts. Build the adapter with the manifest's three defines, existing
TLS helper, native substrate and an explicitly selected backend factory.

The startup overlays use a 16-byte native affinity argument with zeroed 32-byte
SDK backing storage and reject unsupported upper mask bits. Native executable-path
discovery reports unavailable through empty outputs; Linux retains its existing
`/proc/self/exe` query. The CPU-only diagnostic completed a native console cycle
with these overlays. That result does not validate the GPU adapter or its provider
calls. Reproduce patch composition and focused native/Linux behavior with
`python3 tests/clr_startup_port_test.py --upstream /path/to/pinned/clr` from the
repository root (Git and Clang with ASan/UBSan are required).

An AOT executable enters through upstream `addDeviceProgram`, `load` and
`setKernels`. The adapter owns the validated code object and device kernels;
published symbols are immutable. Captured upstream commands retain kernel and
buffer ownership. Upstream host queues perform dependency waits and event/callback
publication. Device operations serialize on a checked mutex, then publish terminal
event status after unlocking so callbacks may safely reenter.

The admitted slice is deliberately bounded: ordinary buffers, synchronous read,
write and tracked device copy, markers, and the existing scratch/LDS-free initial
launch route. Unsupported commands receive an error event. Other factories and
capability queries fail explicitly. This is not a complete ROCclr or HIP port.

Uncertain native submission, poisoned registry or failed teardown retains the
whole device state. A failed queue join also requires retaining its enclosing
context and manually owned device; a retired virtual-device counter is not proof
of a joined thread. The diagnostic clients use checked queue-termination outcomes
before deleting owners. They do not retry a zero-reference retained queue.

Full-source sanitizer qualification exposed upstream issues in detached thread
retirement, volatile thread-state publication, relaxed final reference retirement,
semaphore alignment and tagged-pointer member calls. The isolated preparation
recipe carries the corresponding small lifetime patch, including checked joins,
early self-release rejection and partial startup cleanup. Historical failing
results remain available; a successful diagnostic run does not supersede them
without a source-bound qualification of the corrected copy.

The adapter model uses a genuine compiler-produced code object and actual CLR
commands, capture, event dependencies and native packet preparation. Its transport
is explicitly synthetic: it checks real packet/ownership structure and performs
CPU arithmetic solely as an output oracle. It proves no GPU execution, native SDK
provider availability, or native title startup. ASan's fixed shader-address model
requires `protect_shadow_gap=0`; intentional quarantine processes disable leak
reporting and instead check exact retained backing/release counts. Normal cleanup
uses leak checking. Native linking and execution remain separate gates.
