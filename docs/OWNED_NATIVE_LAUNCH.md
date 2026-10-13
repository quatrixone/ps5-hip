# Shared owned native launch preparation

This development step starts at planner integration `296222d`. One preparation
helper now allocates and packs the selected initial, LDS, buffer-scratch or
geometry route. It reinspects owned code-object bytes and requires a real selected
kernel reference with unchanged metadata. Multiple kernels may share the object;
only the selected entry is planned. Targets remain restricted to gfx1013.

The caller supplies owned argument bits, a serialized live data registry and an
explicit request. Pointer extents must equal the actual registry remainder;
scalar widths, entry resources, shape products and LDS/scratch capacities reject
before native allocation. This does not infer kernel-specific indexing bounds,
pointer alignment requirements beyond the existing planner contract, or dynamic
LDS requirements from shader code. The caller must establish these. The exact
vector wrapper still enforces its single-kernel, 73-element, aligned float profile.
Host argument capture, registration and HIP admission remain unchanged.

`PreparedNativeLaunch::temporaries` explicitly owns code, kernarg and optional
scratch; `resources` includes these plus unique data owners for SubmissionQueue.
The helper checks full rounded backing ranges against all permitted registry
allocations and each other before writing code or arguments. No backend is
initialized during preparation. There is no private-call route: eight-SGPR
private304 metadata cannot substitute for accepted compiler-graph provenance.

The caller retains the prepared launch and queue through submission. After known
completion, or before any submission, it calls `close_native_launch_temporaries`.
That checks all temporary GPU leases before closing anything and closes in reverse
order. Any retained cleanup poisons the registry and raises a typed error. Partial
setup explicitly reconciles already-created temporaries; a retained constructor
failure is terminal even when other temporaries close. On uncertain submission,
the queue remains poisoned with all GPU leases, including scratch; callers must
retain it and refuse reuse/free/unregister success. A destructor is not evidence
of successful native cleanup. The allocator and queue implementation are unchanged.

The existing vector caller now closes the explicit list instead of assuming the
first two resource entries. Regression models use real mapped host pages and
actual native allocation/registry/queue code, with syscall/backend doubles. Seven
routes alternate twice on persistent data using genuine O1/O2 compiler objects;
the models check packed arguments, temporary ownership and unchanged data. They
also exercise three-stage allocation/map failure, full-backing alias rejection,
clean pre-submit failure, returned/standard/unknown submit failures, kick/clock
failure, timeout, and completed/partial cleanup failures. These callbacks only
publish queue completion; they do not execute or emulate GPU arithmetic.

ASan models deliberately map bounded slots in the 8 to 12 GiB shader window with
`protect_shadow_gap=0`. These mappings are not heap-redzone allocations and do not
prove native allocator safety. No package, native execution, generalized HIP
support or combined hardware qualification follows from this step.
