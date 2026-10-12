# Offline compiler argument capture

This diagnostic extends the single-TU registration control with argument-value
capture and the existing initial launch planner/packet encoder. It is not a HIP
runtime or GPU execution test. The genuine pinned LLVM18/ROCm6.2.4 vector stub
still receives `hipErrorNotSupported`; no GPU function is called.

`capture_initial_arguments` accepts a validated kernel, a trusted readable
argument-address array, one trusted readable value span per explicit argument,
and an authoritative inventory of permitted live data-buffer extents. The
initial profile permits nonempty arrays of 8-byte global-buffer pointers and
4/8-byte by-value scalars. Metadata, array count, span arithmetic and value
membership are checked before copying each value. Copies own all argument bits
before the compiler stub returns; they retain no pointers into its stack.

Every data allocation must have a nonzero range below the existing 47-bit GPU
address limit. Overlapping or duplicate inventory records are rejected. A global
pointer must lie inside exactly one current allocation, excluding its endpoint;
its `LaunchArgument.buffer_bytes` is the allocation's remaining extent from that
interior pointer. Several arguments may alias the same allocation. A pointer to
an allocation omitted from the current inventory is rejected. This does not
detect stale-pointer address reuse or independently establish allocation life.

The inventory must contain permitted data allocations only, never code or
kernarg storage. The diagnostic checks each complete data allocation against its
reserved module/kernarg extents once. The existing `prepare_initial_launch`
continues to validate launch storage, supported geometry and argument ranges,
and generates the hidden geometry area. The vector control additionally checks
that every remaining buffer extent covers `n * sizeof(float)`; generic metadata
alone cannot determine a kernel's actual memory accesses.

`probe_host_registration.py --capture-launch` enables the diagnostic macro.
Without that option the existing registration-only control remains selected.
The unchanged genuine compiler object supplies three pointer-value slots and
one 32-bit scalar slot. Its metadata has offsets 0/8/16/24, thirteen hidden
geometry fields beginning at 32, and 288 bytes of kernarg storage. The control
uses explicitly synthetic GPU addresses, captures the values, prepares two
64x1x1 workgroups, encodes packets and saves the owned result before returning
`hipErrorNotSupported`. Main changes its caller values after the stub returns
and verifies the captured arguments and prepared bytes remain intact.

The compiler-produced argument array and slots are trusted live readable stack
storage for this exact measured stub. Constructing their spans does not discover
memory mappings or make arbitrary pointers safe. The synthetic data inventory
does not prove native mappings, visibility, physical ownership, asynchronous
lifetime or address reuse behavior. No allocator, scheduler, submission queue
or production runtime architecture is introduced.

The sanitizer model consumes the actual vector bundle and checks owned values,
interior extents, alias arguments, unaligned value storage, 64-bit scalar bits,
missing/retired pointers, count/span/range overflow, ambiguous allocations,
unsupported metadata and geometry, and launch-storage aliases. An independent
288-byte oracle includes all hidden geometry and padding. The genuine Linux
compiler constructor, launch stub and destructor execute in both normal and
partial-registration-failure controls. The PS5 control is cross-compiled,
linked and converted offline; it is never executed, packaged or deployed.

All raw outputs use fresh directories under ignored `results/`. Previous
registration controls, receipts, shared dependencies and native exports remain
unchanged. Native libc teardown and thread semantics retain the limitations in
`HIP_HOST_REGISTRATION_CONTROL.md`.
