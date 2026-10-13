# Tracked allocation queries

`hipMemPtrGetInfo` reports the whole logical allocation size for a live base or
interior pointer. `hipMemGetAddressRange` additionally reports its base. Queries
reuse the data registry and the runtime gate; they do not initialize a GPU queue,
access allocation contents, or report physical/free device memory.

The semantic reference is pinned CLR65d174c3e35423bf25c9c369513780c9d9ca760c,
`hipamd/src/hip_memory.cpp:715-728,775-792`, and HIP
d3b75fbb82113374e2d6103baa96a9c61425da48 declarations. CLR returns the complete
memory object's size even for an interior pointer. Unknown pointers return
`hipErrorInvalidValue` from PtrGetInfo and `hipErrorNotFound` from AddressRange.
The implementation additionally rejects null output pointers with InvalidValue
instead of dereferencing them. All outputs remain unchanged on an error.

Only owned, live logical allocations are recognized. Padding, one-past-end,
freed, null and unrelated host pointers are rejected. Existing registry ownership
checks remain in force: GPU-owned allocations return NotReady; poisoned state,
failed registration and shutdown retain their existing error precedence. A query
does not initialize missing compiler registration. Output values are published
only after a successful gate unlock. Concurrent freeing after that unlock does
not turn the returned metadata into an allocation lifetime guarantee.

This does not implement pointer attributes, host registration, managed memory,
physical memory reporting or asynchronous query behavior. The sanitizer model
checks API behavior and lifecycle only; native validation remains separate.
