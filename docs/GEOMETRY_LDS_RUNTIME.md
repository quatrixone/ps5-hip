# Combined 2D workgroups and static LDS

`NativeLaunchRoute::GeometryLds` prepares the measured `blockDim` tiled-matmul
entry through the same capture, ownership and submission interfaces as other
routes. Selection depends on metadata and launch shape, not the function name.
This is an offline-qualified launch path; native execution remains pending.

Admission requires wave32, six user SGPRs, natural XY register entry `0x98c`,
properties `0x409`, static LDS512, no scratch/spills/preload/resource3, and the
complete thirteen-field COv5 hidden geometry area. The measured block is 8x8x1;
grid X/Y are checked for zero and overflow and Z must be one. Dynamic LDS is
rejected. The measured explicit-only 36-byte variant remains outside this
profile. Existing separate geometry, LDS and scratch contracts are unchanged.

The pure planner reuses the existing argument packer and dispatch encoder.
LDS512 rounds to 1024 bytes under the existing allocation policy and adds
`0x10000` to RSRC2, producing `0x1098c`. The encoder checks the complete plan and
the hidden geometry bytes against the dispatch shape. No host allocation backs
LDS; owned temporaries remain code and kernargs, plus the existing data owners.

The native helper reinspects owned object bytes, checks the selected kernel
identity, validates dimensions and current logical buffer extents before native
allocation, and retains the existing poison/quarantine behavior. As elsewhere,
the application remains responsible for shader indexing bounds. Metadata and
live pointer ownership do not prove arbitrary shader accesses are safe.

`geometry_lds_test` checks actual Linux/PS5 O1/O2 objects, exact packet registers,
implicit arguments, wrong-profile rejection and mutated metadata/plans.
`native_geometry_lds_test` exercises actual capture, mapped allocations, repeated
submission ownership, setup failures, uncertain completion and cleanup failures
with the existing host doubles. These models do not execute AMDGPU instructions.
The unchanged route models are part of the recorded sanitizer qualification.

The compiler fixture main is a measurement input, not a ready console program.
A runnable client still needs the established application failure cleanup and
lifecycle wrapper, followed by a newly frozen package and native validation.
