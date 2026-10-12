# Single-TU HIP host registration control

This isolated G4 step consumes an actual LLVM 18/ROCm 6.2.4 combined object
through its unchanged compiler constructors and ordinary launch stub. It is a
serialized, metadata-only control, not a public HIP runtime. No GPU calls,
native execution, native package or console test are part of this step.

## Concrete implementation

`read_single_host_wrapper` requires caller-authorized readable host-image spans.
It checks span arithmetic, disjoint regions, one exact 24-byte wrapper slot,
magic/version/reserved fields and a relocated binary pointer equal to the start
of the supplied bundle. Only then does it call the bounded `read_fat_binary`.
The address checks do not discover mapped memory or make an arbitrary pointer
readable; the caller must provide genuine immutable extents.

`HostRegistration` owns the copied device object and binds opaque host function
tokens to actual kernel names. One module may be active. Failed module admission
does not publish a handle; failed function admission adds no binding. Unregister
releases the owned object and bindings. Small retired handle records remain
until registry destruction so stale handles cannot alias a later registration.
There is no GPU mapping, pending dispatch or asynchronous retirement to model.

`hip_registration_control.cpp` uses the pinned HIP header's real types and
configuration/launch declarations plus the measured CLR registration signatures.
Its exports are `__hipRegisterFatBinary`, `__hipRegisterFunction`,
`__hipUnregisterFatBinary`, `__hipPushCallConfiguration`,
`__hipPopCallConfiguration` and `hipLaunchKernel`; libc provides `atexit`.
The saved compiler stub passes its real function identity and argument-address
array. The control verifies a 2-group, 64-thread, 73-element vector request,
records `hipErrorNotSupported`, and never submits work. A second build deliberately
rejects the registered symbol and unwinds the partial module before `main`.

The function-local `State` is constructed during the genuine compiler
constructor. Its destructor is registered before the compiler registers its
module destructor. Therefore the compiler's `atexit` unregister callback runs
after `main` and before the live `State` destructor checks zero active owners.
The parent process checks this actual output order and exit status. There is no
manual call substituted for the compiler destructor, and no callback observes
the registry after its destruction.

## Linker and conversion boundary

`probe_host_registration.py` uses fresh output directories and byte-identical
saved Linux/PS5 gfx1013 vector objects. Their manifest is pinned by
`evidence/offline-compatible-hip-headers.json`; no HIP device compilation or
LLVM build occurs. Source HIP/CLR headers are checked against the pinned archives,
and the authentically generated version header is checked against its receipt.

The Linux linker fragment and private native linker copy retain bundle bytes in
read-only storage, wrapper bytes inside `.data.rel.ro`, and hidden start/end
symbols for both. They also bound the read-only kernel-name strings. The copied
native export is verified against the existing LDS native-source inventory;
shared source files and SDK inputs are not edited.

The audit checks the raw PS5 PIE and converted `eboot.elf` separately: exact
8136-byte bundle, exact 24-byte wrapper, read-only bundle PT_LOAD, wrapper
RELATIVE relocation to the bundle start, and compiler constructor relocation
inside the native initialization array. The raw PIE has a linker-designated
RW `.data.rel.ro` load; the converted image additionally has `PT_GNU_RELRO`.
The audit models relocation arithmetic at several bases; it does not execute
the PS5 loader. Full import identities are recorded in `native-import-names.json`.
The observed native image has 59 imports from `libkernel` and
`libSceLibcInternal`, including `atexit`, `__cxa_atexit`, allocation, libc and
pthread support. HIP registration/launch are definitions, not unresolved imports;
there are no AGC imports.

## Validation and remaining limits

The new sanitizer model exercises invalid slots, null/interior/endpoint bundle
pointers, overflow, overlapping spans, short/oversized bounds, wrong wrapper
fields, failed registration, symbol errors, ownership after source mutation,
unregister, stale handles and repeated module lifetimes. Linux normal and
partial-failure controls execute the genuine compiler constructor/destructor.
The focused CMake test is `host_registration_model`, enabled with an explicitly
supplied verified bundle. Raw commands and outputs are under ignored
`results/host-registration-v*` and `results/cmake-registration-v1`.

An earlier control used a destructor-bearing `thread_local` configuration vector.
The actual native converter rejected its unresolved `__cxa_thread_atexit_impl`;
the failure remains in `host-registration-v4/native-convert.stderr`. The final
single-thread control stores its stack in `State`. This does not implement or
qualify HIP's thread-local configuration/error behavior. The ordinary native
startup still calls `exit` after `main`; that native teardown path is known to
need separate work and is not executed here. Linux lifecycle success cannot
establish PS5 libc callback ordering or process shutdown.

The fixture's workitem-only device-library linkage retains the limits recorded
in the host ABI spike. Multiple translation units, multiple active modules,
globals, asynchronous unload, dynamic shared memory, general dimensions,
threaded calls and working HIP launches remain outside this control. The small
registry does not select the production runtime architecture; section 3.2's
comparison remains necessary before broad G4/client integration.
