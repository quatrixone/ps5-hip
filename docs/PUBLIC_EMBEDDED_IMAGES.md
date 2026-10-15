# Public HIP embedded image bring-up

The inference client uses compiler-generated registration and launches. Existing
public file-module tests do not establish that path. This work must use genuine
CLR registration, not substitute the older project runtime.

## Correctness prerequisite

`StatCO::digestFatBinary` previously assigned its cached pointer before extraction.
After an extraction error, the next call saw a non-null pointer and returned
success without retrying. The PS5 patch now owns a candidate locally, publishes
it only on success, and maps allocation failure to `hipErrorOutOfMemory`.
Failed candidates are destroyed and the cached pointer remains null.
Function and variable lookup reject that null slot explicitly. Late registration
logs extraction failure instead of asserting, retaining the slot for normal
lookup rejection and cleanup.

The focused regression compiles the actual prepared function body under ASan
and UBSan. The original body demonstrably returns false success on retry;
the patched body covers repeated errors, constructor/extraction allocation
failure, late-registration retry, null function/variable lookup rejection,
successful retry, cached success and final cleanup. This is a
prerequisite, not embedded-image support.

## Remaining implementation

The path is the exported registration entry point, `hip_platform.cpp` wrapper
handling, `PlatformState::addFatBinary`, `StatCO::addFatBinary`, deferred digest,
`FatBinaryInfo::ExtractFatBinary`, program loading and the PS5 device adapter.

1. Validate exact linker-owned wrapper and bundle spans before dereferencing
   the wrapper. Reuse `select_host_module` and `read_single_host_wrapper` parser
   primitives. Reject absent spans, malformed offsets, overlapping entries,
   wrong wrapper fields and out-of-range pointers without guessing an extent.
2. Generate per-object span records for actual compiled client objects. The
   existing native linker script names `multi-a.o` and `multi-b.o`; it is not
   a generic client linker layout. A merged bundle range is insufficient.
3. Carry validated extent/identity through the genuine static registration
   path. Do not enable arbitrary pointer-only `hipModuleLoadData` as a side
   effect. Preserve bounded file-module loading.
4. Own the selected code bytes in `ps5_image_storage_`; set `image_` consistently
   with the per-device image. The upstream destructor treats differing image
   pointers as potentially separately allocated ownership. An interior pointer
   into a linked or copied bundle must not become a `delete[]` target.
5. Qualify compiler function registration after a rejected handle, eager loading,
   stale unregister and cleanup must have defined behavior. Preserve ownership
   when GPU completion is uncertain.
6. Qualify actual compiler constructors and launch stubs linked to CLR: one
   module, two modules and reversed order; malformed metadata and registration
   failures; retry and initialization failure; exact output and cleanup. Then
   freeze a native candidate and validate on .40 under the console protocol.

The public adapter's launch geometry, LDS, argument ABI and model allocations
remain separate client blockers. Passing registration alone will not qualify
the LLM backend.
