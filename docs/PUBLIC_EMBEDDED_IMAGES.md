# Public HIP embedded image bring-up

The inference client uses compiler-generated registration and launches. Existing
public file-module tests do not establish that path. This work must use genuine
CLR registration, not substitute the older project runtime.

## Native checkpoint

The bounded compiler-launch case passed on .40 with source `e365c5d`: two GPU
launches, 146 exact values, 16 guards, one allocation/free and one retired queue.
Native finalization, natural exit, installed file readbacks, matching idle mount
snapshots, service health and both lock releases passed independent review.
See `../evidence/console40-public-embedded-success.json`. This proves one native
module/kernel with deferred loading. Native eager/multiple-module behavior,
complete static-registry ownership accounting and LLM inference remain open.
The experiment contract below is now executed; its controls are consumed.

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

## Qualified offline implementation

Bounded compiler registration now runs through genuine CLR. Exact per-object
linker spans validate wrappers, names and bundles before use; selected device
bytes are owned by the static module. Pointer-only dynamic image loading stays
unsupported. Registration rejects invalid/stale handles and unknown kernel
names, and unregister retains ownership when synchronization fails.

The final host qualification covers one module, two modules and reversed
object/span order with eager and deferred loading. Actual compiler constructors,
launch stubs and destructors execute; GPU transport is modeled. Output no-op
and corruption controls fail as required. A bounded but absent kernel name
previously aborted in both loading modes; attribute lookup and real stub launch
now reject it twice without submissions. The complete 87-compilation CLR host
suite and the 76-object native bootstrap package also pass. See
`../evidence/offline-public-embedded-images.json`.

Reproduce from source with fresh output directories:

```sh
make public-hip-embedded-host-test \
  QUALIFICATION_DIR=results/public-hip-host-fresh \
  EMBEDDED_FIXTURE_DIR=build/public-hip-embedded-fixture-fresh \
  EMBEDDED_QUALIFICATION_DIR=results/public-hip-embedded-fresh
```

The native bootstrap build establishes cross-compilation and packaging, not
native compiler-stub execution. The host finalizer checks modeled device
backing and queue cleanup; it does not count live static registry entries.

## Implementation gates and remaining native work

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

Items 1-5 now have bounded implementation and focused regressions. Item 6 has
the host compiler/CLR evidence above; broader initialization-failure coverage
and native execution remain open. Next integrate the native compiler object,
audit constructor/destructor order and linker spans through ELF/SELF conversion,
then freeze and validate the actual compiler-launch title on .40. Do not
substitute a bootstrap package or the modeled transport for that evidence.

The public adapter's launch geometry, LDS, argument ABI and model allocations
remain separate client blockers. Passing registration alone will not qualify
the LLM backend.

## Next native experiment: actual compiler launch

Planned target: idle `.40`, firmware 12.70, title `PPSA99014`; FTP 2121,
klog 3232 and ELF loader 9021. This section is a test contract, not a run receipt.

Hypothesis: the genuine compiler constructor can register its bounded linked
image, and its `<<<>>>` stub can dispatch the already-inspected `thread_words`
kernel through CLR/AGC, then unregister before CLR runtime teardown. The
known-good control is the existing file-module launch path; the changed variable
is compiler registration and launch instead of explicit module APIs.

The candidate must use the generated per-object linker layout, append the HIP
object after CLR runtime initialization, and pass the raw/converted/extracted
image audit. Freeze the committed source, compiler receipt, package and all
installed file hashes before deployment. Use the configured results directory
and Eden/Lapy helper; preserve the currently installed image/log as baseline.

Acceptance: two actual compiler-stub launches, each with 73 exact arithmetic
values and eight untouched guard words; one allocation/free; no terminal or
uncertain submission; one queue retired; device gate destroyed; native finalizer
and natural title exit observed. The application watchdog is 30 seconds. Console
observation must have a bounded controller deadline and fresh title-specific
lifecycle evidence, with installed bytes and services checked after settling.

Reject on any API/output/guard mismatch, missing startup/finalization evidence,
wrong installed hash, missing title stop, or unhealthy service. A constructor
failure can precede application logging, so capture klog before launch. Do not
repeat a suspected panic. Release only owned locks and investigate offline.
Passing this case permits the next real client compile/operator test; it does
not prove LLM inference or complete static-registry retirement accounting.
