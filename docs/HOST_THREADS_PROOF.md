# CPU-only host threading proof

This standalone experiment precedes changes to the serialized HIP runtime. Two
real pthread workers use explicit 512 KiB stack attributes. While main owns the
work mutex, both workers must observe EBUSY. A separate mutex and condition
variable then order 256 alternating updates, including TLS/key checks across
waits. Worker TLS values are 256 and 257; main remains 29. Each worker installs a
pthread-key value and its actual thread exit must invoke exactly one destructor.
Main clears its own key value before deleting the key.

All successfully created workers are joined, including partial creation failure.
Synchronization objects and the key are explicitly closed after known joins.
Any failed join or mutex unlock preserves the static backing, key and synchronization objects;
cleanup failures cannot report success. The proof runs once per process. Worker
waits share one absolute three-second realtime deadline. pthread_join itself has
no portable timeout: native operation still requires the external bounded title
controller; the host recipe bounds every child at twelve seconds. No cancellation
or detached worker is used. Failed mutex unlock/join operations are not evidence
that an operating system guarantees worker termination.

The finite title reuses the existing Eden filesystem request, exclusive result
log, persisted checkpoints, and shell/fini helper. Its separate finalizer must be
installed after the shell helper and before application initialization. Only a
successful report with known cleanup permits final proof and normal fini. This
source checkpoint has not yet been linked, packaged or run on PS5. Native ELF TLS,
actual pthread providers and startup order must be audited after source review.
No HIP runtime, AGC initialization, GPU allocation or GPU submission is linked.
Only trivial scalar C++ TLS is used; destructor-bearing C++ TLS remains excluded
because the earlier converter proof could not resolve __cxa_thread_atexit_impl.

Run the offline proof with:

```
python3 tools/probe_host_threads.py --reference-root /path/to/ps5-hip --output /fresh/output
python3 tools/probe_host_threads.py --check /fresh/output
```

The Linux ASan/UBSan cases call real pthreads. Failure injection redirects only
calls in the proof translation unit, leaving sanitizer/libc internal calls alone.
Cases exercise partial creation, key/stack setup, timeout, missing contention,
missing/duplicate exit destructors, join uncertainty and cleanup errors. The join
error double performs the real join and then reports an error; unlock failure doubles
perform the actual unlock before reporting failure so test workers are not stranded.
Both cases assert zero key, condition and mutex destruction calls. This tests fail
closed ownership classification, not a live abandoned thread. A separate normal
binary has unmodified API calls. The title models reuse the actual shell helper
and proof, replacing only elevation, final shell request, persistence failures
and quarantine sleep. These host observations do not qualify native threading or
worker-thread AGC behavior. Installed host headers and archives are not fully
inventoried; this is not a hermetic toolchain package.

The pinned SDK compiler uses `-femulated-tls`, with scalar descriptors/templates
and `__emutls_get_address` from its libc, rather than a hardware TLS data section.
The host recipe additionally compiles the exact pinned SDK `libc/emutls.c` and
replays all 18 core cases with that lowering under ASan/UBSan. The original
source checkpoint read scalar TLS inside the independent proof-key destructor;
the saved SDK-emutls reproduction correctly failed because the emutls key may
already have freed its storage, causing a fresh scalar lookup to return 17.
This is an invalid destructor-order assumption, not a demonstrated use-after-free.

The corrected key destructor checks only its own static Worker identity/context/id
and exactly-once count. Scalar TLS isolation remains checked in the worker body,
across the condition waits, and observed immediately before return. No behavior
depends on order between the proof key and emutls's key. Native auditing must bind
the actual emutls descriptors, initializers, helper and providers; it must not
claim a populated PT_TLS segment when the SDK uses emulated TLS.