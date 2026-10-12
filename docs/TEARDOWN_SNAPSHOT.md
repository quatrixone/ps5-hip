# Read-only teardown snapshot diagnostic

This is a bounded read-only diagnostic, not a release certificate. Its two .40
observations remain inconclusive. It cannot satisfy or waive the existing G2/private304
release prerequisite. Source is isolated from all native compute candidates.

Version 2 executed once on idle .40 at 15:36 UTC on October 12. Both initial
queries returned 1110 (`0x456`) with errno 0, above the fixed 1024-record ceiling.
Neither snapshot allocated or issued a fill query. The complete transcript,
process/application absence, healthy services and owned-lock release are preserved
in [the observation](../evidence/native-teardown-v2-fw1270.json). No mount rows or
usable baseline were obtained. The same FD-budget warning recurred without an
established cause. Both diagnostic one-use markers are consumed.

After these two inconclusive observations, reassess the provider, record-layout
and enumeration contract offline. A revised budget may use the measured return
as evidence, but it needs a separately frozen and reviewed candidate. Neither
the raw return nor a larger buffer establishes namespace completeness or release.

The [offline reassessment](../evidence/offline-teardown-count1110-design.json)
proposes a fixed 2048-record ceiling with the same provider, layout and call
sequence. Under the SDK count contract, 1110 plus 16 spare slots requires 1126
records / 531,472 bytes; two tables require at most 1,933,312 bytes, excluding
allocator overhead. No
revised collector is implemented or run. The pinned FreeBSD 9 reference confirms
record-count returns and byte-sized buffers, but can skip busy or hidden mounts
and redact identities. A non-full validated returned buffer does not prove a
complete global namespace. Any revised case must retain unknown release status
and reject missing known registration anchors; it cannot itself qualify G2/304.

Version 2 adds query telemetry only. The version 1 observation returned
`count_bound` twice before allocating or filling a mount table. Its printed
`count=0 matching=0` were untouched result fields, not an empty enumeration.
Frozen machine code establishes that both initial imported return values were
nonnegative integers at least 1009, but did not retain their exact values.

Each version 2 snapshot prints an `INITIAL` record with an attempted flag,
signed and hexadecimal raw return, and captured errno. Exactly three `FILL`
records report attempted flags, requested record/byte capacity and signed/hex
returns with errno. Unattempted slots remain zero and must not be interpreted
as successful empty queries. These records are printed even after rejection.
The existing provider, 1024-record cap, 128-matching-row cap, three fill attempts,
query arguments and control flow are unchanged; no extra query is introduced.
The pinned SDK treats the initial result as a record count, but this source
contract alone does not qualify the deployed provider or firmware ABI. No raw
syscall replacement, byte-count reinterpretation or cap increase is made.

The first combined arithmetic cycle on .40 checked 146 exact values and finalized,
self-stopped and completed OS process/application-mount teardown. Its saved logs
do not establish the release of ShadowMount runtime or registered-image mounts.
They identify v1.6beta16. The matching upstream tag identifies commit
`8566c0294cbf37b55375602e950a0e6b6bb928d7`; its exit path clears tracking,
fakelib and kstuff state, while its scan path retains or recreates title nullfs
registration while the source remains available. This source explains a possible
persistent registration policy, but does not identify the deployed binary bytes.
A retained registered image or title mount is not a proven leak.

## Observations and limits

`tools/teardown_snapshot.cpp` queries only PID 134 with signal 0 and the existing
running-application/title-application APIs. A live/reused PID, EPERM or conflicting
app IDs cannot establish process absence. It does not enumerate detached children.
Two consecutive mount-table observations use the SDK's `struct statfs` and
`getfsstat(..., MNT_NOWAIT)`. No syscall number or kernel structure offset is
invented. The snapshot code validates the returned structure version and every
string before comparing or reporting it.

The allocation cap is 1024 records, with 16 spare entries and at most 3 allocation
attempts per observation. Full, overflowing or error results remain unknown.
There is an additional 128 matching-row cap per snapshot; exceeding it suppresses
all row output for that snapshot. The pinned SDK's 16/88/88-byte string fields
bound escaped output below 256 KiB. Every logged string is hex encoded. Unrelated
mount rows are never printed. All records, including duplicates, participate in
the sorted identity comparison; mutable free-space counters do not. Two matching
snapshots establish two matching observations, not atomicity between them.

The report distinguishes:

- Registered image: `/mnt/shadowmnt/PPSA99014_b6d9b2e4` and children.
- Title application path (reported as `runtime_layer`):
  `/system_ex/app/PPSA99014` and children. This label classifies the path, not
  its lifetime or owner; v1.6beta16 can retain this as loader registration.
- Sandbox: exact PPSA99014 or numeric-instance components below `/mnt/sandbox`
  or `/sandbox`.
- Other title-related mount records: exact title components in source/target,
  or consumers of the exact registered-image root.

Unknown filesystem types/sources remain raw observations. The classifier does not
claim every possible firmware namespace or path convention is covered. Exact
component boundaries reject PPSA990140 and foreign suffixes. Duplicate nullfs or
union layers are preserved rather than collapsed.

Every report says `namespace=unqualified release=unknown accepted=false`; the
payload always returns 2. `pid134_esrch_and_app_queries_empty=1` is only the precise
query result, not a statement about all own-title processes. A completed empty
visible table, privileged UID or matching snapshots cannot prove global mount
visibility, lack of outstanding references, or release of GPU allocations.
Transport EOF without the final END record is incomplete. A host caller must not
treat a successful transfer, a zero matching count, or any diagnostic exit as a
gate pass.

There is no unmount, nmount, mount, ioctl, rename/delete, credential change,
launch/close, process signal other than 0, kernel write, automatic elevation or
automatic retry. A denied/restricted view remains unknown. This checkout does
not add a console sender; existing reviewed dual-lock/idle/capture infrastructure
must be separately adapted, reviewed and explicitly authorized before any use.
The system calls have no intrinsic wall-clock deadline; the host controller must
bound the whole observation and classify an incomplete result as unknown.

## Offline build and checks

Use a fresh output directory; the recipe refuses to overwrite even failed output:

```sh
python3 -B tools/build_teardown_snapshot.py --sdk /absolute/pinned/sdk \
  --output "$PWD/build/teardown-snapshot-new"
```

SDK revision 95c08f27386fc698f6bbe21dde3030140a41d10b declares getfsstat in
`target/include/sys/mount.h`. Its provider is `libkernel_sys.sprx`, selected
explicitly instead of the SDK wrapper's default kernel_web. The recipe compiles
against the real headers, verifies all actual undefined symbols against a small
allowlist, requires the exact needed-library set, and confirms that getfsstat is
exported by exactly one of those pinned stubs: kernel_sys. It saves ELF/source/
tool/header/stub hashes and main disassembly. This establishes offline linkage;
the running firmware's ABI, symbol availability and namespace view remain untested.

`teardown_snapshot_model` runs the production collection/classification code
against host records, deliberately without pretending to reproduce the kernel ABI.
It covers count/allocation/query failures, growing/full tables, malformed strings/
versions, duplicate and reordered layers, changed source/target/type/flags/fsid,
the output cap, title-prefix collisions and PID/app-query uncertainty.
`teardown_snapshot_import_audit` rejects missing/wrong/duplicate providers and
mutating or unknown imports. The native disassembly must also retain signal 0 at
both PID 134 existence queries; permitting the `kill` symbol alone is insufficient.

After any future read-only observation, retained registered UFS must remain distinct
from per-app sandbox/runtime state. Even if runtime/sandbox absence becomes
established, the current literal image-unmounted criterion would remain unmet if
the registered image is retained. Resolve that criterion explicitly with evidence;
do not silently weaken it or change the environment to manufacture a pass.
