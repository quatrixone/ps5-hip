# Fixed atomic contention diagnostic

This isolated branch adds a macro-selected native harness and offline tests for
the existing six-entry atomic compiler probe. It does not add HIP APIs, change
the launch core or device kernels, or qualify GPU atomics. The native builder,
stager and log checker do not yet accept this profile. No native image was linked
or packaged and no console was contacted.

`PS5HIP_CANARY_ATOMIC_SUITE=1` selects six dispatches on one submission queue:
`atomic_ticket32`, `atomic_ticket64`, `atomic_cas32`, `atomic_cas64`,
`atomic_indirect32`, then `atomic_indirect64`. Each uses 257 groups of 64 threads
(16,448 invocations), fresh module/kernarg/counter/output allocations, and a
2,000 ms software completion timeout. The indirect cases additionally receive
16,448 explicitly zero-initialized offsets, checked unchanged after completion.
Driver calls themselves remain outside that software poll timeout.

Fetch-add starts at `2^width - 4096`. The final counter must be 12,352 and the
observations must contain each of the 16,448 modular tickets exactly once, in any
order. The original ticket kernels compile to 514 contending wave reservations;
the indirect kernels perform per-lane atomics against addresses loaded through
the zero offset array. These are distinct coverage cases.

CAS starts at zero. One invocation wins with `id + 1`, or
`0x13579bdf00000000 | (id + 1)` for 64 bits. Exactly that invocation must report
zero; every other invocation must report the final winning token. Any winner is
valid. Checking the high-word token prevents a truncated 32-bit result from
satisfying the 64-bit oracle. The fixtures request relaxed operations; this
diagnostic makes no claim about system coherence or other order/scope modes.

All payloads have 64-byte poison prefixes and the complete rounded suffix is
checked, including the unused tail of each allocation. Submission errors, kick
errors, timeouts and exceptions after uncertain submission quarantine the queue
without retries or local teardown. Successfully completed dispatches validate
all results and close their resources before the next dispatch.

Each dispatch logs its ordered `PS5HIP DISPATCH` and `PS5HIP SUBMIT` lines, then
`PS5HIP PASS kernel=<entry> exact_values=16449 guards=unchanged
offsets=absent|unchanged allocations=closed`. The suite logs a final pass only
after all six rounds, with `dispatches=6 exact_values=98694`. These counts include
each final counter and all observations; immutable offset checks are additional.

## Offline validation

The four genuine LLVM 18 O1/O2 objects for gfx1013/gfx1030 are consumed read-only
from `build/atomic-probes`, bound to the main repository's
`evidence/offline-atomic-contention-probe.json`. No device recompilation or
source-level atomic substitute is used by the harness.

The host model executes the actual harness, parser, materializer, packet encoder
and submission queue. A synthetic callback independently supplies legal and
corrupt observations; it is not an instruction emulator. Synthetic GPU addresses
map to sanitizer-covered host storage, replacing native allocation only for
these tests. The model checks exact entry selection, argument routing, dimensions,
module contents, initial counters/poison/zero offsets and completion packets.

Each object exercises 12 valid six-dispatch suites, including reordered tickets
and different CAS winners, and 77 corruption cases across all rounds: wrong final
counter, duplicate/lost tickets, untouched output, 64-bit truncation, bogus CAS
observations/winner identity, all buffer boundaries and immutable offsets.
Thirty failure cases per object cover all six dispatch positions with submit,
kick, timeout, standard-exception and unknown-exception injection. Quarantine
assertions require all earlier allocations gone, every current allocation held
by the poisoned queue, and no attempted close while ownership is unresolved.

The integrated ASan/UBSan suite passed 28 tests, including the four actual-object
models and existing arithmetic/core regressions. The actual native harness also
compiled through the pinned PS5 SDK for both target macros with warnings treated
as errors. Those relocatable objects do not prove native linking, startup,
submission, atomic execution or teardown. Raw commands, logs and a source-bound
receipt live under ignored `results/atomic-host`, `results/atomic-sdk` and
`results/atomic-readiness`.
