# Private-aperture observation diagnostic

This isolated profile reads `SRC_PRIVATE_BASE` and `SRC_PRIVATE_LIMIT` using
64-bit scalar moves. It writes six words per invocation: base low/high, limit
low/high, local ID, and local ID XOR `0x6b18d3a5`. One workgroup has 64 invocations
in two wave32 waves. Its genuine LLVM18 objects require six user SGPRs, one output
pointer, zero private storage, zero LDS and no hidden arguments. They contain no
private-memory access, flat-scratch initialization or device calls.

The profile uses the unchanged initial launch, submission queue, allocation
ownership and Eden elevation paths from frozen arithmetic source `5b74ac8`.
It retains every allocation until completion and quarantines unresolved GPU
ownership without retry or teardown. The shared guarded experiment wrapper
preserves the existing main function's exception handling. Core launch, parser,
memory, submission, AGC, arithmetic kernel and import files remain unchanged.

The native harness scans the entire rounded output allocation outside the
384-word payload. It checks all 64 local IDs and sentinels, then requires the
four raw register words to agree within each wave. Values may differ between
waves. Zero, descending base/limit pairs, high-bit values and `0xa5` values are
accepted as observations; no range or routing validity is inferred.

Each raw payload word starts with a distinct per-lane/per-word poison. This
detects omitted columns and half-wave writes through inconsistent observations.
A single omitted store whose true value happens to equal that word's initial
poison is intrinsically indistinguishable. This bounded transport check cannot
prove every individual store independently for all possible register values.

The log contains the submit line, 64 ordered records with full 16-digit base and
limit values, a within-wave consistency/across-wave equality summary, and an
allocation-closure result. Raw records are emitted before consistency checking
so failed observations remain available. Success ends with:

```text
PS5HIP PASS kernel=private_aperture exact_values=128 raw_words=256 guards=unchanged allocations=closed
```

The 128 exact values are IDs and sentinels. The other 256 words are raw
observations. The strict log parser checks ordering, witnesses, consistency and
the truth of the equality summary; an application pass is not a hardware gate
pass. Builder, stager and local verifier bind the fixed layout and O2 device
profile. Neither output dimensions nor aperture thresholds are configurable.

`tools/probe_aperture.py` records actual compiler/linker and implementation-library
identities, driver traces, IR, ISA, descriptors and object hashes for O1/O2 on
gfx1013/gfx1030. Its exact pinned ISA contract checks the 24-byte per-invocation
store layout; `s_code_end` is allowed only as trailing padding. It refuses an
existing output directory before writing. The selected native profile uses O2.

Offline validation chronology:

- All 29 integrated CTests passed with ASan/UBSan, including existing arithmetic,
  lifecycle, Python staging/identity/log and compiler contract tests.
- Two test-only clean cases for all-zero and all-`0xa5` observations were added;
  all four affected actual-object model tests then passed.
- Independent review found an ISA-padding audit gap. After fixing it and adding
  mid-body padding negatives, all five affected compiler/object tests passed.
  Regenerated device objects remained byte-identical; prior receipts were kept.
- Each of the four real objects feeds the actual host harness, parser, packet
  encoder and queue with synthetic GPU addresses. Tests cover four clean cases,
  17 corruptions, and five error/exception quarantine cases. Quarantine checks
  require all five allocations retained with GPU ownership and no close attempt.
- The actual native harness compiled to PS5 SDK relocatable x86-64 objects for
  both target macros with elevation and shell-exit definitions and warnings as
  errors. These are object-only checks; no final native link or import validation
  was performed. Earlier unused-profile warnings and their fix remain recorded.

The host model's emitted observation records also passed the independent log
checker. Its startup/runtime prefix was explicitly synthetic; native title main
and GPU execution were not simulated as proof. Raw checks and hashes are under
ignored `results/aperture-offline-readiness` and the referenced result folders.

No native package has been created and no console has run this profile. No
private-aperture routing, flat addressing, scratch backing, SPI wave offsets,
private storage, nested calls or G3 completion follows from these offline tests.
Dependency links are read-only; elevation/native export inputs must be localized
before a separately reviewed full native build and staging step.
