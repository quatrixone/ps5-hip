# Aggregate kernel arguments

The LLM compiler probe emits 12-byte vector, 16-byte struct and 128-byte
softmax arguments. The public runtime now captures and packs these by-value
arguments through genuine HIP compiler stubs and CLR. This is an offline
qualification; native aggregate execution and LLM inference remain unqualified.

## Contract

Scalar 4/8-byte arguments retain their existing representation. By-value
arguments from 9 through 128 bytes own an exact byte snapshot, with no scalar
bits or buffer extent attached. Compiler metadata offsets determine placement;
aggregate byte counts are not alignments. Layout validation precedes host reads,
and representation validation precedes native temporary allocation. Gaps and
explicit-only storage rounded to four bytes remain zero.

CLR retains the compiler offsets and copies the full argument value. The host
model mutates caller storage after enqueue and verifies the original snapshot.
C++ internal struct padding is copied but is not a portable member value;
checks distinguish it from the required zero gaps between kernel arguments.

Metadata does not reveal pointers inside structs. Such pointers are opaque:
their pointees are not automatically discovered or retained. Callers must keep
them alive through completion. The real pointer-bearing fixture demonstrates
this metadata ambiguity; nested-pointer lifetime is not qualified. The selected
softmax aggregate has scalar fields only. Larger aggregates remain unsupported.

## Evidence and reproduction

See `../evidence/offline-public-aggregate-arguments.json` for source and receipt
hashes. The qualification includes:

- Genuine Linux and Prospero compiler objects, gfx1013 metadata, host stubs and
  compiler/tool/header identities.
- Full 87-compilation public CLR ASan/UBSan regression qualification.
- Genuine CLR eager/deferred runs with two changed aggregate captures, rejected
  geometry/shared-memory calls followed by valid launches, zero gaps, caller
  mutation, guards and clean modeled ownership.
- No-op and corrupt transport negative controls that fail the byte oracle.
- Focused core argument tests and 32 configured core sanitizer regressions.
- Five compiler-receipt admission tests for missing identities/manifests,
  stale sources, corrupt artifacts, path confinement and false GPU claims.

From the documented WSL/Linux build environment, use fresh output directories:

```sh
make public-hip-aggregate-host-test \
  QUALIFICATION_DIR=results/aggregate-base-new \
  AGGREGATE_FIXTURE_DIR=results/aggregate-compiler-new \
  AGGREGATE_QUALIFICATION_DIR=results/aggregate-host-new
```

This builds the base runtime qualification and compiler fixture before linking
the aggregate model. Existing outputs are not overwritten. The qualifier checks
source/tool identities and hashes reused core objects before and after the run;
the base receipt does not independently bind their original object bytes.

The first model attempt incorrectly assumed a null output kernel pointer must
fail synchronously. That run failed before valid execution and is preserved at
`results/public-hip-aggregate-v1`. Kernel pointers can be queued; the corrected
test uses explicit unsupported geometry/shared-memory admission errors. The
passing receipt was v2; the final repeat using Git-normalized source bytes is
`results/public-hip-aggregate-v3/receipt.json`.

## Next native gate

Integrate the aggregate title with the native builder, exact compiler-object
linking, image audit and fresh console controls. Review emitted ISA and argument
layout, commit/freeze the candidate, then test on idle .40. The draft native
title is not a qualified package. No console state changed in this milestone.
Geometry/LDS/private-memory exposure, model-scale memory, math/library linkage
and complete native tensor/client execution remain separate open requirements.
