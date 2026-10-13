# Native memory-fill candidate

`hip_memory_fill` uses the existing `affine_words` device body in `hip_generic.hip`.
The new host variant first checks the single-device discovery APIs before backend
initialization, then performs six genuine HIP launches, two rounds each at 1, 73,
and 257 elements. Each round uses four hipMemset calls: full input poison, active
input bytes, an unaligned three-byte interior patch, and full output poison.
Values exercise byte truncation (0x123 and 0x106) and negative -1. The second round
changes input bytes on the same allocations. No host-to-device copy masks a fill.

The host checks all 746 input words and 746 output words, including seven guard
words at each end, against independently constructed unsigned integer arrays.
The GPU computation consumes all 662 active filled input words. The existing
unused block_sum registration is retained; its device body is not duplicated.

Offline reproduction, from the repository root with the reference root supplied:

```
python3 tools/probe_hip_generic.py --reference-root ROOT --compiler-only --title-entry --memory-fill --output results/fill-compiler
python3 tests/hip_generic_title_test.py --memory-fill --reference-root ROOT --probe-dir results/fill-compiler --output results/fill-title
python3 tests/hip_generic_title_profile_test.py --memory-fill --compiler results/fill-compiler
```

The title models link the actual runtime and actual generated HIP constructors,
stubs and destructors. Synthetic transport consumes actual backing bytes; it is
not AMD ISA emulation. No-op, stale-second-round, wrong-byte and shifted-patch
faults must fail the host oracle. Submission uncertainty retains ownership;
failed lifecycle closure cannot request shell exit. The exact ordered native-log
checker also rejects deleted or duplicated lines.

The compiler receipt binds the separate host variant while comparing device
code with the existing generic authority. Native build admission uses the frozen
memory-fill profile and current compiler dependencies. Offline evidence does not
establish console execution, cache visibility or hardware correctness.
