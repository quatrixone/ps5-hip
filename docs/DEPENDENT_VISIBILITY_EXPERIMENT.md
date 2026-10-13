# Persistent-buffer visibility experiment

The frozen implementation is `a0498530cb1a2d9d6d0db6a62d6f0d4e8df92e3e`
on `development/g3-dependent-visibility`. Its native image is recorded in
`../evidence/offline-dependent-visibility-readiness.json`. The source, 54 host
tests, native build and package have passed offline review. The hardware result
was subsequently recovered and reviewed; the original bounded cycle is inconclusive.

## Hypothesis and exact case

A completed producer dispatch can write a persistent intermediate buffer that a
subsequent consumer reads correctly, across three rounds of changing CPU input.
The same input, intermediate and output allocations survive all six dispatches.
The CPU never rewrites intermediate data after its initial allocation.

Each kernel launches five groups of 64 threads and handles 257 elements. For round
`r = 0..2` and index `i = 0..256`, all arithmetic is unsigned modulo 2^32:

```
input(r, i) = (0xfedcba98 + r * 0x31415927) XOR (i * 0x1020305)
producer(r, i) = input(r, i) * 17 + i * 31 + 0x13579bdf
output(r, i) = (producer(r, i) XOR 0xa39d71b5)
               + 3 * producer(r, (i + 113) % 257)
```

The host derives expected results from round and index, independently of GPU
output. Every round checks all inputs, intermediate values, final values and full
rounded allocation guards. Final success requires closing all allocations.

## Hardware acceptance

Use title PPSA99014 on idle .40, with both shared locks and current service checks.
Preserve the successful mixed LDS image and its 616-byte result. Deployment must
verify the exact image and mounted executable/object before any application launch.
The reviewed controls live in `results/visibility-deployment-v1` and
`results/visibility-execution-v1`; their receipts retain raw artifact identities.

One run must produce six ordered submission lines, three exact round results and
771 validated final values, followed by complete finalization and natural title
shutdown. Require the original bounded lifecycle, unchanged visible mount identity
multiset, healthy services and released owned locks. GPU waits are 2,000 ms each;
the host observation bound remains 20 seconds. Missing evidence, numerical or guard
mismatch, uncertain GPU ownership or timeout rejects. Preserve evidence and analyze
offline before deciding any further experiment.

This tests completed submissions with the existing CPU cache maintenance path.
It does not establish asynchronous dispatch ordering, GPU-only publication,
release/acquire atomics or full G3 completion. After an accepted audited result,
continue wave, atomic, numerical and multidimensional geometry qualification.

## First observation

PID168/application `0x18` launched once and stopped. Eight resource-manager log
lines interleaved between its title and PID spawn records, so the original strict
adjacency parser rejected the lifecycle and observation exhausted its deadline.
That original receipt remains inconclusive; no GPU retry was performed.

A separate locked, read-only recovery verified the installed bytes and fetched
the same 1,300-byte result twice. All six submissions, three exact rounds, guards,
allocation closure and finalization passed. Independent recovery review verified
the raw evidence and uninterrupted saved application history across the gap.
This is recovered computation evidence, not continuous observation: locks were
released between captures. A subsequent delayed snapshot matched the original
visible mount baseline, with the result and installed bytes verified unchanged
before and after. Independent audit verified the composed evidence. The snapshot
occurred 864.105950 seconds after launch; the original 20-second cycle remains
inconclusive, and the unlocked/klog gaps remain limitations.
See `../evidence/native-visibility-computation.json` and
`../evidence/native-visibility-execution-incomplete.json`.
