# Native two-module allocation-query title

`hip_multi_query` is a bounded PPSA99014 experiment, not general framework support.
Two independent genuine HIP compiler objects register `multi_a` and `multi_b`.
The plain host client dispatches A/B/A with changed inputs, persistent output
poison and seven-word guards. It checks 219 active outputs, 522 readback words
and 84 guard comparisons across two 348-byte logical allocations.

At allocation and after each launch, both metadata APIs inspect both allocation
bases and kernel interior pointers: 32 successful queries. Eight one-past/freed
pointer queries must reject without changing output sentinels; the client consumes
each error. Their exact eight diagnostic lines are part of the native log contract.
Metadata queries are not pointer dereferences or physical-memory capacity reports.

The emitted kernels have 20 bytes of explicit arguments and a **280-byte complete
kernarg segment**, including compiler-hidden fields. Each Linux/PS5 device object
is inspected separately. Host models compare the complete selected owner image
and entry before synthetic transport computes results; this is not ISA emulation.

Native linking selects exactly two bundles and two 24-byte wrappers. The 64-byte
table contains relative wrapper/bundle bounds. Admission requires read-only LOAD
placement with no writable 16-KiB page overlap in raw and converted ELF, exact
relative wrapper relocations, distinct constructor addresses, own host tokens,
distinct retained handles and their own atexit destructor targets. The full updated
runtime, SDK providers, CRT and helper must link; diagnostic unresolved ELF links
cannot satisfy this gate.

The title finishes only after successful results and complete allocation cleanup.
Transport uncertainty retains backing; missing genuine retirement prevents normal
finalization. Both genuine callbacks must retire active modules from two to one to
zero before the final state marker and shell request. Existing profiles keep their
single-module admission and lifecycle checks.

Offline client proofs cover AB and BA object order, skipped B/final-A launches,
wrong owner, source/output/guard corruption, incorrect query outputs, allocation
failures, free failure and exact terminal retention. Title models additionally
exercise persistence, registration, retirement and after-finish rejection. Console
acceptance remains a separate, frozen-package observation; no hardware result is
implied by these source and host-model checks.

Saved title acceptance uses `hip_multi_query_title_receipt_test.py`: it verifies
the exact consumed inventories and command sequence before replaying the original
producer's transcript checks. The preserved producer receipt and binaries remain
unchanged; the new checker independently rejects every inventory omission.
