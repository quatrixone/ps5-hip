# Provisional 304-byte private-call native harness

`PS5HIP_CANARY_PRIVATE_CALLS_BUFFER` selects the distinct `private_calls_buffer`
diagnostic identity while launching the actual `nested_private` entry. This is a
offline harness, based on frozen `716c8eb`; the original 288-byte
`PS5HIP_CANARY_PRIVATE_CALLS` profile remains available and tested. Checkpoint
`962b841` freezes the initial macro-only implementation. Subsequent experimental
builder selection requires `--private-buffer-compiler` and the exact portable
reproduction receipt pinned by `experimental/private-buffer-profile.json`.
It copies the selected O2 object; it never recompiles that shader with installed
Clang. Core admission, packets, kernels and existing dependency locks are unchanged.

The four saved ordinary gfx1013/gfx1030 O1/O2 objects from the guarded compiler's
IPRA-off validation have cumulative private storage 304 bytes: root 144, middle
160, leaf zero. Per lane, the root outer array occupies bytes 16–143; the middle
saves occupy 144–151, its inner array 160–287, and its WWM save 288–291. Remaining
bytes are reserved or padding. The allocation policy rounds 304×32 to a 10,240-byte
wave stride, retaining 1,280 slots: 13,107,200 payload bytes and TMPRING `0xa500`.
The 64 KiB prefix and suffix guards surround the entire scratch payload; guard
validation includes any extra rounded allocation extent. The model's wave-slot
assignment verifies arithmetic, not the hardware's scheduling or SPI offset.

The unchanged structural eight-SGPR route supplies the buffer descriptor in s0:3,
kernargs in s4:5, and two raw address words in s6:7. Actual selected reachable
compiler graphs use buffer scratch and omit flat-scratch initialization. Metadata
alone does not prove this property: generic, volatile or proof-disabled objects
can also report 304 bytes. The raw readiness receipt pins the exact accepted
objects and linked audit for testing only. The connected helper checks all portable
receipt/artifact hashes, the complete four-profile set and exact input/object
identities. Build, stage and verification bind the fixed contract and helper/lock
sources. An unfrozen profile lock fails closed. This macro must not be treated as
an arbitrary-shader qualification API. No
private-aperture or generic-pointer support is claimed.

Sixty ASan/UBSan CTests passed, covering the original profiles and the new planner
and actual-harness models against all four saved objects. New tests independently
decode packet words, distinguish raw address highs from descriptor flags, check
low-word carry, wave padding and non-aliasing frame slots, and reject the old
288-byte plan/metadata. Per object the actual harness receives one successful
source-array callback and sixteen corrupted outcomes, including omitted leaf
stores, input/output changes, untouched scratch and full guard violations. Five
failure cases exercise returned submit/kick errors, timeout, and standard/unknown
exceptions; quarantine checks all seven allocations remain retained with one GPU
hold and no attempted close. These callbacks do not emulate GPU ISA. The shared
guard wrapper is the same control flow invoked by native main.

The connected tooling then passed 61 CTests, adding eight focused Python methods
for exact receipt/profile substitution, every artifact and IR binding, escaping
paths, staging and verifier contracts, required CLI binding, and strict success,
failure and quarantine logs. The portable reproduction receipt pinned by the
experimental lock reports 8 ordinary graph audits, 48 extension audits, 14 guard
controls, 4 register controls and 6 linked alias objects. Its selected object bytes
match the original host-test fixtures; portable success is compiler evidence.

Two genuine PS5 SDK x86-64 relocatable host objects compile with the new macro,
Eden elevation and shell-exit configuration, one per target macro. They are not
linked native executables. No device recompilation, package, console run or GPU
qualification occurred in this milestone. Raw commands, logs, protected-file
comparison and source/tool/object hashes are under `results/private-buffer-host`,
`results/private-buffer-sdk` and `results/private-buffer-readiness`.

## Before a hardware experiment

Hypothesis: the selected, audited buffer-only direct-call graph can execute using
the unchanged AGC eight-SGPR route and guarded scratch ring. Acceptance requires
all 16,448 exact outputs, unchanged inputs, observed scratch activity, complete
guards, a bounded 2,000 ms queue wait, and independently observed clean title
teardown and healthy services. Any mismatch, crash, timeout or quarantine rejects
the experiment. A passing application log alone does not establish teardown or
hardware qualification.

Console .40 first requires evidence of recovered idle state and a known-good
arithmetic control on its firmware, which has no qualified GPU pass in the current
evidence. Freeze and independently verify the compiler-bound native package before
any test. Generic-pointer, volatile-call and hardware qualification claims remain
false; this document authorizes no console operation.
