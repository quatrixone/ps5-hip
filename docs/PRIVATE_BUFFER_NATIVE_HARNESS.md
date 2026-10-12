# Provisional 304-byte private-call native harness

`PS5HIP_CANARY_PRIVATE_CALLS_BUFFER` selects the distinct `private_calls_buffer`
diagnostic identity while launching the actual `nested_private` entry. This is a
macro-only offline harness, based on frozen `716c8eb`; the original 288-byte
`PS5HIP_CANARY_PRIVATE_CALLS` profile remains available and tested. No builder or
stager selects the new profile. Core admission, packets, kernels and dependency
locks are unchanged.

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
objects and linked audit for testing only. Runtime/compiler provenance binding
and portable compiler acceptance are still prerequisites for candidate selection;
this macro must not be treated as an arbitrary-shader qualification API. No
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

Two genuine PS5 SDK x86-64 relocatable host objects compile with the new macro,
Eden elevation and shell-exit configuration, one per target macro. They are not
linked native executables. No device recompilation, package, console run or GPU
qualification occurred in this milestone. Raw commands, logs, protected-file
comparison and source/tool/object hashes are under `results/private-buffer-host`,
`results/private-buffer-sdk` and `results/private-buffer-readiness`.
