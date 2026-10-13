# Native compiler-spill experiment

Status: exact spill fixture passed one bounded .40 cycle. All 25 execution and eight
deployment regression methods pass. Deployment/preflight and saved execution evidence
have been independently reviewed. A content filter initially prevented the post-run
audit; it subsequently completed, confirming the original parent audit. Full G3 remains open.

## Question and frozen candidate

Does baseline LLVM 18.1.8 O2 gfx1013 wave32 code preserve values through actual
compiler-generated buffer scratch spills on the PS5? This is separate from the
successful experimental private304 nested-call fixture.

- Source: `1e0428f1d4d6069cc813f8a280a31460b3e37d63`.
- Stage: `.local/g3-scratch/dist/PPSA99014/gfx1013-register_spill-8ce90d0bc1a6`.
- Image SHA256: `0b081edfb997e8b54320ff01dab9c69010de2df38340497ac79af5c39405da57`.
- Device SHA256: `355d788004f1bffb00456c2de4cd75162551281b3280cf9171a26e8021caf575`.
- Prior build/package evidence: `../evidence/offline-scratch-native-readiness.json`.

The kernel loads 64 volatile input words per invocation before storing any output,
with a 16-VGPR compiler constraint. The saved ISA review identifies 52 buffer spill
stores and 52 reloads, 212 private bytes per lane and no flat-scratch setup. Six
user SGPRs carry the scratch descriptor and kernarg pointer; s7 supplies the private
wave offset. One 64-thread group produces two wave32 waves. The ring has 1,280
slots, a 7,168-byte wave stride and 9,175,040 bytes; TMPRING is `0x7500`.

## Oracle and rejection

For each word index 0 through 4,095, require output and unchanged input to equal
`uint32(index * 0x9e3779b9) ^ 0x6c8e9cf5`. Require modified scratch, unchanged full
data/scratch guards, successful allocation closure, runtime checks and every
finalization checkpoint. The frozen checker requires 64 invocations, 4,096 output
words, one group, 212 private bytes and the exact ring size. It does not print a
user-SGPR count; source/object identity binds that ABI separately.

Queue timeout remains 2,000 ms. Submission uncertainty quarantines resources;
never turn a timeout into success or automatically retry a suspected GPU fault.
Any mismatch, missing checkpoint, guard change, identity change, unexpected title,
incomplete lifecycle, unhealthy service or failed evidence collection rejects the
bounded cycle. Reassess saved evidence offline before a new attempt.

## Deployment and observation

Before enabling controls, run their offline regressions, recheck Git HEAD/clean
frozen trees, inspect the final manifests and obtain independent review. Current
Windows byte revalidation matched 43 source/staged/image files, but does not replace
those checks. Approval authentication recovered and execution regressions pass in
WSL; final executable local verification and deployment review remain pending.

Under both owned workspace locks and fresh idle checks on .40, preserve the installed
private304 image and exact 642-byte result (SHA256
`c66fd5253bb19734d7398b18f2631147563effab8de59c35b621db1b985c3833`). Wait for natural
old-image unmount before promotion. Verify stored and mounted candidate bytes using
acknowledged connection-local raw SELF transfer, keeping exact signed-file bounds.
Do not use payloads to force unmount, change server settings or close foreign titles.

Bind execution to the actual independently reviewed deployment receipt. Require an
absent spill log, fresh complete positive mount baseline, one launch and unique
matching PID/application identity. Within the existing 20-second host window, wait
for natural self-stop, ordered OS teardown and strict shell idle. The focus-lag
correction waits inside this deadline; it does not weaken close authorization.
Collect the exact result, perform exact-title cleanup, compare complete pre/post
visible mount identity multisets, check services and release only owned locks.
Each phase is single-use. Preserve all failed attempts and consumed controls.

## Meaning and next step

A pass establishes this exact O2 spill fixture. Two waves do not exercise ring
wrap, broad launch geometry or repeated scratch ownership. Other optimization
levels, generic private pointers, mixed scratch/LDS and general G3 remain open.
Visible mount equality does not prove global namespace or reference release.

After a pass, qualify static LDS, dynamic LDS and mixed LDS with cross-wave barriers,
then broaden the remaining G3 families in the implementation plan. The accepted
private304 computation and this exact spill result cannot substitute for them.

## Observed result

PID132/application `0x8018` produced 4,096 exact outputs, modified scratch, unchanged
guards, closed allocations and all finalization checkpoints, then stopped naturally.
The original bounded cycle completed, with equal visible pre/post mount identity
multisets, healthy services and released locks. No delayed supplemental capture was
needed. Parent audit rehashed 25 raw files and reproduced the frozen acceptance
checks with networking blocked. A subsequent independent audit confirmed those results,
including natural shutdown within the original bounded observation window.
See `../evidence/native-register-spill.json`. Installed spill image and result must
be preserved before the next candidate replaces them.
