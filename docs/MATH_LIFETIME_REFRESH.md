# Native math lifetime refresh

This isolated refresh starts at `ef9f296`. The three native memory/submission
lifetime sources and focused native-memory test match reviewed atomic refresh
`90df59d`. Submission tests retain the math branch's repeated-launch and late
marker cases and add typed completed cleanup, partial marker allocation unwind
and pre-submit flush/cleanup distinctions. Device math, numerical oracles,
tolerances and compiler fixtures remain unchanged.

The first preparation script stopped before writing `submission_test.cpp` because
its generic cleanup-rejection pattern was not unique. This was a preparation
assertion failure, not a compiled candidate failure. Its source and diagnostic
are preserved under ignored results; a targeted context patch completes the test
without discarding the repeated-launch cases.

Qualification requires the original 45 isolated sanitizer checks with pinned OCML fixtures
and native relocatable compilation. A new linked/signed/staged package and its
independent audit are separate later steps. No new numerical or general HIP
capability follows from this lifetime-only refresh or host callback arithmetic.
