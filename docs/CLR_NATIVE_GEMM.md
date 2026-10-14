# Native ROCclr GEMM candidate

This is a bounded numerical experiment, not public HIP or G6 acceptance. It
uses the already compiled gfx1013 `ps5hip_sgemm_nn` code object through the
upstream ROCclr Program, Kernel, command and queue path and the PS5 AgcBackend.
The adapter admits `GeometryLds` only after the existing profile and native
launch validators select and check that route. The prior `Initial` route stays
available; other routes remain rejected by the adapter.

The case is row-major FP32 NN, M=9, N=11, K=13, LDA=LDB=LDC=16,
alpha=1.25 and beta=-0.5. Three genuine device buffers receive A, B and C;
one 2x2 grid of 8x8 workgroups computes C. A full C readback checks 99 active
values and 45 untouched row-padding cells. The small integer operands and
dyadic scalars make the reference results exactly representable in FP32, so
the comparison is exact. This does not claim general FP32 error bounds.

Before console use, freeze and independently review the 46-unit strict native
build, extracted code-object identity, converted package, imports, source
hashes and stage files. The hypothesis is that the proven ROCclr GPU dispatch
also supports this GeometryLds kernel with numerical output dependent on
both uploaded matrices. Expected outcome: one launch, 99 exact active values,
45 unchanged padding cells, three buffer frees, one checked queue join,
watchdog join, application finalization, natural title exit, preserved prior
package/log and matching pre/post mounts. Reject the case on any missing log,
incorrect value, guard change, timeout, terminal state, unexpected import or
cleanup failure. Use the existing 15-second in-title watchdog and a bounded
title-specific launcher/observer. Never replay a consumed or ambiguous case;
diagnose offline and prepare a fresh frozen candidate. Keep raw logs local and
commit only a compact, hash-bound reviewed result.

The native numerical test is one irregular shape. G6 still requires a
representative shape and tolerance matrix, relevant upstream library tests,
a real client, memory and time measurements, and broader API coverage.
