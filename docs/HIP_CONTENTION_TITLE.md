# Contending host HIP calls: hip_contend_main

This separate, bounded experiment keeps the production runtime, backend, launch
policy and `thread_words` kernel unchanged. It does not complete G4 or claim
general HIP/ROCm compatibility. Native validation remains required.

Main allocates two independent 271-word device buffers and initializes the queue
with a one-value warmup (salt 11). Main then poisons both buffers again. A writes
73 values with salt 101; B writes 257 with salt 1009. Each worker reads its own
buffer and verifies all 271 words, including seven leading guards, inactive words
and trailing guards. Separate host templates remain poisoned. Success requires
three dispatches, 331 exact active values, 813 device comparisons and 813 template
comparisons. Worker allocation and queue initialization are not under test here.

The experiment interposes six functions with their exact header signatures and
forwards every original call exactly once without changing its return value:
`hipLaunchKernel`, `__hipPushCallConfiguration`, `pthread_mutex_lock`, `sceAgcInit`,
`sceAgcDriverSubmitDcb`, and `sceAgcSuspendPoint`. Fixed per-actor atomic slots
record events; there is no concurrent append to a plain trace vector.

A's HIP launch wrapper identifies the real runtime gate. Once its successful
lock is held and the backend reaches submit, the submit wrapper pauses before
forwarding. B waits for this hold, then calls the genuine `threads_launch` stub.
The stub's actual push wrapper records B's request to the identical mutex while
A remains inside HIP and owns it. A releases its instrumented hold after observing
that request, or on error/deadline, and always forwards the original submission.
Seven AGC events must remain main init, main submit/kick, A submit/kick, B
submit/kick. Both workers use checked 512 KiB stacks and are joined before buffer
cleanup or title finalization. No harness mutex is held across HIP.

The bounded poll has a two-second monotonic deadline and an iteration ceiling.
It is not a scheduler or GPU watchdog. A skipped push, failed lock, missing peer,
clock or sleep error cannot pass qualification. Diagnostic failure still forwards
an already reached submission and may produce correct arithmetic; that does not
make the run pass. An unknown join retains persistent storage and enters the
title's no-runtime-query quarantine, without reading the trace or finalizing.

`CONTEND_PROOF` establishes overlapping HIP API lifetimes and a same-gate request
while owned. It explicitly reports `kernel_blocked_wait=unproven`: B can be
preempted between publishing the request and entering the real lock function.
The reported lock-call interval includes the deliberate hold and scheduling. It
is not a throughput, fairness, latency or kernel blocking measurement.

Offline evidence is separated into actual runtime/compiler ASan transport models,
CPU-only TSAN of the actual client/interposers with synthetic HIP, actual title
lifecycle/log rejection models, and a Linux link-only audit using unchanged PS5
compiler code plus production runtime/backend sources. The latter image is never
executed. The native builder must repeat its six-function disassembly/import audit
on the actual candidate; host linking is insufficient to certify native redirection.

Build after review and source freeze using `build_native_vector_title.py --kernel
hip_contend_main --workers-probe <reviewed-workers-compiler-v2>`. Stage and verify
with the same separate profile; the log is
`ps5-hip-gfx1013-hip_contend_main.log`. Existing main-initialized and worker-initialized
profiles keep their prior contracts.
