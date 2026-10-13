# Focused ROCclr thread lifetime tests

`tools/probe_clr_lifetime.py --proof <completed adapter receipt> --output <new directory>` reuses the exact bound upstream/adapter objects and builds this test against their recorded compiler flags. It validates all receipt inputs and artifacts before and after the run. It does not build a replacement scheduler.

The eleven subprocess cases exercise three semaphore allocation failures and Monitor allocation failure, thread and queue creation failure, publication at startup, joining an already-finished thread, rejecting self-join, normal queue retirement, injected join failure, and last-reference release from an actual queue completion callback. Failure injection wraps real allocation or pthread interfaces; successful calls forward to the actual implementations. A genuine stack-sizing warmup completes before queue faults are armed.

Join-error and self-release cases intentionally keep the zero-reference HostQueue, context, device, and observation alive until process exit. They check destructor and ownership counts before `_Exit`; leak detection is disabled only for those two deliberate-retention cases. Virtual-device destruction alone is not evidence of a successful host-thread join. The join-error case returns EINVAL without calling the actual join on an already-finished worker. It does not establish behavior for an operating-system failure involving an arbitrarily stuck worker.

The included adapter transport model supplies explicit synthetic backend definitions. Its renamed main is never invoked; these cases perform no GPU submission. Every subprocess has a 30-second outer timeout. Linux sanitizer results are not native PS5 lifecycle qualification.
