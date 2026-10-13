# Main-initialized joined-worker title

`hip_workers_main` uses the unchanged genuine `tests/hip_threads.hip` compiler
object and the existing reusable runtime. Main initializes AGC with a warm-up;
two joined workers then issue ordered calls, followed by one main-thread call.
The measured compiler receipt is `f6c57047` (workers-compiler-v2), independently
reviewed in `6fbe3ee8`. It exports `threads_launch` with C linkage and registers
one `thread_words` kernel, a 24-byte wrapper and 272-byte kernargs. Device bytes
match the earlier threaded host proof. This is an offline identity, not GPU proof.

Five dispatches check 661 active words and all 1,355 device words. Another 1,355
comparisons cover host templates; they are not GPU outputs. Seventeen ordered
host checks exercise independent configuration stacks and per-thread errors.
The wrapper requires a successful client, complete cleanup, no unsafe-teardown
latch, two created/joined threads and all exact counts before calling finish.
Any failed join quarantines without runtime/static destruction or shell request.

Experiment-only link wrappers forward the exact SDK AGC init/submit/kick calls
once, recording fixed POD events under the runtime gate. The accepted actor order
is init/main, submit+kick/main, A, B, A, main. The builder verifies linked backend
redirection and forwarding to the original import relocations. These observations
qualify only this ordered cross-thread experiment; no concurrent submissions,
worker allocations, worker initialization, streams or broad ROCm claim follows.

The profile is integrated into the existing builder, staging, case verification
and exact transcript checker. Preserve the existing profiles and runtime bytes.
Native build, package, imports, startup and final lifecycle require separate
review after source freeze. The prospective external observation bound is 35s;
it does not turn driver calls or pthread joins into bounded operations.
