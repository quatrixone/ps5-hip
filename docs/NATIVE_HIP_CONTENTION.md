# Bounded native HIP caller contention

Frozen `70a604a6fd702faff7862384b8aa7a11cca3270e` passed on firmware 12.70,
console .40. Main initialized the queue, then A and B made overlapping HIP API
calls. Instrumentation observed B requesting the same runtime gate while A
owned it. Three dispatches produced 331 exact active values, 813 device-readback
and 813 separate host-template comparisons; all seven AGC events passed.
Both workers joined before cleanup, compiler unregister and State destruction.
PID336/application `0x6018` stopped naturally; independent audit accepted the
bound lifecycle, mounted bytes, service health and lock release.

The trace does not establish that B entered a kernel-level blocked wait:
preemption can occur between publishing the request and entering the real lock.
The recorded interval includes deliberate instrumentation delay and scheduling,
and is not a performance measurement. No asynchronous GPU overlap, fairness,
general concurrency or full G4 completion is claimed.

Exact source, compiler/build/package, offline qualification and actual deployment/
execution authorities are linked in [the evidence record](../evidence/native-hip-contention.json).
The experiment remains isolated pending reviewed integration.
