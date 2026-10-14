# Public HIP allocation bridge (offline)

This candidate combines the genuine upstream HIP stream/event admission with
the separately qualified ROCclr SVM allocation patch. The added model calls
the exported `hipMalloc` and `hipFree` entry points, checks backing allocation
and retirement through the PS5 device model, and rejects an interior-pointer
free and a second free. The same ten-case closure retains the earlier stream,
event, reset-rejection and failed-join checks.

`tools/compose_hip_memory.py` verifies the exact admission source and both
follow-on patch hashes before producing `results/source-v3`. The qualifier
binds the composed source, compiler-discovered headers, tools and link inputs.
Its full ASan/UBSan receipt is under `results/asan-memory-v4`. The 69-unit
TSan closure compiled and linked, but its intentional failed-join case printed
the expected final ownership line and then timed out during process exit.
The previously qualified TSan binary also stalls at that point under the
current WSL environment. A separate receipt under
`results/tsan-memory-targeted-v1` binds a passing TSan run of the new public
allocation case; full TSan qualification remains inconclusive.

The transport is synthetic and this is a Linux model. It does not prove native
public HIP startup, GPU execution, module loading, launch, copied input
consumption, graphics interop, or G5 acceptance. The SVM allocation profile
remains one device, zero flags and at most 64 MiB.
