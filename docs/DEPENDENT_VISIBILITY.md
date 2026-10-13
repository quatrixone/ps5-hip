# Completed-queue dependent visibility fixture

This is an offline native candidate, not hardware qualification. Base0444fede89d82295d285115e85cf50abb25f8b3c includes the current typed cleanup and memory ownership implementation. No loader, packet encoder, memory or queue code changes are needed for this experiment.

Two genuine HIP kernels execute in producer then consumer order, for three rounds. The same guarded257-word input, intermediate and output allocations, code image and kernarg allocation survive all six submissions. Five64-thread groups per kernel leave63 inactive invocations; every rounded backing word outside the257 logical words is checked. CPU writes a different input pattern and output poison only after the prior round completes. It never writes the intermediate after initial allocation.

For roundr=0..2 and indexi=0..256, input=(0xfedcba98+r*0x31415927) XOR (i*0x1020305); producer=input*17+i*31+0x13579bdf; consumer=(producer[i] XOR0xa39d71b5)+3*producer[(i+113)%257]. All arithmetic is modulo2^32. The host independently validates input immutability, intermediate and final results, and full guards after each round. Only final allocation closure produces the suite PASS line.

The producer submission completes via the existing compute-drain/cache-release marker before the consumer is submitted. Existing CPU cache operations still occur at both submission boundaries. Thus a future pass would establish this conservative completed-queue path, not autonomous GPU synchronization, dispatch overlap, release/acquire atomics or absence of CPU cache mediation.

The host model decodes actual packets from all four gfx1013/gfx1030 O1/O2 objects. It computes callback outputs independently, checks persistent allocation IDs and exact source/destination pairs, changes CPU inputs across rounds, rejects omitted/wrong producers, wrong consumer source, stale later-round data, input/intermediate/output and guard corruption. Thirty child cases cover submit/kick/timeout/typed/unknown exceptions at each dispatch and retain all seven in-flight allocations without retry or teardown. Models do not emulate GPU ISA or prove cache coherence.

The standalone compiler probe refuses existing output paths, checks pinned LLVM18 versions/linker, retains ISA/metadata/driver outputs, and requires real global loads/stores, wave32,16-byte explicit kernargs and zero private/LDS. The checked input lock pins the exact source/recipe/report and four objects. Build, stage, verification and strict log grammar carry the fixed three-round/six-dispatch profile.

Offline commands:

```
python3 tools/probe_visibility_compiler.py --output build/visibility-probes
cmake -S . -B build/host-visibility -DCMAKE_CXX_COMPILER=clang++-18 -DPS5HIP_SANITIZE=ON
cmake --build build/host-visibility -j4
ctest --test-dir build/host-visibility --output-on-failure
python3 tools/build_native_canary.py --native-source ../pinned-native-source --target gfx1013 --kernel dependent_visibility --visibility-probes build/visibility-probes --shell-exit-probe
```

The checked-in lock binds the already preserved compiler output. Regeneration into a different path changes driver/report provenance and needs a deliberate reviewed lock update, even when object bytes match. Mutable native-tool exports and elevation sources are private to this worktree. Hardware execution requires a separate frozen-candidate experiment and current lifecycle controls; no console action is part of this recipe.
