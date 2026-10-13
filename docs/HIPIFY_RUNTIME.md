# Unedited HIPIFY client through owned runtime (offline)

This unit preserves `tests/cuda_vector_add.cu` and the reviewed HIPIFY 6.2.4
translation byte for byte. The ordinary `int main()` is renamed with
`-Dmain=hipify_client_main`; no model hooks or error paths are inserted into the
generated HIP. Real compiler constructors bind `translated_add`, its ordinary
`<<<>>>` stub calls the reusable runtime, and its genuine destructor unregisters.

The wrapper uses the actual NativeMemory, live-data registry, launch preparation,
submission queue and generic metadata policy. Its explicit host transport model
checks the encoded packet and all 288 owned argument bytes, then computes unsigned
32-bit sums independently. This is not AMD ISA execution or a native mapping test.
The existing initial route admits the exact measured hidden geometry; no new HIP
API or launch admission is added.

The client runs lengths 1, 73 and 257 twice with changed wrapping inputs: 662
active sums and 2,238 whole-buffer word comparisons, including both unchanged
inputs and seven-word guards on both ends, across the six launches.
The model checks persistent buffer identity between each pair, one lazy backend,
six ordered completions, exact allocation closure and post-main compiler/State
destruction. Failure cases cover clean allocation/init/preflush unwind, missing,
wrong, stale output, input/guard corruption, registration rollback before/after
six completions, missing finish, uncertain queue ownership, and known-complete
cleanup failure. Terminal cases reject later operations without releasing backing
and exit the model process before destructors; they cannot report finalization.

Run from this worktree with fresh output:

```
python3 tools/probe_hipify_runtime.py --reference-root ../.. --output results/runtime-v1
python3 tests/hipify_runtime_recipe_test.py
python3 tests/hipify_runtime_recipe_test.py --check results/runtime-v1
```

Set `--reference-root` to the primary ps5-hip checkout if the relative path differs.
The recipe binds the original checkpoint, exact translated text, all three renamed
compiler objects and selected device bytes, consumed headers/libraries/tools,
source before/after, commands and nineteen model transcripts. Linux and PS5 device
whole-object hashes may differ; each must reproduce its own checkpoint exactly.
ASan host models use checked fixed shader-window mappings with
`protect_shadow_gap=0`, matching existing ownership tests. SDK/compiler support
trees and static host libraries are not fully inventoried; this is not hermetic.
No console, native title, CUDA execution or general CUDA compatibility is claimed.
