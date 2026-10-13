# Ordinary CUDA translation client

`tests/cuda_vector_add.cu` is a self-checking CUDA example with normal launch
syntax, allocation/copy/free and error handling. It covers unsigned wrapping
addition at lengths 1, 73 and 257, with two changed-input rounds per length,
unchanged inputs and complete logical guards. It contains no PS5 runtime calls.

The pinned official [HIPIFY ROCm6.2.4 source](https://github.com/ROCm/HIPIFY/tree/0776ed88a0386fea6a10c991485083bb6b8b3433)
provides `hipify-perl`. The recipe records its revision and script hash, runs it
without in-place changes, and requires that the output differs only by its
inserted HIP header, runtime-header substitution and ten named CUDA/HIP API
substitutions. The generated duplicate HIP include is retained as produced.
Kernel bodies, launch expressions and numerical checks receive no hand edits.

```
python3 -B tests/hipify_recipe_test.py
python3 -B tools/probe_hipify_client.py --reference-root <main> --hipify <pinned-checkout> --output results/compiler-new
```

The compiler proof uses the existing pinned LLVM18/ROCm6.2.4 header and workitem
library slice for gfx1013 wave32. Linux, PS5 and instrumented Linux host objects
are independently inspected for genuine compiler registration, launch stubs,
unregister and code-object metadata. No CUDA runtime imports survive translation.
The generated input is compiled as HIP with the same target/ABI flags as other
clients; host-side ASan flags apply only to the instrumented profile.

This evidence establishes translation and compilation, not CUDA correctness,
runtime execution, native linking, GPU results or complete G4 acceptance.
The example has not run under CUDA. Runtime ownership models and a native title
with the existing checked lifecycle are still required; ordinary source remains
separate from that platform wrapper. No general CUDA compatibility is implied.
