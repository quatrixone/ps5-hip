# Real upstream HIP compilation probe

This offline probe advances V1.2 from a lexical API inventory to actual compiler and linker results. It does **not** build a runnable llama.cpp HIP backend, run inference, or validate GPU mathematics. The machine used LLVM 18.1.8, authentic `gfx1013`, wave32, code object v5, and the existing HIP 6.2.4 headers. No console was contacted. See [compact evidence](../evidence/offline-llm-hip-compile-probe.json).

## Header compatibility decision remains open

The locked llama.cpp revision is `b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`. Its shared HIP vendor header includes `hipblas/hipblas.h` even for normalization and RoPE, so the repository's existing HIP/CLR headers alone cannot compile these translation units.

Official current `ROCm/rocm-libraries` snapshot `e7cd12e461ea953a9c99e9fb7852920fa94cd197` supplies hipBLAS 3.8.0 headers. These expose an actual version mismatch: llama.cpp's `ggml-cuda/vendors/hip.h` selects legacy `hipblasDatatype_t` when `HIP_VERSION < 60500000`, while the current hipBLAS header has removed that type. We did not spoof `HIP_VERSION` or the GPU architecture.

For diagnosis only, official `ROCm/hipBLAS` tag `rocm-6.2.4`, revision `f4095274efb0892a1ef3ce0435352849f721d393`, supplies matching hipBLAS 2.2.0 headers. They establish the remaining compiler and device-library requirements. This is **not** a selected project dependency, a retained fork, a BLAS implementation, or a decision to depend on old releases. Select a coherent supported HIP/hipBLAS interface before integrating the real client; evaluate current upstream versions together.

Both sets came directly from official source URLs recorded with hashes in the evidence. Export macros were generated with CMake's `GenerateExportHeader`; version headers used the upstream templates and upstream version numbers. The imported CMake library target used for macro generation did not provide or link any implementation.

## Measured outcomes

| Stage | Actual result |
| --- | --- |
| Missing hipBLAS baseline | Real norm, rope and backend translation units fail on the missing header |
| Matching-header syntax | All nine probes pass: norm/rope/backend, each Linux host, Prospero host, gfx1013 device |
| Host objects | norm, rope, softmax, getrows, mmvq and backend compile for both Linux and Prospero (12 objects) |
| Device link without libraries | All six fail with genuine OCKL/OCML unresolved symbols |
| Existing verified OCML + workitem slice | Backend links; norm/rope/getrows/mmvq require `__ockl_mul_hi_u32`; softmax requires `__ockl_grid_sync` |
| Unchanged upstream mul_hi added | getrows and mmvq link; norm/rope advance to unresolved `__oclc_daz_opt` despite supplied, hash-verified daz-off bitcode |

The successful backend device module has one kernel; getrows has 109; mmvq has 317. `llvm-nm --undefined-only` reports no remaining undefined symbols for those three linked objects. This does not establish runtime admission or reachability of every included kernel.

The mul-hi body is the genuine existing device-library `ockl/src/mul_hi.cl`, compiled using its already-generated upstream CMake rule, then processed with upstream `prepare-builtins`. No function was reimplemented. The source revision is `1e2c94795ee0d6ab8e2ff3035965a6b74e11b475`. Initial attempts with a raw bitcode path exposed the driver's library-path lookup behavior; subsequent fresh runs copied the verified bitcodes into a separate probe directory and used basenames. Raw and prepared variants both expose the norm/RoPE daz residual. Preserve that failure as a compiler/library integration issue to investigate, not a reason to suppress or fake the symbol.

## ABI and launch requirements observed

Compiler metadata exposes 12-byte by-value norm arguments and 12/16-byte RoPE arguments. Softmax includes a 128-byte aggregate parameter. These are real generated ABI requirements, beyond the current scalar 4/8-byte path. Metadata argument-size inventories also include hidden arguments; do not infer user argument semantics from size alone.

Norm kernarg segments reach 488 bytes; RoPE reaches 488; softmax reaches 416. Linked mmvq has kernarg segments of 164, 192, 424 and 448 bytes, fixed LDS up to 12,288 bytes and private segments up to 104 bytes. Its workgroup bounds vary across 64, 128 and 256, with wave32 throughout. Max-flat-workgroup bounds of 1024 in other modules are compiler bounds, not proof that every runtime launch uses 1024 threads. Dynamic shared-memory requests must be accounted for separately from fixed LDS.

Current integration constraints to address are `src/host_launch.cpp` (`validate_capture_layout`, scalar snapshots limited to eight bytes) and `ports/clr/ps5-device.cpp` (ValueObject 4/8 handling, fixed initial 64x1x1 route and current allocation limit). Extend capture, metadata validation and dispatch coherently; do not relax checks merely to admit these modules. The full converted model also exceeds a single 64 MiB allocation.

## Compiled imports versus required execution paths

The actual Prospero backend object imports seven hipBLAS entry points: Create, Destroy, SetStream, Sgemm, GemmEx, GemmBatchedEx and GemmStridedBatchedEx. Their implementations and a real final host link are still missing. Its HIP imports include registration/launch, device properties, allocation, copies, events and streams, plus managed memory, host registration and peer APIs. The evidence records the exact symbol list. This is compiled closure, not a trace proving that TinyLlama executes every API.

Upstream already gates managed memory behind `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, host registration behind `GGML_CUDA_REGISTER_HOST`, and peer paths behind device/configuration conditions. Keep unsupported optional capabilities honest; link completeness is distinct from capability support.

The cooperative softmax kernel is compiled into the TU and references `__ockl_grid_sync`. Its launch branch requires cooperative capability and more than 8192 columns per row, no mask/sinks, and further conditions. Ordinary TinyLlama attention with context <=512 takes the ordinary softmax path when its small shared-memory request is supported. That does not remove the unused cooperative kernel's link requirement. Either supply genuine cooperative ABI support or implement a justified upstream-compatible build/admission strategy; never return fake cooperative success. The existing upstream `ockl/src/cg.cl` expects cooperative grid information/GWS semantics and cannot safely be treated as an ordinary workgroup barrier.

## Reproducing the bounded probe

Use the source-only prerequisites and locked client from [SOURCE_ONLY_REBUILD.md](SOURCE_ONLY_REBUILD.md). Obtain the exact official headers listed in the evidence, generate their export/version headers with upstream values, and keep the two header roots separate. The diagnostic header comparison is optional research tooling, not a new Make dependency.

The representative command shape is:

```sh
clang++-18 -x hip -nogpuinc -nogpulib --offload-arch=gfx1013 \
  -mcode-object-version=5 -mno-wavefrontsize64 -D__HIP_PLATFORM_AMD__ \
  -std=c++17 -DNDEBUG -O2 -fPIC -DGGML_USE_HIP -DGGML_HIP_NO_VMM \
  -DGGML_HIP_NO_MMQ_MFMA -DGGML_CUDA_NO_FA \
  -I.deps/sources/hip-headers/include -I.deps/sources/clr/hipamd/include \
  -Iports/hip -I.deps/sources/llama-client/ggml/include \
  -I.deps/sources/llama-client/ggml/src -I<diagnostic-header-root> \
  -include __clang_hip_runtime_wrapper.h --cuda-device-only -c \
  .deps/sources/llama-client/ggml/src/ggml-cuda/norm.cu -o norm.o
```

For syntax use `-fsyntax-only` **and** `-c`; for host objects use `--cuda-host-only`. Prospero host uses `tools/ps5-clangxx`, `-nostdlib`, and `-Xarch_device -fno-short-wchar`. Omitting `-c` under the SDK wrapper can inject `crt1.o` into a global `-x hip` syntax command; that is a probe invocation error. Release `NDEBUG` avoids debug assertions routing device code to SDK host `__assert`.

For library-enabled links, remove `-nogpulib` and `-c`, add `--no-gpu-bundle-output -B/usr/lib/llvm-18/bin`, `--rocm-device-lib-path=<verified-libs>` and explicit `--hip-device-lib=<basename>` flags for the nine gfx1013 libraries in the evidence. Add the prepared upstream mul-hi bitcode for the extension. Inspect linked objects with `llvm-readobj-18 --notes` and `llvm-nm-18 --undefined-only`. The existing [OCML probe](OCML_MATH_PROBE.md) documents the bitcode provenance. These old baseline bitcodes are used to measure link closure; their use here makes no new numerical qualification claim.

Local full argv, diagnostics, dependency hashes and objects are in `results/llm-hip-probe-v2` through `v13`; normalized command copies and notes are in `results/llm-hip-probe-report-v1`. These ignored outputs are not required source inputs. Evidence contains their hashes and outcomes. The assembly-only mmvq experiment hit a 180-second timeout; its separately linked HSACO supplied metadata without repeating the expensive assembly step. Earlier failed invocation variants are preserved and not counted as passes.

Next work: resolve the header-version integration decision and daz constant link issue; turn the genuine mul-hi slice into reproducible build input; select/validate real operator modules with aggregate ABI, LDS/private memory and launch geometry; supply the needed BLAS/runtime implementations; then perform full client link and compare native logits against the frozen CPU fixtures. No full SDK is necessary for this sequence.
