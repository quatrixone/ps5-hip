# HIP client requirements for G4 and G7

First-release priority (2026-10-15): follow [LLM_V1_PLAN.md](LLM_V1_PLAN.md).
The standalone upstream llama.cpp inventory baseline is
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`; its build/profile inspection is recorded
in `../evidence/llm-v1-planning-baseline.json`. Refresh the full active API/kernel
census in V1.1. The Prism-specific inventory below remains historical and must
not add its model formats or application ABI to the new release contract.

Scope update (2026-10-14): ProsperoAI integration is no longer a project
requirement. References below retain historical research provenance. Refresh
client-specific requirements against the standalone G7 client selected in
`IMPLEMENTATION_PLAN.md`; do not treat the old application ABI or model list as
acceptance gates.

Offline source inventory, 2026-10-11. This is input to the G4 runtime decision,
not an implemented API list or GPU qualification. G1-G3 remain prerequisites.

## Exact clients and provenance

- ProsperoAI reference: `45fe9bbbc8a34d7ff49f1e281a7e144321474ee1`,
  [`docs/VULKAN_RUNTIME.md`](https://github.com/blackbearreloaded/ProsperoAI/blob/45fe9bbbc8a34d7ff49f1e281a7e144321474ee1/docs/VULKAN_RUNTIME.md)
  and `tools/deps.json`. Its intended shared client is **PrismML-Eng/llama.cpp
  `e8fd209c975c80aa7fb36a8423571ff3b1b7e955`**, with `GGML_MAX_NAME=160`.
  The application presently uses Vulkan; this inventory concerns its alternative
  HIP backend, not an existing ProsperoAI HIP integration.
- Inspected local Prism source: WSL
  `/home/denis/ProsperoAI/build/prism-llama-source`.
  Its cached archive SHA-256 is
  `5cb0bf3bba4e96ae1d34bb99e213fdd39cbc405f1e0dbe031b7004e812eb733d`,
  matching the ProsperoAI lock. The five files used for the core inventory
  (`ggml/CMakeLists.txt`, `ggml/src/ggml-hip/CMakeLists.txt`, and
  `ggml/src/ggml-cuda/{vendors/hip.h,common.cuh,ggml-cuda.cu}`) were checked
  byte-for-byte against that archive and match. The additional cited
  `cpy.cu`, `out-prod.cu`, `solve_tri.cu`, `softmax.cu`, `sum.cu` and
  `ggml/src/CMakeLists.txt` also match. The extracted directory is not
  an independent Git checkout: running `git rev-parse HEAD` there reports its
  parent ProsperoAI repository, not the Prism revision.
- HIP examples and declarations: local TheRock component `rocm-systems` at
  `2f2a92736dbe3f65108539b6659d2c7289d300e8` (TheRock-applied ROCr patch),
  based on upstream `d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d`.
  Paths below are relative to its `projects/` directory. Compiler source is
  TheRock's AMD LLVM `6da731510c233517c1eeb691d3446d80e9e32525`.
  These sources describe an ABI candidate; they do not qualify newer headers
  against the selected LLVM 18 compiler or the PS5 host ABI.

## First coherent example slice

| Client | Host API and device semantics actually needed |
| --- | --- |
| Upstream `hip/docs/tools/example_codes/add_kernel.hip` | `hipMalloc`, `hipMemcpy` H2D/D2H, `hipFree`, `hipGetLastError`, `hipGetErrorString`; normal `<<<grid,block>>>`, thread/block indices, global pointers and FP32 addition. The blocking D2H copy must observe completion of the preceding default-stream kernel. No BLAS library. |
| Upstream `hip/docs/tools/example_codes/warp_size_reduction.hip` | The same allocation/copy/error slice plus `hipDeviceGetAttribute(...hipDeviceAttributeWarpSize...)` and `hipDeviceSynchronize`; dynamic shared memory from the launch's third parameter, `__syncthreads`, `__shfl_down`, `warpSize`, masked input, integer reduction and one result per workgroup. Its actual example uses 1,024 threads and 4 KiB dynamic LDS per block. |
| Planned tiled FP32 matmul acceptance client | Reuse the same host slice, add two-dimensional thread/workgroup indexing, static or dynamic LDS tiles, repeated barriers and FP32 multiply/accumulate. Specify partial-tile bounds and float tolerance. The inspected upstream `hip/docs/tutorial/programming-patterns/matrix_multiplication.rst` provides a **naive**, untiled 2D matmul control; it must not be presented as evidence for tiled matmul. No BLAS dependency is necessary for this example. |

The examples do not by themselves exercise every G4 API. Add focused clients for
device selection/properties, memset, pointer queries, module/function/global
lookup, unload and error cases required by the plan. Do not infer those APIs
from a successful `hipMalloc`/kernel/copy test. The upstream addition example
only prints results, and the reduction example returns success even after a
validation mismatch; the qualification harness must enforce its own oracle and
failure status.

## Registration and launch contract

The source-compatible path requires more than the module loader. In
`clr/hipamd/src/hip_table_interface.cpp:73-118`, the compiler-facing entrypoints
include `__hipRegisterFatBinary`, `__hipRegisterFunction`,
`__hipUnregisterFatBinary`, and `__hipRegisterVar` for device globals.
Registration associates host function identity with a device symbol inside a
fat binary; the host function pointer is not a GPU code address.

AMD LLVM `clang/lib/CodeGen/CGCUDANV.cpp:420-510` constructs an argument-address
array, pops grid/block/shared-memory/stream configuration through
`__hipPopCallConfiguration`, and calls `hipLaunchKernel` (or
`hipLaunchKernel_spt` for the per-thread default-stream compiler option).
`hip/include/hip/hip_runtime_api.h:7978-7999` declares the matching push/pop
interfaces. Kernel configuration therefore needs correct nesting/thread-local
behavior, argument packing from metadata, stream ordering and asynchronous
argument lifetime handling. Copy argument values before the host stub returns.

Before choosing this ABI, compile tiny ordinary HIP C++ host/device fixtures
with the **selected** compiler and headers, inspect their undefined symbols,
constructors, fat-binary wrapper and target identifiers, then cross-link the host
side for PS5. Cover multiple translation units, templates, device globals and
unregister after outstanding work. The inspected newer compiler source is not
proof that LLVM 18 emits identical stubs. Keep module loading as a diagnostic
control, not a replacement for ordinary launch syntax.

## The pinned ggml client is a substantially larger contract

The references below are relative to
[`Prism e8fd209`](https://github.com/PrismML-Eng/llama.cpp/tree/e8fd209c975c80aa7fb36a8423571ff3b1b7e955).

| Requirement group | Concrete symbols and call sites |
| --- | --- |
| Discovery and memory reporting | `hipGetDeviceCount`, `hipGetDevice`, `hipSetDevice`, `hipGetDeviceProperties`, `hipDeviceGetAttribute`, `hipDeviceGetPCIBusId`, `hipMemGetInfo`; `ggml/src/ggml-cuda/ggml-cuda.cu:222-346,5332,6241`. Properties drive allocation and kernel routing: `totalGlobalMem`, `integrated`, `multiProcessorCount`, `sharedMemPerBlock`, `warpSize`, `gcnArchName`, name and fallback major/minor. A false architecture/property value can select incorrect kernels. |
| Storage and transfers | `hipMalloc`, `hipFree`, `hipMemset`, `hipMemsetAsync`, `hipMemcpyAsync`, `hipMemcpy2DAsync`, `hipHostMalloc`, `hipHostFree`; `ggml-cuda.cu:743-856,1278-1292` and `cpy.cu:477`. Device-to-device copies and pitched/strided tensor transfers matter, not just H2D/D2H. Host allocations may fall back when pinned allocation fails. |
| Ordering and lifetime | `hipStreamCreateWithFlags` with nonblocking streams, `hipStreamDestroy`, `hipStreamSynchronize`, `hipStreamWaitEvent`, `hipEventCreateWithFlags`, `hipEventRecord`, `hipEventSynchronize`, `hipEventDestroy`, `hipLaunchHostFunc`; `common.cuh:1438,1718`, `ggml-cuda.cu:2535-2538,4601-4605,4694-4696,5008-5024`. Explicit per-thread stream handles are used even without selecting the compiler's per-thread default-stream mode. Disabling graphs does not remove stream/event requirements. |
| BLAS | `hipblasCreate`, `hipblasDestroy`, `hipblasSetStream`, `hipblasSgemm`, `hipblasGemmEx`, `hipblasGemmBatchedEx`, `hipblasGemmStridedBatchedEx`; `common.cuh:1734-1736`, `ggml-cuda.cu:1550-1610`. Broader operators add `hipblasSgemmBatched`/`hipblasSgemmStridedBatched` in `out-prod.cu:83,110` and `hipblasStrsmBatched` in `solve_tri.cu:70`. FP16/BF16/FP32 types, transpose, leading dimensions, batching, stream binding and accumulation contracts need real implementations. |
| Device code | Ordinary launches throughout the CUDA sources compiled as HIP; shuffle/vote helpers in `common.cuh:450-570`, LDS/barriers, half/half2/BF16 operations, math functions, atomics and private storage across operators. `softmax.cu:6` includes HIP cooperative groups. The `sum.cu` HIP path uses the local `sum_rows` fallback; its rocPRIM/hipCUB mention is a suggestion, not an active dependency. |
| Quantized application coverage | `ggml-cuda.cu:5657-5658,5698-5699` includes Prism `PQ2_0`/`PTQ1_0` cases; `common.cuh:755` provides mixed unsigned/signed byte dot arithmetic used by ternary paths. Preserve exact quantization and Hadamard behavior. Dense GEMM coverage does not establish Bonsai support. |

This is a source inventory, not a preprocessed/link-complete symbol census.
`vendors/hip.h` contains aliases whose call sites are excluded on HIP: for
example `hipFuncSetAttribute` is not required by the shared-memory-limit macro
on that path (`common.cuh:237-251`). Do not implement every alias indiscriminately.

## Bounded first ggml configuration and remaining costs

`ggml/src/ggml-hip/CMakeLists.txt` requires packages `hip`, `hipblas` and
`rocblas`, rejects HIP older than 6.1, links `hip::host`, `roc::hipblas` and
`roc::rocblas`, and explicitly rejects `GGML_STATIC`. It compiles the shared
CUDA operator/template files as HIP. The backend uses C++17 through
`ggml/src/CMakeLists.txt:608`. Header dependencies include `hip_runtime.h`,
`hip_fp16.h`, `hip_bf16.h`, `hipblas/hipblas.h`, cooperative groups, and
`hip_fp8.h` when `HIP_VERSION >= 60200000` (`vendors/hip.h`). Device-library
linkage and the actual emitted builtins must be measured in a pinned build spike.

A sensible first client spike uses one device, `GGML_HIP_GRAPHS=OFF`,
`GGML_HIP_NO_VMM=ON`, `GGML_HIP_RCCL=OFF` and no unified-memory/host-registration
environment opt-ins. These are existing client controls, not claims of full
compatibility. Graphs default **on** and VMM defaults **off** in
`ggml/CMakeLists.txt:221-224`. Broader scope adds:

- Graph capture/instantiate/update/launch/destroy when graphs are enabled.
- `hipMemAddressReserve`, allocation handles, map/unmap, access permissions and
  release when VMM is enabled (`ggml-cuda.cu:548-650`).
- Peer access and peer copies for multiple devices; RCCL only when selected.
- `hipMallocManaged`/`hipMemAdvise` for unified-memory opt-in. The client handles
  `hipErrorNotSupported` from managed allocation by falling back to `hipMalloc`
  (`ggml-cuda.cu:140-165`). Host registration is separately opt-in and checks errors.
- Cooperative launch can remain honestly false in device properties; softmax
  has an ordinary-kernel fallback (`softmax.cu:353-372`). The referenced launch
  symbol may still need an explicit unsupported implementation at link time.

### PS5 architecture caveat

`vendors/hip.h` identifies `gfx1030` as RDNA2 but its RDNA1 list contains only
`gfx1010`/`gfx1012`, **not gfx1013**. `common.cuh:712-738` chooses
`__builtin_amdgcn_sdot4` for RDNA2, inline scalar-product instructions for
recognized RDNA1, and a scalar byte-product fallback otherwise. Its physical
warp helper still returns 32 for these GFX10 targets (`common.cuh:382-388`).
The PS5 baseline records disabled accelerated dot products in its Vulkan route;
therefore choosing gfx1030 for compiler compatibility can also select an
unqualified dot instruction in this client. Neither renaming the device nor
blindly adding gfx1013 to a family list establishes correctness. Compile both
paths, inspect ISA and validate a narrowly specified PS5 capability override.

## G4 architecture decision input

Implement the example slice over the existing native substrate first. Its host
registration and stream/error semantics must remain compatible with a future
upstream-backed runtime. Then compare a pinned ggml build against (a) that small
runtime and (b) a PS5 device backend beneath HIP/CLR. Reassess before duplicating
graph/VMM/scheduler behavior. An HSA adaptation adds a different queue/signal
contract; these client symbols alone do not require it.

The decisive build spike must resolve native host linkage, CMake package targets,
hipBLAS/rocBLAS execution and device-library/header versions as well as kernel
compilation. The current ggml static-link rejection needs an explicit PS5 build
decision; removing that check alone is not library compatibility. Refresh this
inventory for the pinned standalone G7 client and use a matched qualified control.

No client build, native execution, performance gain or G4/G7 completion is proved
by this inventory. No console was accessed and no source dependency was changed.
