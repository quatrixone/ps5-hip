# Bounded upstream reduction admission

This offline path admits the unmodified upstream `warp_size_reduction.hip`
kernel at its original 1024x1x1 block size and4096 dynamic LDS bytes. It does
not advertise a general1024-thread capability or establish GPU correctness.

The sample is pinned to rocm-systems d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d
with SHA25642ffe47b3d8dfdd59d2c8ebaabcf7ae6451c97f1055fe9ca6fafebdcc274a57a.
Its absent popcount header is replaced by an explicitly local host-only pair
of32/64-bit overloads. The upstream sample file and kernel remain unchanged.
The original main prints errors but returns success after mismatches, and uses
random masks; its exit status is not an acceptance oracle.

`probe_hip_reduction.py` captures actual Linux/PS5 host and device include
closures with dependency-only compiler invocations before compilation. It binds
selected LLVM18 tools, SDK wrapper/dispatcher/config/revision, prepared HIP
headers and device libraries, rechecking inputs afterward. The saved checker
requires these inventories and rejects omitted artifacts, headers and wrappers.

The ReductionLds route now also supports the unchanged upstream RMSNorm
operator's resource layout. It requires wave32, no private/static LDS storage,
no spills, properties 0x409, fixed rsrc1 mode bits 0x60af0000, and SGPR/VGPR
allocation fields at most 3. Descriptor rsrc2 must be exactly 0x8c or 0x38c.
Explicit arguments use the shared validated scalar/pointer/aggregate layouts;
13 COv5 geometry fields and the dynamic-LDS field occupy the final 256 bytes.
The dynamic size field is at implicit-base + 120, not a fixed module offset.

The old canary still requests 4096 bytes, uses rsrc1 0x60af0080/rsrc2 0x8c,
and has a 288-byte kernarg area. The real RMSNorm case requests 128 bytes,
uses rsrc1 0x60af00c3/rsrc2 0x38c, and has 488 bytes for 26 explicit arguments
plus implicit storage. Existing LDS policy rounds its allocation to 1024 bytes;
the hidden argument remains 128 and the encoded rsrc2 is 0x1038c.

Dynamic requests are now admitted over the existing allocator's nonzero
1..65536-byte range instead of requiring 4096 for every reduction. This is
resource capacity, not algorithmic safety or hardware qualification of every
size. As with ordinary HIP dynamic shared memory, the caller must request the
bytes its kernel actually accesses. Metadata cannot infer that minimum. In
particular, launching the old canary with fewer than 4096 bytes is a caller
error; these changes do not make that invocation safe. Native numerical fixtures
must preserve the selected upstream launcher's exact dynamic size.

Preparation and packet encoding independently validate resource fields, the LDS
plan, 1024x1x1 geometry and all 256 implicit bytes, including reserved zeros.
Unsupported register modes, spills, private/static LDS and other geometry
remain rejected. No public CLR resource-capacity change is included here.

Resource policy comes from pinned PS5_Mesa0b2d6d1a61d9bbf89cf8beb88a696144f67c61f8:
`radv_pipeline_compute.c:38-44` computes32 waves and one CU group with no occupancy
cap; `ac_gpu_info.c:2481` sets SIMD_DEST_CNTL when waves/group is divisible by4.
The gfx10/gfx103 register descriptions place it at bit22, giving
COMPUTE_RESOURCE_LIMITS=0x00400000. LDS allocation4096 produces resource bits
0x40000, combined rsrc2=0x4008c. This is source-derived, not hardware validation.

The actual-codeobject sanitizer test independently checks packet register
addresses/values, all hidden/reserved bytes, max-workgroup/resource admission,
and oversized SGPR/VGPR allocations at selector, direct prepare and encode.
Compiler proof, host packet/model proof and native execution remain separate.
Before a native experiment, require a decisive actual-runtime host oracle and
independently verified deterministic masks/results/guards while preserving the
same kernel and16-block launch. No native package or console test is included.

## RMSNorm preparation evidence

`evidence/offline-rmsnorm-resource-preparation.json` records actual-object packet
and host-mapped preparation tests, old reduction regressions and native library
cross-compilation. Configure `PS5HIP_RMSNORM_OBJECT` with an authenticated
unchanged upstream `norm.hsaco` to register `rmsnorm_packet` and, with SDK
platform headers available, `native_rmsnorm_prepare`. Both tests explicitly
report `gpu_execution=false`; diagnostic argument patterns are not numerical
RMSNorm inputs. Source-only fixture generation now passes through
`make llm-rmsnorm-fixture`; see `evidence/offline-rmsnorm-source-fixture.json`.
Public CLR resource routing now passes the host qualification below; native
numerical execution remains the next gate.

## Public CLR RMSNorm resource routing

`evidence/offline-public-clr-rmsnorm-resources.json` records the genuine public
HIP path loading the unchanged mixed norm module. Unsupported sibling kernels
remain absent. Selected-kernel geometry and shared-memory limits reject invalid
requests synchronously; valid 1024-thread, 128-byte LDS launches then succeed
through the host transport for rows 1/2/17.

The oracle checks the actual shader entry and materialized code, all 488 kernarg
bytes, two null optional pointers, captured caller aggregates and clean ownership.
Eager and deferred loading pass; a deliberate packet mismatch is rejected.
The full 87-compilation CLR sanitizer suite, three focused CTests and a 77-object
native aggregate cross-build pass. This is packet/runtime evidence, not execution
of RMSNorm math. Dynamic shared-memory sizing remains the caller's responsibility.

Reproduce with a fresh source-qualified base and authenticated fixture:

```sh
python3 tools/qualify_public_hip_source.py --output results/source-public-rmsnorm-new
make llm-rmsnorm-fixture LLM_RMSNORM_FIXTURE_DIR=results/llm-rmsnorm-fixture-new
python3 tools/qualify_public_hip_rmsnorm.py --base results/source-public-rmsnorm-new --fixture results/llm-rmsnorm-fixture-new --output results/public-rmsnorm-model-new
```

Every output directory must be fresh. Native numerical packaging must include
the verified module and frozen CPU tensors, retain complete GPU readbacks and
use reviewed lifecycle controls before testing on .40.

## Native RMSNorm numerical candidate

The native title uses the unchanged module and six frozen CPU cases. It requests
1024 threads and 128 dynamic LDS bytes, initializes active output to NaN, and
checks 81,920 F32 values against `2e-5 + 2e-4 * abs(reference)`. Both allocations
include guards; inactive output and every input byte must remain unchanged.
This candidate is an operator test, not an inference client.

From a prepared source-only build environment, the complete optional workflow is:

```sh
make public-hip-rmsnorm-package RESULTS_DIR=/data/ps5-hip/results \
  RMSNORM_BUILD_DIR=build/public-hip-rmsnorm-new \
  LLM_RMSNORM_FIXTURE_DIR=results/llm-rmsnorm-fixture-new \
  LLM_RMSNORM_REFERENCE_DIR=results/llm-rmsnorm-reference-new
```

This fetches the locked model, converts Q4_0 weights, prepares CPU references and
device code, builds the title and audits the linked images. Fresh fixture and
reference directories preserve earlier evidence. To package already verified
fixtures, invoke `tools/build_public_hip.py --fixture rmsnorm` with explicit
`--rmsnorm-fixture`, `--rmsnorm-reference`, `--output` and `--results-dir`, then
run `tools/audit_public_hip_rmsnorm_native.py` on its three ELF images. Neither
command deploys or launches a console title.

The title saves twelve exclusive raw files in its configured results directory:
`rmsnorm-{stress,embedding}-{1,2,17}.{output,input-readback}.raw`. Each is
139,296 bytes; active F32 payload starts at byte 16 and contains `rows * 2048`
little-endian floats. Existing files cause rejection rather than overwrite.
Partial files remain diagnostic evidence. A new console experiment must preserve
old results and establish fresh names before launch.

The freeze/control path binds the exact module, CPU oracle and verifier, checks
freshness, and collects all raw files. `tools/verify_public_hip_rmsnorm.py`
independently checks their values, guards and input preservation before acceptance.
Its passing receipt alone cannot prove GPU origin: installed-image identity,
the observed run, natural teardown, service health and owned-lock cleanup must
also pass. Synthetic verifier tests are never native execution evidence.
