# FP32 NN GEMM operation subset (offline)

`ps5hipSgemmNN(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)` computes row-major
`C = alpha*A*B + beta*C` using the genuine HIP compiler and existing blocking
GeometryLds runtime route. Strides count floats. Link the generated
`linux-libps5hip-gemm.a` or PS5 archive with the matching runtime and registration
linker sections. This is a project operation subset, not a rocBLAS/hipBLAS port.

The interface is in `include/ps5hip/gemm.h`; implementation is `src/gemm.hip`.
The Linux archive is sanitizer-instrumented for qualification; the PS5 archive
is a compile-only native artifact. Neither is a qualified SDK release.
The operation allocates no application workspace; each 8x8 workgroup uses 512
bytes of static LDS. Runtime transient code, kernarg, command and marker backing
is separate and is not counted as zero peak memory. There is no CPU fallback.

## Contract

- Only FP32, NN, row-major, synchronous execution. No streams, transpose, batching,
  FP16 or performance guarantee. `alpha` and `beta` must be finite.
- Zero M or N returns success without touching pointers or initializing a backend.
- K=0 or alpha=0 does not read or validate A/B; C is scaled by beta. Beta=0 does
  not read C, allowing an uninitialized/NaN prior output. Signed-zero identity is
  not promised. General NaN/infinity matrix semantics are not qualified.
- Nonempty participating buffers must be tracked live device allocations with
  float alignment and sufficient remaining bytes. Strides must cover the logical
  columns. Checked products reject byte/address and runtime grid overflow.
- C's bounding span must not overlap A or B's bounding spans, including padding;
  A and B may overlap each other. No host pointer dereferences for validation.
- Caller owns buffers throughout the call and prevents concurrent free/write.
  Operation returns HIP errors; its return value is authoritative. Locally rejected
  scalar/layout arguments do not promise to update HIP's thread-local last error.
- Terminal runtime errors retain uncertain GPU ownership. Caller must obey the
  runtime's quarantine contract, not force-free allocations or continue execution.

## Qualification

Run `PYTHONPATH=tools:tests python3 tools/probe_gemm.py --reference-root <root>
--output <fresh-directory>` in WSL with the existing pinned toolchain inputs.
The recipe emits genuine Linux and PS5 combined objects and static archives,
extracts gfx1013 HSACO, checks LDS/scratch/ABI/ISA and compares device text across
hosts. Linux links the real registration, allocation, launch and submission
runtime under ASan/UBSan; only the transport and GPU arithmetic are synthetic.

Seven shapes, two changed-input rounds, padded strides, input preservation and
output padding/guards are checked against independent FP64 accumulation. Active
sizes range from 17 to 4745 values, K through 129. The tolerance is absolute
`2e-6 + 8*epsilon_fp32*(K+2)*(abs(alpha)*sum(abs(A*B))+abs(reference))`.
This is a declared conservative finite-input test bound, not a universal accuracy
guarantee. Thirteen bad scalar/layout/span cases must reject without submission;
zero extents also produce no work. Transport tests exercise allocation/mapping,
initialization, flush, missing/corrupt output/input/guard, submit/throw/kick,
timeout and cleanup faults with exact retained or released allocation counts.

The synthetic model cannot establish actual GPU arithmetic, barriers, numerical
accuracy, speed or native startup. G6 is not accepted: real client execution,
representative device numerical tests, peak memory and initialization/steady-state
timings remain. The current ROCclr adapter only admits the Initial route; using
this operation through ROCclr requires separately qualified GeometryLds support
and repaired native Os initialization. The custom HIP route used by this offline
proof is deliberately identified, not presented as upstream ROCclr execution.
