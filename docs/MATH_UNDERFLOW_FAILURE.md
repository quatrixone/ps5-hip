# Exponential underflow found by native validation

The frozen math candidate `722a59a` failed the unchanged engineering contract at
output 645 (`6 * 107 + 3`, the `expf` result). Input bits `0xc2cff1b4` represent
`-103.972076416015625`. The MPFR reference rounds to the smallest positive FP32
subnormal, bits `0x00000001`; the declared envelope permits bits 1 through 3 and
explicitly rejects zero. The recovered application log reports `0x00000000`.

The original 20-second observation lost its FTP stream and remains inconclusive.
A separate read-only collection recovered the failure. A later exact-title close
returned the console to idle. Neither collection nor explicit cleanup converts
the original attempt into a clean native cycle. Compact provenance is in
`../evidence/native-math-current-failure.json`.

The failure has an offline compiler explanation. The HIP header's `expf` wrapper
calls `llvm.exp.f32`. LLVM 18's `AMDGPUTargetLowering::lowerFEXP` explicitly
selects zero when the input is below `-0x1.9d1da0p+6f` (`0xc2ce8ed0`, or
`-103.2789306640625`). The original shader contains that comparison at `0x1ca4`,
the zero selection at `0x1cc0`, and the result store at `0x1d1c`. The failed input
is below this cutoff. The explicit zero selection explains this point without
assuming a GPU denormal-mode defect or blaming an OCML exponential body.

The small input `../evidence/math-exp-underflow.ll` reproduces the cutoff without
HIP headers, device libraries or a console:

```sh
/usr/lib/llvm-18/bin/llc -march=amdgcn -mcpu=gfx1013 \
  -mattr=+wavefrontsize32,-wavefrontsize64 -O2 \
  evidence/math-exp-underflow.ll -o exp-underflow.s
```

The recorded compiler SHA-256 is
`15300bab42ea40e4b5c35779ac95d43640c0187fd70580f9fe8e7038d0d26c6d`.
Its output contains the same `0xc2ce8ed0` comparison and zero selection.

Next, derive the correct FP32 underflow boundary with directed MPFR evaluation,
qualify an isolated compiler correction, and build a new frozen candidate.
Preserve the existing oracle, failed image and raw evidence. This single-point
explanation does not establish correctness over the remaining inputs or operations.
