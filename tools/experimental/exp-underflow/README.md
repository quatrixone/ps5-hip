# Natural-exp underflow cutoff experiment

This isolated LLVM18 experiment changes only the natural-exp f32 cutoff in
`AMDGPUTargetLowering::lowerFEXP`: `-0x1.9d1da0p+6f` becomes
`-0x1.9fe368p+6f`. The exp10 branch, ordered comparison, overflow handling,
range reduction and EXP/FLDEXP path remain unchanged. No oracle is changed.

`certificate.c` uses pinned MPFR4.2.2 directed 512-bit bounds to bracket
log(2^-150). Adjacent input `0xc2cff1b5` lies below the half-minimum-subnormal
boundary and rounds to zero; `0xc2cff1b4` lies above and rounds to bit1.
The latter was incorrectly selected to zero by the original compiler cutoff.
The certificate also retains a boundary neighborhood, old-cutoff neighbors,
minimum-normal-transition neighbors, zero and overflow controls. Its analytic
NaN/infinity statements are reference expectations, not target execution.

`probe.py` reads an existing pinned LLVM source build without rebuilding or
mutating it. It first recompiles the original lowering into private output,
replaces only that member in a copied static AMDGPU archive, and privately
relinks llc. The no-op compiler must emit byte-identical controls before the
patch is attempted. A second private source/object/archive/llc carries the
single constant correction. All other archive members remain exact.

The scalar/vector O1/O2 matrix for gfx1013/gfx1030 requires whole-assembly
equality except the intended literal inside the two natural-exp functions.
Exp10, fast natural exp, unrelated operations, all directives and other data
remain exact. The numerical path and IEEE denormal descriptor are retained.
`regenerate_math.py` then replays exact retained full HIP/OCML compiler inputs
to produce optimized device-library-linked bitcode and code with all three
compilers. Baseline regeneration must reproduce frozen `.text`, `.rodata`
and `.note`; patched full assembly differs at exactly two natural-exp
comparison sites: exp(x), and OCML tanh's exp(2*abs(x)). The latter's argument
is nonnegative, so its ordered underflow selection cannot trigger for finite
values. Both are instantiations of the same one-constant lowering correction,
with descriptors/metadata unchanged. It does not package a replacement.

Run from this worktree with fresh output directories:

```
python3 -B tests/exp_underflow_recipe_test.py
python3 -B tools/experimental/exp-underflow/probe.py --baseline <existing-build-baseline> --reference-root <main> --output results/experiment-new
python3 -B tools/experimental/exp-underflow/regenerate_math.py --experiment results/experiment-new --original <frozen-ocml-math-v2> --output results/math-regeneration-new
```

The retained development evidence is `results/experiment-v3` and
`results/math-regeneration-v1`. Earlier v1/v2 recorder-only failures remain:
resolving the lld symlink changed its driver personality; preprocessing under
`-Werror` incorrectly retained linker inputs. Neither reached a LLVM rebuild.
V3's audit-only supplement strengthens whole-assembly checking without
rebuilding correct saved binaries. The regeneration supplement records the
exact-two-site check and inspection-tool before-use binding improvement;
original raw receipts and source snapshots are preserved.
An initial supplemental assertion incorrectly expected only one generated
comparison; its failure and source are also retained, followed by the actual
optimized IR identifying both calls. This was an audit-count assumption,
not a new compiler or numerical failure.

This is a relink of supplied source-built components on one host, not a clean
or hermetic LLVM build. Actual consumed headers, link inputs, tools, libraries
and immutable oracle are bound; unused support trees are not inventoried.
MPFR proves the mathematical cutoff, not the GPU approximation of native EXP
or FLDEXP rounding. No numerical hardware pass, all-domain accuracy, package
qualification or production compiler acceptance follows from these receipts.
Independent review and a newly frozen candidate remain necessary before a
bounded hardware test; the original numerical failure remains unchanged.
