# Corrected natural-exp numerical candidate

This separate descendant of `722a59a` retains its C++ harness, lifetime handling,
257-input corpus, independent oracle and declared tolerances byte-for-byte.
The earlier candidate's numerical failure and cleanup evidence remain unchanged.

The new fixture explicitly derives from the original `999f73c9...` OCML receipt
and independently reviewed compiler correction `5e856b4`. The original library,
header, artifact and source verification runs first. The derived authority binds
the actual compiler experiment, supplements, source freeze, review, regenerated
objects and consumed input/output maps; it does not pretend to be the old receipt.
Each corrected ELF must equal the original except the two reviewed natural-exp
immediates, `0xc2ce8ed0` to `0xc2cff1b4`. The second occurs in tanh's
exp(2*abs(x)) path. Metadata, descriptors and all other bytes remain exact.

The native selection is gfx1013 O2, 5168 bytes, SHA256
`bbf825113bc2b8136062b9654868b446a3eab668e0bff2a6126f339e134fc61c`.
The immutable contract remains
`373c94b8ad8dac7326401ad54a1299d750298a5922c1a226d0d4816c854430b8`.
The existing numeric/lifetime/full-finalization checker remains unchanged.

Prepare a fresh fixture in this checkout (the lock update is explicit and
refuses an already derived lock):

```
python3 -B tools/export_math_correction.py --experiment-root <reviewed-exp-worktree> --original <frozen-ocml-math-v2> --reference-root <main> --output build/exp-corrected-fixture-v1 --write-lock
python3 -B tests/math_native_profile_test.py --fixtures build/exp-corrected-fixture-v1
```

For the isolated host suite, retain `PS5HIP_OCML_FIXTURE_DIR` as the original
compiler/library authority and separately set `PS5HIP_MATH_NATIVE_FIXTURE_DIR`
to the derived fixture. The four launch tests use the corrected objects; the
original OCML recipe audit continues to use the original objects and reports.

The artifact comparison and MPFR certificate are offline evidence. They do not
prove native EXP/FLDEXP subnormal behavior or any six-operation numerical pass.
No GPU result, production compiler acceptance, all-domain accuracy or broad G3
completion is claimed. A newly frozen and independently reviewed native package
must precede one new bounded console experiment, preserving the original failure
and the oracle's rejection of flushing a finite nonzero result to zero.
