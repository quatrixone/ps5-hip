# Frozen HIPIFY native title build

The measured `hipify` profile consumes the unedited translated client's PS5
combined object from `results/threaded-hipify-runtime-v4`
(receipt `2eef5b1b`, checked threaded runtime). Its exact
renamed entry is `_Z18hipify_client_mainv`: no argc/argv parameters. The single
`translated_add` kernel and 4,024-byte device ELF are embedded in eboot; this
profile has no `/app0/canary.hsaco`. The compiler proof keeps the original CUDA,
HIPIFY output and tool/header/library identities. Every compiler-proof source,
including CMake and the host model, must match the current tree. The title's
independent source inventory also binds the actual linked wrapper/core.

After independent source review and a clean candidate freeze:

```
python3 tools/build_native_vector_title.py --reference-root ../.. --kernel hipify \
  --hipify-probe results/threaded-hipify-runtime-v4
python3 tests/hipify_native_title_test.py --compiler results/threaded-hipify-runtime-v4 \
  --native-build build/native/gfx1013-hipify
python3 tools/stage_native_title.py --target gfx1013 --kernel hipify --results-dir /existing/ordinary/results
```

The builder uses a specific read-only-use SDK link, private native/Eden copies and
the existing pinned zlib/SDK/tool inputs, checking consumed identities before and
after. Startup must install shell and title finalizers before `_init`. Actual
linked audits bind the no-arg client call, one genuine compiler registration,
atexit unregister, wrapper/RELRO/bundle regions, imports and FSELF segments.
Reconstruction may differ only at the existing exact final unmapped 24-byte SIE
note; all other bytes remain exact. Staging re-inspects the bound artifacts rather
than trusting adjacent audit booleans. The generic UFS packager remains unchanged;
fsck and a full byte comparison of extracted app files are required.

The title's exclusive ordinary log is `ps5-hip-gfx1013-hipify.log`. Eden helper and
configurable result paths reuse the existing shell-managed lifecycle. Six launches
check 662 active sums and 2,238 complete buffer words including unchanged inputs
and guards. Late unregister/State/finalizer proof is mandatory. Native build and
package are offline observations; deployment and bounded GPU qualification need
separately frozen prospective controls. This is one translated example, not general
CUDA compatibility. Installed support trees are not fully inventoried or hermetic.
