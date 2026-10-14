# Native public HIP bootstrap candidate

This isolated candidate links genuine upstream `hipInit(0)` and
`hipGetDeviceCount` against the PS5 AgcBackend and ROCclr port. It proves an
offline host-binary boundary only. The clean 74-object PPSA99014 package is
frozen and independently reviewed; it has not run on a PS5.

The reviewed public HIP admission patch applies exactly to the GPU-qualified
ROCclr source. Its profiling header needs the existing
`ports/clr/ps5-hip-correlation.patch`. Under `PS5HIP_CLR_NO_GRAPHICS`, upstream
`hip_gl.cpp` still included Linux GL headers and compiled helpers that are never
reachable. `tools/guard_native_hip_gl.py` makes those includes and helpers
conditional; the resulting native object retains the seven unsupported graphics
exports and has no GL import.

The PS5 SDK provides `__emutls_get_address` and libc++abi's
`__cxa_thread_atexit`, but libc++abi leaves its preferred
`__cxa_thread_atexit_impl` hook weak and unresolved. The native ELF converter
rejects that import. The SDK documents and implements the same hook as
`ps5___cxa_thread_atexit_impl`, including per-thread LIFO destruction. The
title's one-function forwarding definition makes the standard hook local.
The patched upstream HostThread also requires the reviewed pthread TLS helper's
`ps5_clr::adopt_current_thread`; an older GPU title object does not define it.

Reproduce offline in WSL from the ps5-hip repository root. Use a fresh result
directory for each stage:

```sh
python3 .local/g5-hip-native-link/tools/prepare_hip_admission.py \
  --base-source .local/g5-clr-gpu-startup/results/source-v1/source \
  --hip-source results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c \
  --output .local/g5-hip-native-link/results/native-source-v1
cd .local/g5-hip-native-link/results/native-source-v1
patch --batch --forward -p1 -i ../../../../ports/clr/ps5-hip-correlation.patch
cd ../../../..
python3 .local/g5-hip-native-link/tools/guard_native_hip_gl.py \
  .local/g5-hip-native-link/results/native-source-v1
python3 .local/g5-hip-native-link/tools/probe_native_public_hip_link.py \
  --output .local/g5-hip-native-link/results/native-hip-compile-all-v2 --all
python3 .local/g5-hip-native-link/tools/link_native_public_hip_bootstrap.py \
  --output .local/g5-hip-native-link/results/native-hip-console-clean-link-v2 --console-title
python3 .local/g5-hip-native-link/tools/convert_native_public_hip_bootstrap.py \
  --link .local/g5-hip-native-link/results/native-hip-console-clean-link-v2 \
  --output .local/g5-hip-native-link/results/native-hip-console-clean-convert-v2 --console-title
python3 .local/g5-hip-native-link/tools/verify_native_public_hip_bootstrap.py --console-title
```

The converter step uses the pinned `ps5-native-tool` and SDK/AGC stubs. The
verifier checks the resulting FSELF against the converter's known unmapped
24-byte SIE note.

The link rebuilds all 74 selected ROCclr, backend, startup and public HIP
objects and binds each direct source and object hash in its receipt. The clean
package and startup/teardown source path passed independent offline review.
This candidate does not include the separate public SVM memory bridge or
establish GPU-backed public HIP work.

The separate PPSA99014 title variant adds the existing Lapy elevation flow,
configurable result directory, exclusive log creation, bounded watchdog,
application finalizer and natural shell-exit probe. It records only public
initialization and one discovered device. Its offline build is:

```sh
python3 .local/g5-hip-native-link/tools/package_native_public_hip_bootstrap.py \
  --results-dir-source .local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42/app/results-dir.txt
```

Packaging requires a clean committed source tree and makes a new package with
verified UFS2 extraction. It never deploys or launches the image. Independent
review must check the one-use controls before any console validation. The
reviewed image SHA-256 is
`836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8`.

Future context-option work must repair upstream `hip_context.cpp`'s failed
host-context creation branch: release `hContext`, set `*status=false`, and
return. The frozen bootstrap passes zero flags and null properties under
`PS5HIP_CLR_NO_GRAPHICS`, so independent review found that branch unreachable
for this exact case absent memory corruption.
