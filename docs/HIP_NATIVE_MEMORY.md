# Native public HIP allocation candidate

The isolated `development/g5-native-public-memory` branch composes the reviewed
ROCclr SVM patch over the frozen public HIP bootstrap source. The native title
calls genuine upstream `hipInit`, `hipGetDeviceCount`, `hipMalloc` and `hipFree`.
It checks two backing allocations, rejects an interior free and double free,
and requires both allocations to retire before finalization.

`tools/compose_native_public_hip_memory.py` checks exact preimages, applies the
memory patch and hashes every unchanged source. The strict linker compiles all
74 selected ROCclr, HIP and title units anew; `tools/verify_native_public_hip_memory.py`
checks the object/source manifest, public symbols and converter roundtrip. The
package tool stages PPSA99014 using the existing Lapy helper and configurable
results directory, then fscks and extracts the UFS2 image to compare every file.
The exact paths and hashes are in `evidence/offline-native-public-hip-memory.json`.

The title has **not** run on PS5. The package is an offline candidate, subject
to independent review and separate one-use deployment/launch controls. Console
allocation behavior, public HIP GPU launch and G5 remain unqualified.
