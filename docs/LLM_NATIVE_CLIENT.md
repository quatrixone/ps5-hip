# Native client link prerequisite (M0)

From the repository root in WSL/Linux x86-64:

```sh
make llm-native-client-link LLM_NATIVE_CLIENT_BUILD_DIR=build/llm-native-client-v1
```

The target prepares the locked SDK and client sources, runs four focused checks,
then compiles and links the genuine loader/tokenizer client. Choose a fresh
output directory for every run; failed outputs are retained for inspection.
No previous experiment's objects or receipts are required.

The build derives source selection from the locked upstream CMake files and
uses its model unity-build approach. It compiles 71 objects, including the real
CPU backend and all 161 factory-referenced model implementations. Keeping those
implementations resolves the real upstream factory references; it does not
advertise support for those models on PS5.

The receipt binds source, SDK, compiler tools, generated files, compiler header
dependencies, all objects and link inputs. Actual compiler dependency sets must
match the precompiled inventory. The linker plan is restricted to the expected
SDK libraries and toolchain; unknown link inputs fail qualification. The final
receipt also records the ELF and dynamic imports. SDK build provenance comes
from the existing dependency build; this target authenticates its installed
bytes rather than rebuilding the SDK independently.

This output is an SDK **payload ELF**, not an audited native application. Its
RWX layout and imports still need adaptation and validation through the existing
application startup, conversion and packaging path. The caller requests a
vocabulary-only load, but this build does not execute it. No model-loading,
tokenization, HIP backend integration or inference result is claimed.

Next integrate the app startup and genuine HIP backend, validate loading and
tokenization with the small model-derived vocabulary fixture, and enforce the
no-CPU-fallback check before scheduled tensor computation. Source compatibility,
native linkage and native execution remain separate acceptance gates.
