# Native client link prerequisite (M0)

From the repository root in WSL/Linux x86-64:

```sh
make llm-native-client-link LLM_NATIVE_CLIENT_BUILD_DIR=build/llm-native-client-v1
```

The target prepares the locked SDK and client sources, runs focused build and
validation-cost checks,
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

Application assembly uses `app_objects()` to authenticate a schema-1 receipt
against the current producer, exact source/command selection, headers, SDK,
tools and generated artifacts. It returns only the 70 upstream client objects;
the payload caller and payload CRT are excluded. Older receipts must be rebuilt
in a fresh directory before application assembly; they are retained as historical
link evidence rather than accepted as current build inputs.

This output is an SDK **payload ELF**, not an audited native application. Its
RWX layout and imports still need adaptation and validation through the existing
application startup, conversion and packaging path. The caller requests a
vocabulary-only load, but this build does not execute it. No model-loading,
tokenization, HIP backend integration or inference result is claimed.

Next qualify the application startup and genuine HIP backend, validate loading
and tokenization with the small model-derived vocabulary fixture, and verify the
no-CPU-fallback check before scheduled tensor computation. Source compatibility,
native linkage and native execution remain separate acceptance gates.

## Vocabulary application build

`make llm-native-app` assembles a separate vocabulary-only native application.
It prepares missing client/vocabulary prerequisites, authenticates existing
receipts, and requires a fresh `LLM_NATIVE_APP_BUILD_DIR`. Select prerequisite
directories with `LLM_NATIVE_CLIENT_BUILD_DIR` and `LLM_VOCAB_FIXTURE_DIR`;
set the console logging location with `RESULTS_DIR`.

The application links 76 freshly compiled runtime/title objects, one ggml
backend object and the 70 authenticated client objects. Its package contains
the small vocabulary GGUF and an exact `vocab` phase configuration. A full
model asset is rejected by this package route. The driver checks native image
conversion, import owners, startup/finalization, mandatory wrapper code and
package extraction before accepting the build. The production build in
`build/llm-native-app-v3` passed all 197 commands and final input checks; its
19,890,176-byte package passed the 24-function native-image audit. This is an
offline build result; no console execution is established by this target.

The backend currently supports storage and rejects every tensor operation.
The vocabulary application tests tokenization, storage and synthetic rejection;
it cannot establish the actual native model execution gate. That gate needs
the authenticated full model and a separately reviewed console candidate.
