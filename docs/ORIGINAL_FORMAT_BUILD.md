# Original TinyLlama loading: reproducible qualification

Main `6a7b941` ships the bounded reader, mapping, tokenizer metadata, target-byte
reader and callback. The optional-loader CPU producer, production generation
runner and Make wiring are now integrated. Their first complete supported-producer
CPU build, portable tests and full-model comparison pass. Native
original-format loading remains unqualified.
Use the source checkout and Linux/WSL prerequisites in `SOURCE_ONLY_REBUILD.md`;
no `.local` script or artifact is a build dependency.

## Portable tests without original model files

```sh
make llm-original-format-tests JOBS=4 CCACHE=auto \
  LLM_ORIGINAL_CPU_BUILD_DIR=build/llm-original-format-cpu-v1
```

This builds/checks locked CPU prerequisites and runs eight focused scripts.
The shared optional-tensor transformation generates a loader under the build
directory; the upstream checkout remains clean. It preserves the generated
loader's unity exclusion. Reuse an unchanged directory, but choose a new one
when producer inputs/configuration change; do not reuse historical pre-fix CPU
receipts. CPU builds honor JOBS and CCACHE=auto/0/1. The ccache launcher has
passed a real tiny C/C++ reuse test; full-build speedup is not measured.
Tests create/remove only synthetic files, including a sparse inventory fixture.

Optional original tokenizer check, with explicit local metadata and a small
vocabulary-only GGUF fixture:

```sh
make llm-original-metadata-test JOBS=4 \
  LLM_ORIGINAL_CPU_BUILD_DIR=build/llm-original-format-cpu-v1 \
  LLM_ORIGINAL_METADATA_DIR=models/tinyllama-source \
  LLM_ORIGINAL_VOCAB_FIXTURE=results/llm-vocab-fixture-v1/tinyllama-vocab.gguf
```

Use your actual fixture path. This target reads no weights.

## Fresh original-model short generation comparison

Supply the original locked TinyLlama directory and the qualified Q4_0 GGUF
oracle already built by the source-only model workflow. Select two distinct,
nonexistent directories below `results/`:

```sh
make llm-original-generation-reference JOBS=4 \
  LLM_ORIGINAL_CPU_BUILD_DIR=build/llm-original-format-cpu-v1 \
  LLM_ORIGINAL_SOURCE_DIR=models/tinyllama-source \
  LLM_ORIGINAL_GGUF=/absolute/path/to/tinyllama-q4_0.gguf \
  LLM_ORIGINAL_REFERENCE_DIR=results/original-short-oracle-v1 \
  LLM_ORIGINAL_GENERATION_DIR=results/original-short-comparison-v1
```

The target checks patched CPU tools, captures a fresh short GGUF reference,
then runs the same production decoder using original safetensors/config/tokenizer
files. No GGUF is passed to that runtime loader. The source lock authenticates
original files with a fresh weight hash; dependency, code, runtime-library and
oracle identities are checked again before acceptance. Existing result paths
are rejected; use fresh `-v2` names for another qualification, not overwritten
receipts. This is explicit, substantial host model work, not part of the default
portable tests.

Acceptance requires complete201, exact metadata/history/argmax/reset and all
384 layer/norm/logit boundaries over 16 decoder commits within unchanged
tolerances. Its `original-format-host-short-generation-compared` receipt cannot
serve as an independent GGUF oracle. Neither successful compilation nor the
small tests substitutes for this run.

## Proven and still missing

The supported Make comparison now passes with fresh weight hashes and a fresh CPU
build: complete201 and all384 boundaries bitwise-equal across two reset runs.
See `../evidence/offline-supported-original-model-generation.json`.

Scratch host evidence already loaded all 201 tensors and compared every prepared
byte (619,094,016) exactly; short decoding also matched all 384 boundaries bitwise.
Those runs used a scratch relink and prior source authentication with fresh
sidecar hashes/stat checks, without a fresh whole-weight SHA. See
`../evidence/offline-original-format-host-prerequisites.json`.

Native compilation and standalone linking of the new headers passed, but no
original-format PS5 app load/decode has run. Keep source/config/metadata and
callback owners alive and immutable; discard failed partial uploads. The
default backend setter cannot report silent backend failures. Native allocation,
error handling, ownership and GPU execution need their own milestone.

Current support is the strict locked TinyLlama single-file F16/BF16 safetensors
profile with original SentencePiece/HF sidecars, prepared into F32 norms and
Q4_0 matrices through the verified F16 roundtrip. A bounded sharded index reader
exists independently; full sharded callback integration, generic model variants,
PyTorch `.bin/.pt/.pth` files and training are not implemented by these targets.
