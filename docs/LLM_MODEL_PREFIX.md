# Actual-model embedding and QKV prefix

This is a native-app qualification milestone toward one transformer layer. It is not LLM inference, a complete transformer layer, training, or a general ROCm/PyTorch implementation.

The `model-prefix` phase loads the authenticated TinyLlama Q4_0 GGUF and selects the first layer's real graph dependencies: embedding GET_ROWS, RMSNorm, learned normalization weight, and full Q/K/V projections. Four inputs (tokens 0, 1, 2 and 31999) exercise first/last vocabulary rows and ordinary tokens. Each selected graph contains six ggml nodes and launches nine kernels through the actual PS5 ggml backend. Q8 packing is an internal step before each Q4 projection. The original full graph still rejects unsupported work, with no CPU activation fallback.

RoPE, attention/KV-cache computation, the attention output projection/residual, and the feed-forward path remain outside this prefix. Completing and validating those operations is necessary before claiming one whole layer, followed by complete-model decoding and token generation.

## Build from source

Use the repository's normal Linux/WSL prerequisites and dependency setup. These commands run offline from the console's perspective; dependency/model fetches can access their upstream download sources.

```sh
make deps-build
make llm-prefix-assets
make llm-prefix-host-tests
```

`llm-prefix-assets` prepares missing normalized/weighted/Q4/GET_ROWS modules, the authenticated CPU reference, and the aggregate receipt. Missing default model files are fetched and converted through the existing source-only model pipeline. Existing receipts are reused only after their consumer validates current sources, toolchains, model bytes and artifact hashes. Stale receipts fail; builds do not overwrite them automatically. Use a fresh output directory when source changes invalidate an experiment.

The component targets can be run separately:

```sh
make llm-getrows-fixture
make llm-prefix-reference
make llm-prefix-assets
```

The reference executes the genuine CPU graph twice and requires bitwise-identical outputs and source weights. It independently decodes Q4 embedding rows and checks normalization against the established FP64 error budgets. CPU projections use upstream Q8_0 arithmetic; the GPU path's Q8_1 arithmetic receives its own numerical validation. The host-test target uses explicit HIP mocks with sanitizers; passing it establishes no native execution or GPU numerical result.

For an already authenticated model outside the default model directory:

```sh
make llm-prefix-assets LLM_NATIVE_APP_MODEL=/absolute/path/tinyllama-q4_0.gguf
make llm-prefix-host-tests LLM_NATIVE_APP_MODEL=/absolute/path/tinyllama-q4_0.gguf
```

This path override selects the same qualified model bytes; it does not admit arbitrary models or formats. `LLM_CPU_BUILD_DIR` selects the existing CPU toolchain build. `LLM_PREFIX_REFERENCE_DIR`, `LLM_GETROWS_FIXTURE_DIR`, `LLM_PREFIX_FIXTURE_DIR` and `LLM_PREFIX_HOST_TEST_DIR` select output directories. All other existing fixture directory variables remain available. Do not delete prerequisite source or reference artifacts while a receipt still depends on them.

## Native package and model reuse

Build the separate phase with:

```sh
make llm-native-app LLM_NATIVE_APP_PHASE=model-prefix \
  LLM_NATIVE_APP_BUILD_DIR=build/llm-native-prefix-app \
  RESULTS_DIR=/user/homebrew/ps5-hip-prefix
```

The app target automatically prepares a missing prefix aggregate. `LLM_NATIVE_APP_PREFIX_FIXTURE` defaults to `LLM_PREFIX_FIXTURE_DIR`; an explicit override is propagated to the aggregate build and then authenticated by the app builder. Existing vocabulary and model-gate phases do not request prefix assets.

The model stays external at `/data/hip/models/tinyllama-q4_0.gguf` by default. `LLM_NATIVE_APP_MODEL_REMOTE_PATH` changes the configured console path. The native app uses the existing Eden elevation helper before opening that path. The package contains the application, vocabulary fixture and five authenticated modules, not the large model. Upload the model once and reuse it while its authenticated contents remain unchanged. A package rebuild does not require a new model upload.

Package creation alone is not validation. The source-only freezer, final-image audit and console controller must authenticate the selected package/assets/model before launching the native app. A fresh results directory is required because readbacks are created exclusively and synced. The four runs retain 24 F32 tensors, totaling 139264 bytes. Native completion, balanced cleanup and shell return must all appear in the strict transcript; those lines still leave numerical verification pending. The separate numerical consumer must authenticate the original model/reference and compare the readbacks before accepting this milestone.

The expected final observed data-buffer counts are five allocations/frees and one created/retired default queue. The transcript also measures largest-free direct-memory extents; these are not total-free-memory measurements. A discrepancy after first-time AGC initialization requires investigation rather than assuming either a successful cleanup or a proven model-buffer leak.

## Compiler cache reuse across fresh builds

Native app builds prepare a fresh source tree by default. Runtime units compile
with generated source/include paths relative to that tree, allowing ccache to
reuse unchanged units across output directories. Changing one runtime source
file does not require an identical prior tree to reuse other units. Header
changes still invalidate affected units, and every compile retains fresh
dependency discovery and before/after source/header verification. Always select
a fresh output directory. No completed-build hint or previous tree is required
for default cache reuse.

The optional `LLM_NATIVE_APP_PREPARED_SOURCE_BUILD=build/prior-app` remains
available for explicitly selecting prior compiler paths. The builder prepares
a fresh tree and compares the complete inventories, content and preparation
identities before accepting that selection; a mismatch fails. The selected
prior source tree and preparation identities become receipt dependencies and
must be retained. This option does not restore missing receipts or guarantee
cache hits.

Cross-directory reuse has been checked with a real unchanged runtime unit and
SDK source/header invalidation tests. Whole-package savings remain unmeasured;
cache hits still perform prerequisite authentication, dependency scans, hashing,
linking and packaging. See [source-only build controls](SOURCE_ONLY_REBUILD.md)
for cache and worker settings and the measured warm-build limit.
