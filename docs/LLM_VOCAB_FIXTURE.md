# Vocabulary-only GGUF fixture

Run `make llm-vocab-fixture`. Existing locked CPU build and conversion environment are authenticated and reused; missing model inputs are fetched and converted by the existing Make targets. Use `LLM_VOCAB_FIXTURE_DIR=results/llm-vocab-fixture-v2` for a fresh run. Existing output directories are rejected and preserved.

The producer uses the locked upstream GGUF writer to copy every model metadata key with its exact type and value, and writes zero weight tensors. It executes the genuine upstream vocabulary-only model loader and tokenizer on both original and derived files. Eight text cases (including empty, whitespace, Unicode and special-token strings), each with all four combinations of tokenizer flags, must yield identical tokens. Generated asset bytes are hashed before execution and checked afterward, alongside source, tool, environment, model and linked-library identities.

The receipt reports the standalone asset size. Final application packaging must enforce its own package size limit; this fixture does not depend on a historical application package. A successful fixture proves host vocabulary loading/tokenization only, not native execution, model inference, GPU execution or numerical equivalence.
