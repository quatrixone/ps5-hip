PYTHON ?= python3
CMAKE ?= cmake
CTEST ?= ctest
CCACHE ?= auto
export PS5_HIP_CCACHE := $(CCACHE)
JOBS ?= $(shell $(PYTHON) tools/build_jobs.py)
export PS5_HIP_BUILD_JOBS := $(JOBS)
# Parallelize compiler workers, not recipes sharing dependency/build directories.
.NOTPARALLEL:
UFS_REV := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["ufs2tool"]["revision"])')
ZLIB_VERSION := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["host_zlib"]["version"])')
TITLE_ID := $(shell $(PYTHON) -c 'import json; print(json.load(open("title.json"))["title_id"])')
RESULTS_DIR ?= /user/homebrew/ps5-hip
export RESULTS_DIR
.DEFAULT_GOAL := all

.PHONY: all help deps-fetch deps-check deps-build device-fixtures host-test native-core public-hip-package clean-build

all: host-test native-core public-hip-package

help:
	@echo 'make llm-dense-projection-test  Host F16 projection primitive; reuse LLM_CPU_BUILD_DIR (no GPU qualification)'
	@echo 'make llm-original-generation-reference  Explicit fresh host comparison of original weights against GGUF; requires four input/output path variables'
	@echo 'make llm-original-format-tests  Build CPU prerequisites and test original-format readers/mapping/metadata/callbacks using synthetic fixtures'
	@echo 'make llm-original-metadata-test  Explicit local original tokenizer comparison; set LLM_ORIGINAL_METADATA_DIR and LLM_ORIGINAL_VOCAB_FIXTURE'
	@echo 'CCACHE=auto uses native compiler caching when installed; 0 disables, 1 requires it'
	@echo 'JOBS=N controls compiler workers (JOBS=1 is serial); default is bounded by CPU/available memory, capped at 8'
	@echo 'make deps-fetch  Fetch and verify locked upstream sources and archives'
	@echo 'make llm-deps-fetch  Fetch the optional locked standalone inference client'
	@echo 'make llm-cpu-reference  Build host-only inference reference tools (no PS5 execution)'
	@echo 'make llm-generation-reference  Capture short greedy CPU generation and reset references'
	@echo 'make llm-generation-corpus-reference  Capture five prompts with 16 generated tokens and reset'
	@echo 'make llm-native-generation-corpus-tests  Compare the production corpus helper with its CPU oracle'
	@echo 'make llm-generation-numeric-tests  Compare live HIP-interpreted greedy feedback to CPU (no GPU execution)'
	@echo 'make llm-native-generation-tests  Qualify the native helper under host HIP interpretation (no PS5 execution)'
	@echo 'make llm-model-profile  Verify locked model metadata/tensor headers without downloading weights'
	@echo 'make llm-model-fetch  Fetch and verify locked source weights/tokenizer (2.2 GB; fresh receipt required)'
	@echo 'make llm-conversion-env  Prepare isolated, hash-locked CPU conversion dependencies (Linux x86-64)'
	@echo 'make llm-model-convert  Convert verified source model to F16 and pure Q4_0 (fresh output required)'
	@echo 'make llm-rmsnorm-fixture  Build unchanged upstream RMSNorm device module from authenticated sources'
	@echo 'make llm-weighted-rmsnorm-fixture  Build selected unchanged learned-weight RMSNorm module'
	@echo 'make public-hip-weighted-rmsnorm-host-test  Qualify weighted RMSNorm through public HIP host model'
	@echo 'make llm-q4-halfword-fixture  Build authenticated halfword-load diagnostic from original Q4 fixture'
	@echo 'make llm-q4-fixture  Build selected upstream Q8_1 and Q4_0 kernels (fresh outputs required)'
	@echo 'make public-hip-weighted-qkv-package  Build and audit native weighted normalization/QKV diagnostic'
	@echo 'make llm-weighted-qkv-reference  Build learned normalization and full Q/K/V CPU oracles'
	@echo 'make llm-q4-reference  Build authenticated GPU-format Q8/Q4 numerical references offline'
	@echo 'make llm-rmsnorm-reference  Capture genuine CPU RMSNorm tensors and frozen comparison budget (fresh output required)'
	@echo 'make llm-executed-graph  Capture actual batch-one CPU graph through context512 (fresh output)'
	@echo 'make llm-vocab-fixture  Build zero-tensor upstream vocabulary GGUF and verify tokenizer equivalence'
	@echo 'make llm-native-client-link  Cross-build the real loader/tokenizer payload (no app execution)'
	@echo 'make llm-native-scheduler-test  Exercise scheduler allocation/cleanup with the real host GGML runtime'
	@echo 'make llm-native-app  Build/audit native app (default vocab; LLM_NATIVE_APP_PHASE=model-gate requires locked full model)'
	@echo 'make llm-reference-fixtures  Capture repeated CPU logits/tokens from the locked Q4_0 model (fresh output required)'
	@echo 'make deps-build  Build SDK, zlib, UFS2Tool, Lapy helper, and controllers'
	@echo 'make device-fixtures  Compile the host suite device objects'
	@echo 'make host-test   Build and run the host test suite'
	@echo 'make public-hip-copy-gpu-package  Build captured copy/kernel graph fixture'
	@echo 'make public-hip-host-test  Build and qualify genuine HIP/ROCclr with host sanitizers'
	@echo 'make public-hip-llm-host-test  Qualify fresh runtime and actual llama client with sanitizers'
	@echo 'make public-hip-q4-host-test  Qualify selected Q8/Q4 public HIP dispatch and ownership offline'
	@echo 'make public-hip-aggregate-package Build native aggregate compiler-launch title offline (fresh outputs)'
	@echo 'make public-hip-rmsnorm-package Build native RMSNorm numerical title and verified model/reference assets (fresh outputs)'
	@echo 'make public-hip-embedded-package Build native compiler registration/launch title offline (fresh outputs)'
	@echo 'make public-hip-embedded-host-test  Build compiler objects and qualify genuine CLR registration (fresh outputs)'
	@echo 'make public-hip-aggregate-host-test Qualify compiler aggregate arguments through genuine CLR (fresh outputs)'
	@echo 'make console-controls  Build launch/close controllers and read-only teardown helper offline'
	@echo 'make native-core Cross-compile the native core library'
	@echo 'make public-hip-package  Build, sign, package, and verify PPSA99014 offline'
	@echo 'make public-hip-memory-package  Build the public HIP allocation/free fixture offline'
	@echo 'make public-hip-capture-gpu-package Build captured GPU graph replay fixture offline'
	@echo 'make public-hip-graph-gpu-package Build explicit GPU graph replay fixture offline'
	@echo 'make public-hip-stream-package Build explicit-stream GPU/event ordering fixture offline'
	@echo 'make public-hip-graph-package Build host graph replay and capture-recovery fixture offline'
	@echo 'make public-hip-fill-package  Build the public HIP contiguous and pitched fill fixture offline'
	@echo 'make public-hip-launch-package  Build the public HIP kernel-launch fixture offline'
	@echo 'make public-hip-module-package  Build the public HIP module ownership fixture offline'
	@echo 'make public-hip-copy-package  Build the checked public HIP synchronous-copy fixture offline'
	@echo 'Package output overrides: PUBLIC_BUILD_DIR MEMORY_BUILD_DIR COPY_BUILD_DIR MODULE_BUILD_DIR LAUNCH_BUILD_DIR FILL_BUILD_DIR GRAPH_BUILD_DIR STREAM_BUILD_DIR'
	@echo 'make           Build dependencies, host tests, native core and bootstrap package'

deps-fetch:
	$(PYTHON) tools/fetch_locked_deps.py --fetch

deps-check:
	$(PYTHON) tools/fetch_locked_deps.py

.deps/sdk/.ps5-sdk-revision: dependencies.json tools/prepare_sdk.py | deps-fetch
	$(PYTHON) tools/prepare_sdk.py --sdk-source .deps/sources/sdk --release-archive .deps/downloads/sdk.zip

.deps/host-zlib-$(ZLIB_VERSION)/receipt.json: dependencies.json tools/prepare_host_zlib.py | deps-fetch
	$(PYTHON) tools/prepare_host_zlib.py --archive .deps/downloads/zlib.tar.gz

.deps/ufs2tool-$(UFS_REV)/receipt.json: dependencies.json locks/ufs2tool-packages.lock.json tools/prepare_ufs2tool.py | deps-fetch
	$(PYTHON) tools/prepare_ufs2tool.py --source .deps/sources/ufs2tool

.deps/elevation/receipt.json: elevation-inputs.json dependencies.json tools/prepare_elevation.py tools/validate_helper_elf.py .deps/sdk/.ps5-sdk-revision | deps-fetch
	$(PYTHON) tools/prepare_elevation.py --eden-source .deps/sources/eden --lapy-source .deps/sources/lapy --logging-header .deps/sources/ps5log/native/ps5log/ps5log.h

build/controllers/$(TITLE_ID)/receipt.json: controller-inputs.json title.json dependencies.json tools/build_controllers.py .deps/sdk/.ps5-sdk-revision | deps-fetch
	$(PYTHON) tools/build_controllers.py --protocol-source .deps/sources/protocol

deps-build: .deps/sdk/.ps5-sdk-revision .deps/host-zlib-$(ZLIB_VERSION)/receipt.json .deps/ufs2tool-$(UFS_REV)/receipt.json .deps/elevation/receipt.json build/controllers/$(TITLE_ID)/receipt.json

build/canaries/build.json: tools/build_canaries.py kernels/canary.hip dependencies.json
	$(PYTHON) tools/build_canaries.py >/dev/null

build/canaries/resources-build.json: tools/build_canaries.py kernels/resources.hip dependencies.json
	$(PYTHON) tools/build_canaries.py --fixture resources >/dev/null

device-fixtures: build/canaries/build.json build/canaries/resources-build.json

host-test: deps-build device-fixtures
	$(PYTHON) tools/build_jobs.py >/dev/null
	$(PYTHON) tests/build_jobs_test.py
	$(PYTHON) tests/native_ccache_test.py
	$(CMAKE) -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build build/host --parallel $(PS5_HIP_BUILD_JOBS)
	$(CTEST) --test-dir build/host --output-on-failure

# A separate fresh directory preserves every sanitizer run's evidence.
QUALIFICATION_DIR ?= results/public-hip-host

LLM_EXECUTED_GRAPH_DIR ?= results/llm-executed-graph
LLM_NATIVE_CLIENT_BUILD_DIR ?= build/llm-native-client
LLM_NATIVE_CLIENT_PREPARED_GENERATED_INPUTS ?=
LLM_NATIVE_APP_BUILD_DIR ?= build/llm-native-app
LLM_NATIVE_APP_PREPARED_SOURCE_BUILD ?=
LLM_NATIVE_APP_PHASE ?= vocab
LLM_NATIVE_APP_PREFIX_FIXTURE ?= $(LLM_PREFIX_FIXTURE_DIR)
LLM_NATIVE_APP_LAYER_FIXTURE ?= $(LLM_LAYER_ASSETS_DIR)
LLM_NATIVE_APP_FULL_FIXTURE ?= $(LLM_FULL_ASSETS_DIR)
LLM_NATIVE_APP_MODEL ?= $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf
LLM_NATIVE_APP_MODEL_SOURCE ?= $(LLM_MODEL_DIR)
LLM_NATIVE_APP_MODEL_FORMAT ?= gguf
LLM_NATIVE_APP_MODEL_REMOTE_PATH ?= $(if $(filter original_safetensors,$(LLM_NATIVE_APP_MODEL_FORMAT)),/data/hip/models/tinyllama-source,/data/hip/models/tinyllama-q4_0.gguf)
ifneq ($(words $(LLM_NATIVE_APP_MODEL_FORMAT)),1)
$(error LLM_NATIVE_APP_MODEL_FORMAT must be gguf or original_safetensors)
endif
ifeq ($(filter $(LLM_NATIVE_APP_MODEL_FORMAT),gguf original_safetensors),)
$(error LLM_NATIVE_APP_MODEL_FORMAT must be gguf or original_safetensors)
endif
ifeq ($(LLM_NATIVE_APP_MODEL_FORMAT),original_safetensors)
ifneq ($(LLM_NATIVE_APP_PHASE),model-full)
$(error original_safetensors requires LLM_NATIVE_APP_PHASE=model-full)
endif
endif
LLM_NATIVE_APP_CPU_BUILD ?= $(LLM_CPU_BUILD_DIR)
ifneq ($(filter $(LLM_NATIVE_APP_PHASE),vocab model-gate model-prefix model-layer model-full),$(LLM_NATIVE_APP_PHASE))
$(error LLM_NATIVE_APP_PHASE must be vocab, model-gate, model-prefix, model-layer or model-full)
endif
LLM_NATIVE_APP_MODEL_ARGS = $(if $(filter model-gate model-prefix model-layer model-full,$(LLM_NATIVE_APP_PHASE)),--model "$(LLM_NATIVE_APP_MODEL)" --model-source "$(LLM_NATIVE_APP_MODEL_SOURCE)" --cpu-build "$(LLM_NATIVE_APP_CPU_BUILD)" --model-remote-path "$(LLM_NATIVE_APP_MODEL_REMOTE_PATH)")
LLM_CPU_BUILD_DIR ?= build/llm-cpu-reference
LLM_PROFILE_DIR ?= results/llm-model-profile
LLM_MODEL_DIR ?= models/tinyllama-source
LLM_MODEL_RECEIPT ?= results/llm-model-source/receipt.json
LLM_CONVERSION_ENV ?= .deps/llm-conversion-env
LLM_GGUF_DIR ?= models/tinyllama-gguf
LLM_REFERENCE_DIR ?= results/llm-reference
LLM_RMSNORM_REFERENCE_DIR ?= results/llm-rmsnorm-reference
LLM_RMSNORM_FIXTURE_DIR ?= results/llm-rmsnorm-fixture
.PHONY: llm-rmsnorm-fixture
llm-rmsnorm-fixture: llm-deps-fetch deps-fetch
	$(PYTHON) tests/llm_rmsnorm_fixture_test.py
	$(PYTHON) tools/build_llm_rmsnorm_fixture.py --output $(LLM_RMSNORM_FIXTURE_DIR)

LLM_Q4_FIXTURE_DIR ?= results/llm-q4-fixture
.PHONY: llm-q4-fixture
llm-q4-fixture: llm-rmsnorm-fixture
	$(PYTHON) tests/llm_q4_fixture_test.py
	$(PYTHON) tools/build_llm_q4_fixture.py --rmsnorm-fixture $(LLM_RMSNORM_FIXTURE_DIR) --output $(LLM_Q4_FIXTURE_DIR)

LLM_WEIGHTED_RMSNORM_FIXTURE_DIR ?= results/llm-weighted-rmsnorm-fixture
.PHONY: llm-weighted-rmsnorm-fixture
llm-weighted-rmsnorm-fixture: llm-rmsnorm-fixture
	$(PYTHON) tests/llm_weighted_rmsnorm_fixture_test.py
	$(PYTHON) tools/build_llm_weighted_rmsnorm_fixture.py --rmsnorm-fixture $(LLM_RMSNORM_FIXTURE_DIR) --output $(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)

LLM_Q4_HALFWORD_FIXTURE_DIR ?= results/llm-q4-halfword-fixture
.PHONY: llm-q4-halfword-fixture
llm-q4-halfword-fixture: llm-q4-fixture
	$(PYTHON) tests/llm_q4_halfword_fixture_test.py
	$(PYTHON) tools/build_llm_q4_halfword_fixture.py --original-fixture $(LLM_Q4_FIXTURE_DIR) --output $(LLM_Q4_HALFWORD_FIXTURE_DIR)

LLM_Q4_REFERENCE_DIR ?= results/llm-q4-reference
.PHONY: llm-q4-reference
llm-q4-reference: llm-q4-fixture
	$(MAKE) llm-model-fetch
	$(MAKE) llm-model-convert
	$(MAKE) llm-rmsnorm-reference
	$(PYTHON) tests/llm_q4_reference_test.py
	$(PYTHON) tests/llm_q4_reference_build_test.py
	$(PYTHON) tools/build_llm_q4_reference.py --rmsnorm-reference $(LLM_RMSNORM_REFERENCE_DIR) --fixture $(LLM_Q4_FIXTURE_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf --output $(LLM_Q4_REFERENCE_DIR)

LLM_WEIGHTED_QKV_REFERENCE_DIR ?= results/llm-weighted-qkv-reference
.PHONY: llm-weighted-qkv-reference
llm-weighted-qkv-reference:
	@if ! test -f "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; fi
	@if ! test -f "$(LLM_RMSNORM_REFERENCE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-reference; fi
	$(PYTHON) tests/llm_weighted_qkv_reference_test.py
	$(PYTHON) tools/build_llm_weighted_qkv_reference.py --rmsnorm-reference $(LLM_RMSNORM_REFERENCE_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf --output $(LLM_WEIGHTED_QKV_REFERENCE_DIR)

.PHONY: llm-rmsnorm-reference
llm-rmsnorm-reference: llm-cpu-reference
	$(PYTHON) tools/build_llm_rmsnorm_reference.py --output $(LLM_RMSNORM_REFERENCE_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf

.PHONY: llm-native-client-link
llm-native-client-link: deps-build llm-deps-fetch
	$(PYTHON) tests/build_jobs_test.py
	$(PYTHON) tests/native_ccache_test.py
	$(PYTHON) tests/llm_native_client_build_test.py
	$(PYTHON) tests/llm_native_loader_test.py
	$(PYTHON) tests/llm_native_validation_cost_test.py
	$(PYTHON) tools/build_llm_native_client.py --output $(LLM_NATIVE_CLIENT_BUILD_DIR) $(if $(strip $(LLM_NATIVE_CLIENT_PREPARED_GENERATED_INPUTS)),--prepared-generated-inputs "$(LLM_NATIVE_CLIENT_PREPARED_GENERATED_INPUTS)")

.PHONY: llm-native-scheduler-test
llm-native-scheduler-test: llm-cpu-reference
	$(PYTHON) tests/llm_native_scheduler_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)"

.PHONY: llm-native-app
llm-native-app: deps-build llm-deps-fetch
	$(PYTHON) tests/build_jobs_test.py
	$(PYTHON) tests/native_ccache_test.py
	@if ! test -f "$(LLM_NATIVE_CLIENT_BUILD_DIR)/receipt.json"; then $(MAKE) llm-native-client-link; fi
	@if ! test -f "$(LLM_VOCAB_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-vocab-fixture; fi
	$(PYTHON) tests/watchdog_test.py
	$(PYTHON) tests/llm_native_host_allocator_test.py
	$(if $(filter model-gate model-prefix model-layer model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/llm_native_scheduler_test.py --cpu-build "$(LLM_NATIVE_APP_CPU_BUILD)",@true)
	$(PYTHON) tests/llm_native_app_build_test.py
	$(PYTHON) tests/llm_native_model_package_test.py
	$(PYTHON) tests/source_model_auth_test.py
	$(PYTHON) tests/source_model_callsites_test.py
	$(PYTHON) tests/source_model_cleanup_budget_test.py
	$(PYTHON) tests/source_model_execute_test.py
	$(PYTHON) tests/source_model_phase_budget_test.py
	$(PYTHON) tests/source_model_storage_edges_test.py
	$(PYTHON) tests/source_model_streaming_test.py
	$(PYTHON) tests/source_external_model_test.py
	$(PYTHON) tests/source_original_model_test.py
	$(PYTHON) tests/native_model_path_test.py
	$(PYTHON) tests/llm_native_validation_cost_test.py
	$(PYTHON) tests/public_hip_compile_inputs_test.py
	$(PYTHON) tests/native_header_batch_test.py
	$(PYTHON) tests/public_hip_llm_client_build_test.py
	$(PYTHON) tests/source_llm_client_controls_test.py
	$(PYTHON) tests/source_console_finalization_test.py
	$(PYTHON) tests/source_console_deployment_boundaries_test.py
	$(PYTHON) tests/native_call_transcript_test.py
	$(PYTHON) tests/llm_client_native_audit_test.py
	$(if $(filter model-prefix,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_model_prefix_freeze_test.py,@true)
	$(if $(filter model-prefix,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_prefix_console_test.py,@true)
	$(if $(filter model-prefix,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_prefix_authentication_test.py,@true)
	$(if $(filter model-prefix,$(LLM_NATIVE_APP_PHASE)),@if ! test -f "$(LLM_NATIVE_APP_PREFIX_FIXTURE)/receipt.json"; then $(MAKE) llm-prefix-assets LLM_PREFIX_FIXTURE_DIR="$(LLM_NATIVE_APP_PREFIX_FIXTURE)" LLM_PREFIX_MODEL="$(LLM_NATIVE_APP_MODEL)" LLM_CPU_BUILD_DIR="$(LLM_NATIVE_APP_CPU_BUILD)"; fi,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_model_layer_freeze_test.py,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_layer_console_test.py,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_layer_authentication_test.py,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/native_layer_transcript_test.py,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/llm_layer_readbacks_test.py,@true)
	$(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),@if ! test -f "$(LLM_NATIVE_APP_LAYER_FIXTURE)/receipt.json"; then $(MAKE) llm-layer-assets LLM_LAYER_ASSETS_DIR="$(LLM_NATIVE_APP_LAYER_FIXTURE)" LLM_LAYER_MODEL="$(LLM_NATIVE_APP_MODEL)" LLM_CPU_BUILD_DIR="$(LLM_NATIVE_APP_CPU_BUILD)"; fi,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/public_hip_full_assets_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/native_full_title_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/extended_native_admission_test.py,@true)
	$(if $(filter original_safetensors,$(LLM_NATIVE_APP_MODEL_FORMAT)),$(PYTHON) tests/native_full_original_loader_test.py,@true)
	$(if $(filter original_safetensors,$(LLM_NATIVE_APP_MODEL_FORMAT)),$(PYTHON) tests/native_original_title_test.py,@true)
	$(if $(filter original_safetensors,$(LLM_NATIVE_APP_MODEL_FORMAT)),$(PYTHON) tests/llm_original_native_audit_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/native_full_transcript_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/llm_full_readbacks_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_full_console_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(PYTHON) tests/source_model_full_freeze_test.py,@true)
	$(if $(and $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(filter generation corpus,$(LLM_FULL_PROFILE))),$(PYTHON) tests/native_generation_application_test.py,@true)
	$(if $(and $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(filter generation corpus,$(LLM_FULL_PROFILE))),$(PYTHON) tests/llm_generation_readbacks_test.py,@true)
	$(if $(and $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(filter generation corpus,$(LLM_FULL_PROFILE))),$(PYTHON) tests/source_generation_console_test.py,@true)
	$(if $(and $(filter model-full,$(LLM_NATIVE_APP_PHASE)),$(filter corpus,$(LLM_FULL_PROFILE))),$(PYTHON) tests/source_corpus_console_test.py,@true)
	$(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),@if ! test -f "$(LLM_NATIVE_APP_FULL_FIXTURE)/receipt.json"; then $(MAKE) llm-full-assets LLM_FULL_ASSETS_DIR="$(LLM_NATIVE_APP_FULL_FIXTURE)" LLM_FULL_GRAPH_MODEL="$(LLM_NATIVE_APP_MODEL)" LLM_CPU_BUILD_DIR="$(LLM_NATIVE_APP_CPU_BUILD)"; else $(PYTHON) -c 'import json; import sys; from pathlib import Path; json.loads((Path(sys.argv[1])/"receipt.json").read_text())["profile"]==sys.argv[2] or sys.exit("requested full profile differs")' "$(LLM_NATIVE_APP_FULL_FIXTURE)" "$(LLM_FULL_PROFILE)"; fi,@true)
	$(PYTHON) tools/build_llm_native_app.py --output "$(LLM_NATIVE_APP_BUILD_DIR)" --native-client "$(LLM_NATIVE_CLIENT_BUILD_DIR)" --vocab-fixture "$(LLM_VOCAB_FIXTURE_DIR)" --results-dir "$(RESULTS_DIR)" --phase "$(LLM_NATIVE_APP_PHASE)" $(LLM_NATIVE_APP_MODEL_ARGS) $(if $(filter original_safetensors,$(LLM_NATIVE_APP_MODEL_FORMAT)),--model-format "original_safetensors") $(if $(filter model-prefix,$(LLM_NATIVE_APP_PHASE)),--prefix-fixture "$(LLM_NATIVE_APP_PREFIX_FIXTURE)") $(if $(filter model-layer,$(LLM_NATIVE_APP_PHASE)),--layer-fixture "$(LLM_NATIVE_APP_LAYER_FIXTURE)") $(if $(filter model-full,$(LLM_NATIVE_APP_PHASE)),--full-fixture "$(LLM_NATIVE_APP_FULL_FIXTURE)") $(if $(strip $(LLM_NATIVE_APP_PREPARED_SOURCE_BUILD)),--prepared-source-build "$(LLM_NATIVE_APP_PREPARED_SOURCE_BUILD)")

.PHONY: llm-executed-graph
llm-executed-graph: llm-cpu-reference
	@if ! test -f "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch; fi
	@if ! test -f "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-convert; fi
	$(PYTHON) tests/llm_executed_graph_test.py
	$(PYTHON) tools/build_llm_executed_graph.py --output $(LLM_EXECUTED_GRAPH_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf

.PHONY: llm-reference-fixtures
llm-reference-fixtures: llm-cpu-reference
	$(PYTHON) tools/build_llm_reference_fixtures.py --output $(LLM_REFERENCE_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf
.PHONY: llm-conversion-env llm-model-convert
llm-conversion-env: llm-tool-test
	$(PYTHON) tools/prepare_llm_conversion.py --output $(LLM_CONVERSION_ENV)

llm-model-convert: llm-conversion-env llm-cpu-reference
	$(LLM_CONVERSION_ENV)/venv/bin/python tools/convert_llm_model.py --source $(LLM_MODEL_DIR) --output $(LLM_GGUF_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --environment $(LLM_CONVERSION_ENV)
.PHONY: llm-model-fetch
llm-model-fetch: llm-tool-test
	$(PYTHON) tools/fetch_llm_model.py --output $(LLM_MODEL_DIR) --receipt $(LLM_MODEL_RECEIPT)
.PHONY: llm-deps-fetch llm-cpu-reference llm-model-profile llm-tool-test
llm-deps-fetch:
	$(PYTHON) tools/fetch_locked_deps.py --llm-client --fetch

llm-cpu-reference: llm-deps-fetch llm-tool-test
	$(PYTHON) tools/build_llm_cpu_reference.py --output $(LLM_CPU_BUILD_DIR)

# Portable original-format prerequisites; no model download or PS5 qualification.
# Keep the optional-user-loader configuration separate from historical CPU builds.
LLM_ORIGINAL_CPU_BUILD_DIR ?= build/llm-original-format-cpu-reference
LLM_ORIGINAL_METADATA_DIR ?=
LLM_ORIGINAL_VOCAB_FIXTURE ?=
.PHONY: llm-original-format-cpu llm-original-format-tests llm-original-metadata-test
llm-original-format-cpu:
	$(MAKE) llm-cpu-reference LLM_CPU_BUILD_DIR="$(LLM_ORIGINAL_CPU_BUILD_DIR)"

llm-original-format-tests: llm-original-format-cpu
	$(PYTHON) tests/safetensors_reader_test.py
	$(PYTHON) tests/tinyllama_source_tensors_test.py
	$(PYTHON) tests/tinyllama_target_reader_test.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)"
	$(PYTHON) tests/tinyllama_source_metadata_test.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)" --sanitize
	$(PYTHON) tests/tinyllama_tensor_callback_test.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)"
	$(PYTHON) tests/llm_user_loader_test.py
	$(PYTHON) tests/llm_native_loader_test.py
	$(PYTHON) tests/llm_original_generation_test.py

# Explicit opt-in: authenticate/provide local originals and an independent vocab oracle.
# Check arguments before fetching/building prerequisites. No weights are opened.
llm-original-metadata-test:
	@test -n "$(LLM_ORIGINAL_METADATA_DIR)" -a -n "$(LLM_ORIGINAL_VOCAB_FIXTURE)" || { echo 'Set LLM_ORIGINAL_METADATA_DIR and LLM_ORIGINAL_VOCAB_FIXTURE explicitly' >&2; exit 2; }
	$(MAKE) llm-original-format-cpu
	$(PYTHON) tests/tinyllama_source_metadata_test.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)" --sanitize --original-metadata "$(LLM_ORIGINAL_METADATA_DIR)" --vocab-fixture "$(LLM_ORIGINAL_VOCAB_FIXTURE)"

llm-tool-test:
	$(PYTHON) tests/llm_rmsnorm_reference_test.py
	$(PYTHON) tests/llm_reference_fixtures_test.py
	$(PYTHON) tests/llm_conversion_test.py
	$(PYTHON) tests/fetch_llm_model_test.py
	$(PYTHON) tests/profile_llm_model_test.py
	$(PYTHON) tests/llm_reference_test.py
	$(PYTHON) tests/llm_cpu_ccache_test.py

llm-model-profile: llm-tool-test
	$(PYTHON) tools/profile_llm_model.py --output $(LLM_PROFILE_DIR) --check-lock models/tinyllama-1.1b-chat-v1.0.json

.PHONY: public-hip-host-test
public-hip-host-test: deps-build
	$(PYTHON) tools/qualify_public_hip_source.py --output $(QUALIFICATION_DIR)

Q4_QUALIFICATION_DIR ?= results/public-hip-q4
.PHONY: public-hip-q4-host-test
public-hip-q4-host-test: public-hip-host-test llm-q4-fixture
	$(PYTHON) tools/qualify_public_hip_q4.py --base $(QUALIFICATION_DIR) --fixture $(LLM_Q4_FIXTURE_DIR) --output $(Q4_QUALIFICATION_DIR)

WEIGHTED_RMSNORM_QUALIFICATION_DIR ?= results/public-hip-weighted-rmsnorm
.PHONY: public-hip-weighted-rmsnorm-host-test
public-hip-weighted-rmsnorm-host-test: public-hip-host-test llm-weighted-rmsnorm-fixture
	$(PYTHON) tools/qualify_public_hip_rmsnorm.py --base $(QUALIFICATION_DIR) --fixture $(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR) --output $(WEIGHTED_RMSNORM_QUALIFICATION_DIR)

Q4_HALFWORD_QUALIFICATION_DIR ?= results/public-hip-q4-halfword
.PHONY: public-hip-q4-halfword-host-test
public-hip-q4-halfword-host-test: public-hip-host-test llm-q4-halfword-fixture
	$(PYTHON) tools/qualify_public_hip_q4.py --base $(QUALIFICATION_DIR) --fixture $(LLM_Q4_HALFWORD_FIXTURE_DIR) --output $(Q4_HALFWORD_QUALIFICATION_DIR)

EMBEDDED_FIXTURE_DIR ?= build/public-hip-embedded-fixture
EMBEDDED_QUALIFICATION_DIR ?= results/public-hip-embedded
.PHONY: public-hip-embedded-fixture public-hip-embedded-host-test
public-hip-embedded-fixture: deps-build
	$(PYTHON) tests/public_hip_embedded_fixture_test.py
	$(PYTHON) tools/build_public_hip_embedded_fixture.py --output $(EMBEDDED_FIXTURE_DIR)

public-hip-embedded-host-test: public-hip-host-test public-hip-embedded-fixture
	$(PYTHON) tools/qualify_public_hip_embedded.py --base $(QUALIFICATION_DIR) --compiler $(EMBEDDED_FIXTURE_DIR) --output $(EMBEDDED_QUALIFICATION_DIR)

AGGREGATE_FIXTURE_DIR ?= results/hip-aggregate-compiler
AGGREGATE_QUALIFICATION_DIR ?= results/public-hip-aggregate
.PHONY: public-hip-aggregate-fixture public-hip-aggregate-host-test
public-hip-aggregate-fixture: deps-build
	PS5HIP_AGGREGATE_OUTPUT="$(AGGREGATE_FIXTURE_DIR)" $(PYTHON) tests/hip_aggregate_compiler_test.py

public-hip-aggregate-host-test: public-hip-host-test public-hip-aggregate-fixture
	$(PYTHON) tests/public_hip_aggregate_qualification_test.py
	$(PYTHON) tools/qualify_public_hip_aggregate.py --base $(QUALIFICATION_DIR) --compiler $(AGGREGATE_FIXTURE_DIR) --output $(AGGREGATE_QUALIFICATION_DIR)

SNAPSHOT_BUILD_DIR ?= build/source-make-snapshot
.PHONY: console-controls
console-controls: deps-build $(SNAPSHOT_BUILD_DIR)/receipt.json
	$(PYTHON) tools/verify_teardown_snapshot.py --sdk .deps/sdk --output $(SNAPSHOT_BUILD_DIR)

$(SNAPSHOT_BUILD_DIR)/receipt.json: tools/teardown_snapshot.cpp tools/teardown_snapshot.h tools/build_teardown_snapshot.py dependencies.json .deps/sdk/.ps5-sdk-revision
	$(PYTHON) tools/build_teardown_snapshot.py --sdk .deps/sdk --output $(SNAPSHOT_BUILD_DIR)

native-core: .deps/sdk/.ps5-sdk-revision
	$(PYTHON) tools/build_jobs.py >/dev/null
	$(CMAKE) -S . -B build/native -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/ps5-toolchain.cmake -DBUILD_TESTING=OFF
	$(CMAKE) --build build/native --parallel $(PS5_HIP_BUILD_JOBS) --target ps5hip_native_memory

PUBLIC_BUILD_DIR ?= build/public-hip
public-hip-package: deps-build
	$(PYTHON) tools/build_public_hip.py --output $(PUBLIC_BUILD_DIR)

MEMORY_BUILD_DIR ?= build/public-hip-memory
.PHONY: public-hip-memory-package
public-hip-memory-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture memory --output $(MEMORY_BUILD_DIR)

COPY_BUILD_DIR ?= build/public-hip-copy
.PHONY: public-hip-copy-package
public-hip-copy-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture copy --output $(COPY_BUILD_DIR)

MODULE_BUILD_DIR ?= build/public-hip-module
.PHONY: public-hip-module-package
public-hip-module-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture module --output $(MODULE_BUILD_DIR)

LAUNCH_BUILD_DIR ?= build/public-hip-launch
.PHONY: public-hip-launch-package
public-hip-launch-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture launch --output $(LAUNCH_BUILD_DIR)

FILL_BUILD_DIR ?= build/public-hip-fill
.PHONY: public-hip-fill-package
public-hip-fill-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture fill --output $(FILL_BUILD_DIR)

clean-build:
	$(CMAKE) -E rm -rf build/host build/native build/public-hip build/public-hip-memory build/public-hip-copy build/public-hip-module build/public-hip-launch

GRAPH_BUILD_DIR ?= build/public-hip-graph
.PHONY: public-hip-graph-package
public-hip-graph-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture graph --output $(GRAPH_BUILD_DIR)

STREAM_BUILD_DIR ?= build/public-hip-stream
.PHONY: public-hip-stream-package
public-hip-stream-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture stream --output $(STREAM_BUILD_DIR)

GRAPH_GPU_BUILD_DIR ?= build/public-hip-graph-gpu
.PHONY: public-hip-graph-gpu-package
public-hip-graph-gpu-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture graph_gpu --output $(GRAPH_GPU_BUILD_DIR)

CAPTURE_GPU_BUILD_DIR ?= build/public-hip-capture-gpu
.PHONY: public-hip-capture-gpu-package
public-hip-capture-gpu-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture capture_gpu --output $(CAPTURE_GPU_BUILD_DIR)


COPY_GPU_BUILD_DIR ?= build/public-hip-copy-gpu
.PHONY: public-hip-copy-gpu-package
public-hip-copy-gpu-package: deps-build
	$(PYTHON) tools/build_public_hip.py --fixture copy_gpu --output $(COPY_GPU_BUILD_DIR)

EMBEDDED_BUILD_DIR ?= build/public-hip-embedded-native
.PHONY: public-hip-embedded-package
public-hip-embedded-package: public-hip-embedded-fixture
	$(PYTHON) tests/public_hip_embedded_build_test.py
	$(PYTHON) tools/build_public_hip.py --fixture embedded --compiler-fixture $(EMBEDDED_FIXTURE_DIR) --output $(EMBEDDED_BUILD_DIR) --results-dir "$(RESULTS_DIR)"

AGGREGATE_BUILD_DIR ?= build/public-hip-aggregate-native
.PHONY: public-hip-aggregate-package
public-hip-aggregate-package: public-hip-aggregate-fixture
	$(PYTHON) tests/public_hip_embedded_build_test.py
	$(PYTHON) tests/public_hip_aggregate_qualification_test.py
	$(PYTHON) tools/build_public_hip.py --fixture aggregate --compiler-fixture $(AGGREGATE_FIXTURE_DIR) --output $(AGGREGATE_BUILD_DIR) --results-dir "$(RESULTS_DIR)"

RMSNORM_BUILD_DIR ?= build/public-hip-rmsnorm-native
.PHONY: public-hip-rmsnorm-package
public-hip-rmsnorm-package: deps-build llm-rmsnorm-fixture
	$(MAKE) llm-model-fetch
	$(MAKE) llm-model-convert
	$(MAKE) llm-rmsnorm-reference
	$(PYTHON) tests/public_hip_rmsnorm_build_test.py
	$(PYTHON) tests/rmsnorm_dump_test.py
	$(PYTHON) tools/build_public_hip.py --fixture rmsnorm --rmsnorm-fixture $(LLM_RMSNORM_FIXTURE_DIR) --rmsnorm-reference $(LLM_RMSNORM_REFERENCE_DIR) --output $(RMSNORM_BUILD_DIR) --results-dir "$(RESULTS_DIR)"
	$(PYTHON) tools/audit_public_hip_rmsnorm_native.py --raw $(RMSNORM_BUILD_DIR)/title.elf --converted $(RMSNORM_BUILD_DIR)/eboot.elf --extracted $(RMSNORM_BUILD_DIR)/extracted.elf --output $(RMSNORM_BUILD_DIR)/rmsnorm-native-audit.json

Q4_BUILD_DIR ?= build/public-hip-q4-native
.PHONY: public-hip-q4-package
public-hip-q4-package: deps-build llm-q4-reference
	$(PYTHON) tests/public_hip_q4_assets_test.py
	$(PYTHON) tests/public_hip_q4_build_test.py
	$(PYTHON) tests/public_hip_q4_log_test.py
	$(PYTHON) tests/source_q4_console_test.py
	$(PYTHON) tests/source_q4_freeze_test.py
	$(PYTHON) tests/q4_dump_test.py
	$(PYTHON) tests/public_hip_q4_native_audit_test.py
	$(PYTHON) tools/build_public_hip.py --fixture q4 --q4-fixture $(LLM_Q4_FIXTURE_DIR) --q4-reference $(LLM_Q4_REFERENCE_DIR) --output $(Q4_BUILD_DIR) --results-dir "$(RESULTS_DIR)"
	$(PYTHON) tools/audit_public_hip_q4_native.py --raw $(Q4_BUILD_DIR)/title.elf --converted $(Q4_BUILD_DIR)/eboot.elf --extracted $(Q4_BUILD_DIR)/extracted.elf --output $(Q4_BUILD_DIR)/q4-native-audit.json

WEIGHTED_QKV_BUILD_DIR ?= build/public-hip-weighted-qkv-native
.PHONY: public-hip-weighted-qkv-package
public-hip-weighted-qkv-package: deps-build llm-weighted-rmsnorm-fixture llm-q4-halfword-fixture llm-weighted-qkv-reference
	$(PYTHON) tests/public_hip_weighted_qkv_assets_test.py
	$(PYTHON) tests/public_hip_weighted_qkv_build_test.py
	$(PYTHON) tests/public_hip_weighted_qkv_native_audit_test.py
	$(PYTHON) tests/weighted_qkv_readbacks_test.py
	$(PYTHON) tests/source_weighted_qkv_controls_test.py
	$(PYTHON) tools/build_public_hip.py --fixture weighted_qkv --weighted-rmsnorm-fixture $(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR) --q4-fixture $(LLM_Q4_HALFWORD_FIXTURE_DIR) --weighted-qkv-reference $(LLM_WEIGHTED_QKV_REFERENCE_DIR) --output $(WEIGHTED_QKV_BUILD_DIR) --results-dir "$(RESULTS_DIR)"

LLM_VOCAB_FIXTURE_DIR ?= results/llm-vocab-fixture
.PHONY: llm-vocab-fixture
llm-vocab-fixture: llm-cpu-reference llm-conversion-env
	@if ! test -f "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch; fi
	@if ! test -f "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-convert; fi
	$(PYTHON) tests/llm_vocab_fixture_test.py
	$(LLM_CONVERSION_ENV)/venv/bin/python tools/build_llm_vocab_fixture.py --output $(LLM_VOCAB_FIXTURE_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf --environment $(LLM_CONVERSION_ENV)

# Fresh same-process runtime and actual llama host gate; output must not exist.
LLM_HOST_QUALIFICATION_DIR ?= results/public-hip-llm
.PHONY: public-hip-llm-host-test
public-hip-llm-host-test: deps-build llm-cpu-reference
	$(PYTHON) tests/hip_call_coverage_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)"
	$(PYTHON) tests/public_hip_llm_qualification_test.py
	$(PYTHON) tools/qualify_public_hip_source.py --output "$(LLM_HOST_QUALIFICATION_DIR)" --llm-cpu-build "$(LLM_CPU_BUILD_DIR)" --llm-model "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"

# Actual model prefix tests use HIP API mocks; this target never contacts a console.
LLM_PREFIX_HOST_TEST_DIR ?= results/llm-prefix-host-tests
.PHONY: llm-prefix-host-tests
llm-prefix-host-tests: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	$(PYTHON) tests/native_prefix_transcript_test.py
	$(PYTHON) tests/llm_prefix_readbacks_test.py
	$(PYTHON) tools/build_llm_prefix_host_tests.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_NATIVE_APP_MODEL)" --output "$(LLM_PREFIX_HOST_TEST_DIR)"

# Explicit real-model prefix qualification; each producer authenticates reused inputs.
LLM_GETROWS_FIXTURE_DIR ?= results/llm-getrows-fixture
LLM_PREFIX_REFERENCE_DIR ?= results/llm-prefix-reference
LLM_PREFIX_FIXTURE_DIR ?= results/llm-prefix-assets
LLM_PREFIX_MODEL ?= $(LLM_NATIVE_APP_MODEL)
.PHONY: llm-getrows-fixture llm-prefix-reference llm-prefix-assets
llm-getrows-fixture: llm-deps-fetch deps-fetch
	@if ! test -f "$(LLM_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-fixture; fi
	$(PYTHON) tests/llm_getrows_fixture_test.py
	$(PYTHON) tests/llm_getrows_validation_test.py
	@if ! test -f "$(LLM_GETROWS_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_fixture.py --get-rows-only --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_GETROWS_FIXTURE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from qualify_public_hip_q4 import fixture_inputs; fixture_inputs(Path(sys.argv[1]).resolve())' "$(LLM_GETROWS_FIXTURE_DIR)"

llm-prefix-reference: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_PREFIX_MODEL)"; then if test "$(LLM_PREFIX_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_PREFIX_MODEL: $(LLM_PREFIX_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tests/llm_prefix_reference_test.py
	@if ! test -f "$(LLM_PREFIX_REFERENCE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_prefix_reference.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_PREFIX_MODEL)" --output "$(LLM_PREFIX_REFERENCE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from build_llm_prefix_reference import authenticate_prefix_reference; authenticate_prefix_reference(Path(sys.argv[1]).resolve())' "$(LLM_PREFIX_REFERENCE_DIR)"

llm-prefix-assets: llm-deps-fetch deps-fetch
	@if ! test -f "$(LLM_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-fixture; fi
	@if ! test -f "$(LLM_GETROWS_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-getrows-fixture; fi
	@if ! test -f "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_weighted_rmsnorm_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_Q4_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_HALFWORD_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_halfword_fixture.py --original-fixture "$(LLM_Q4_FIXTURE_DIR)" --output "$(LLM_Q4_HALFWORD_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_PREFIX_REFERENCE_DIR)/receipt.json"; then $(MAKE) llm-prefix-reference; fi
	$(PYTHON) tests/public_hip_prefix_assets_test.py
	@if ! test -f "$(LLM_PREFIX_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_public_hip_prefix_assets.py --getrows "$(LLM_GETROWS_FIXTURE_DIR)" --normalized "$(LLM_RMSNORM_FIXTURE_DIR)" --weighted "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)" --q4 "$(LLM_Q4_HALFWORD_FIXTURE_DIR)" --reference "$(LLM_PREFIX_REFERENCE_DIR)" --output "$(LLM_PREFIX_FIXTURE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from public_hip_prefix_assets import prefix_inputs; prefix_inputs(Path(sys.argv[1]).resolve())' "$(LLM_PREFIX_FIXTURE_DIR)"

# Bounded one-layer qualification. Reuse receipts only after producer authentication.
LLM_LAYER_MODEL ?= $(LLM_NATIVE_APP_MODEL)
LLM_LAYER_FIXTURE_DIR ?= results/llm-layer-fixture
LLM_LAYER_REFERENCE_DIR ?= results/llm-layer-reference
LLM_LAYER_ASSETS_DIR ?= results/llm-layer-assets
LLM_LAYER_HOST_TEST_DIR ?= results/llm-layer-host-tests
.PHONY: llm-layer-fixture llm-layer-reference llm-layer-assets llm-layer-host-tests
llm-layer-fixture: llm-deps-fetch deps-fetch
	@if ! test -f "$(LLM_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-fixture; fi
	$(PYTHON) tests/llm_layer_fixture_test.py
	@if ! test -f "$(LLM_LAYER_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_layer_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_LAYER_FIXTURE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from build_llm_layer_fixture import fixture_inputs; fixture_inputs(Path(sys.argv[1]).resolve())' "$(LLM_LAYER_FIXTURE_DIR)"

llm-layer-reference: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_LAYER_MODEL)"; then if test "$(LLM_LAYER_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_LAYER_MODEL: $(LLM_LAYER_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tests/llm_layer_reference_test.py
	@if ! test -f "$(LLM_LAYER_REFERENCE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_layer_reference.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_LAYER_MODEL)" --output "$(LLM_LAYER_REFERENCE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from build_llm_layer_reference import authenticate_layer_reference; authenticate_layer_reference(Path(sys.argv[1]).resolve())' "$(LLM_LAYER_REFERENCE_DIR)"

llm-layer-assets: llm-deps-fetch deps-fetch
	@if ! test -f "$(LLM_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-fixture; fi
	@if ! test -f "$(LLM_GETROWS_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-getrows-fixture; fi
	@if ! test -f "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_weighted_rmsnorm_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_Q4_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_HALFWORD_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_halfword_fixture.py --original-fixture "$(LLM_Q4_FIXTURE_DIR)" --output "$(LLM_Q4_HALFWORD_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_LAYER_REFERENCE_DIR)/receipt.json"; then $(MAKE) llm-layer-reference; fi
	@if ! test -f "$(LLM_LAYER_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-layer-fixture; fi
	$(PYTHON) tests/public_hip_layer_assets_test.py
	@if ! test -f "$(LLM_LAYER_ASSETS_DIR)/receipt.json"; then $(PYTHON) tools/build_public_hip_layer_assets.py --getrows "$(LLM_GETROWS_FIXTURE_DIR)" --normalized "$(LLM_RMSNORM_FIXTURE_DIR)" --weighted "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)" --q4 "$(LLM_Q4_HALFWORD_FIXTURE_DIR)" --layer "$(LLM_LAYER_FIXTURE_DIR)" --reference "$(LLM_LAYER_REFERENCE_DIR)" --output "$(LLM_LAYER_ASSETS_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from public_hip_layer_assets import layer_inputs; layer_inputs(Path(sys.argv[1]).resolve())' "$(LLM_LAYER_ASSETS_DIR)"

llm-layer-host-tests: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	$(PYTHON) tests/native_layer_transcript_test.py
	$(PYTHON) tools/build_llm_layer_host_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_LAYER_MODEL)" --output "$(LLM_LAYER_HOST_TEST_DIR)"

# Original successful CPU decoder graphs, including inactive branch flags.
LLM_FULL_GRAPH_MODEL ?= $(LLM_NATIVE_APP_MODEL)
LLM_FULL_GRAPH_TRACE_DIR ?= results/llm-full-graph-trace
LLM_FULL_GRAPH_ADMISSION_DIR ?= results/llm-full-graph-admission
.PHONY: llm-full-graph-trace llm-full-graph-admission
llm-full-graph-trace: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_FULL_GRAPH_MODEL)"; then if test "$(LLM_FULL_GRAPH_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_FULL_GRAPH_MODEL: $(LLM_FULL_GRAPH_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tests/llm_full_graph_trace_test.py
	@if ! test -f "$(LLM_FULL_GRAPH_TRACE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_full_graph_trace.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --output "$(LLM_FULL_GRAPH_TRACE_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from build_llm_full_graph_trace import authenticate_full_graph_trace; authenticate_full_graph_trace(Path(sys.argv[1]).resolve())' "$(LLM_FULL_GRAPH_TRACE_DIR)"
	$(PYTHON) tests/llm_full_graph_trace_data_test.py --raw "$(LLM_FULL_GRAPH_TRACE_DIR)/raw"

llm-full-graph-admission: llm-full-graph-trace
	$(PYTHON) tools/build_llm_full_graph_admission.py --trace "$(LLM_FULL_GRAPH_TRACE_DIR)" --output "$(LLM_FULL_GRAPH_ADMISSION_DIR)"

LLM_FULL_HOST_TEST_DIR ?= results/llm-full-backend-host-tests
.PHONY: llm-full-host-tests
llm-full-host-tests: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_FULL_GRAPH_MODEL)"; then if test "$(LLM_FULL_GRAPH_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_FULL_GRAPH_MODEL: $(LLM_FULL_GRAPH_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tools/build_llm_full_backend_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --output "$(LLM_FULL_HOST_TEST_DIR)"

LLM_FULL_DISPATCH_TEST_DIR ?= results/llm-full-dispatch-host-tests
LLM_FULL_REFERENCE_DIR ?= results/llm-full-model-reference
.PHONY: llm-full-dispatch-tests llm-full-reference
llm-full-dispatch-tests: llm-full-graph-trace
	$(PYTHON) tools/build_llm_full_dispatch_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --graph "$(LLM_FULL_GRAPH_TRACE_DIR)/raw/run-0-pos-0.graph.json" --output "$(LLM_FULL_DISPATCH_TEST_DIR)"

llm-full-reference: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_FULL_GRAPH_MODEL)"; then if test "$(LLM_FULL_GRAPH_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_FULL_GRAPH_MODEL: $(LLM_FULL_GRAPH_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tests/llm_full_model_reference_test.py
	$(PYTHON) tools/build_llm_full_model_reference.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --output "$(LLM_FULL_REFERENCE_DIR)"

LLM_FULL_NUMERIC_TEST_DIR ?= results/llm-full-numeric-host-tests
LLM_GENERATION_REFERENCE_DIR ?= results/llm-generation-reference
LLM_GENERATION_REFERENCE_PROFILE ?= short
LLM_GENERATION_CORPUS_REFERENCE_DIR ?= results/llm-generation-corpus-reference
LLM_GENERATION_NUMERIC_TEST_DIR ?= results/llm-generation-numeric-host-tests
LLM_NATIVE_GENERATION_TEST_DIR ?= results/llm-native-generation-numeric-host-tests
LLM_NATIVE_GENERATION_CORPUS_TEST_DIR ?= results/llm-native-generation-corpus-host-tests
LLM_NATIVE_GENERATION_REFERENCE_TARGET ?= llm-generation-reference
.PHONY: llm-generation-reference llm-generation-corpus-reference llm-generation-numeric-tests
llm-generation-corpus-reference: LLM_GENERATION_REFERENCE_PROFILE = corpus
llm-generation-corpus-reference: LLM_GENERATION_REFERENCE_DIR = $(LLM_GENERATION_CORPUS_REFERENCE_DIR)
llm-generation-corpus-reference: llm-generation-reference
llm-generation-reference: llm-deps-fetch
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_FULL_GRAPH_MODEL)"; then if test "$(LLM_FULL_GRAPH_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_FULL_GRAPH_MODEL: $(LLM_FULL_GRAPH_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tests/llm_generation_test.py
	$(PYTHON) tests/llm_generation_reference_test.py
	$(PYTHON) tools/build_llm_generation_reference.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --output "$(LLM_GENERATION_REFERENCE_DIR)" --profile "$(LLM_GENERATION_REFERENCE_PROFILE)"

llm-generation-numeric-tests: llm-deps-fetch
	@if ! test -f "$(LLM_GENERATION_REFERENCE_DIR)/receipt.json"; then $(MAKE) llm-generation-reference; fi
	$(PYTHON) tests/llm_cpu_math_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)"
	$(PYTHON) tests/llm_generation_numeric_test.py
	$(PYTHON) tests/llm_extended_numeric_test.py
	$(PYTHON) tools/build_llm_generation_numeric_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --reference "$(LLM_GENERATION_REFERENCE_DIR)" --output "$(LLM_GENERATION_NUMERIC_TEST_DIR)"

.PHONY: llm-native-generation-tests llm-native-generation-corpus-tests
llm-native-generation-corpus-tests: LLM_GENERATION_REFERENCE_DIR = $(LLM_GENERATION_CORPUS_REFERENCE_DIR)
llm-native-generation-corpus-tests: LLM_NATIVE_GENERATION_TEST_DIR = $(LLM_NATIVE_GENERATION_CORPUS_TEST_DIR)
llm-native-generation-corpus-tests: LLM_NATIVE_GENERATION_REFERENCE_TARGET = llm-generation-corpus-reference
llm-native-generation-corpus-tests: llm-native-generation-tests
llm-native-generation-tests: llm-deps-fetch
	$(PYTHON) tests/llm_extended_numeric_test.py
	@if ! test -f "$(LLM_GENERATION_REFERENCE_DIR)/receipt.json"; then $(MAKE) $(LLM_NATIVE_GENERATION_REFERENCE_TARGET); fi
	$(PYTHON) tests/native_generation_transcript_test.py
	$(PYTHON) tests/native_generation_application_test.py
	$(PYTHON) tests/llm_generation_readbacks_test.py
	$(PYTHON) tests/source_model_full_freeze_test.py
	$(PYTHON) tests/source_generation_console_test.py
	$(PYTHON) tools/build_llm_generation_numeric_test.py --native-helper --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --reference "$(LLM_GENERATION_REFERENCE_DIR)" --output "$(LLM_NATIVE_GENERATION_TEST_DIR)"

.PHONY: llm-full-numeric-tests
llm-full-numeric-tests: llm-deps-fetch
	@if ! test -f "$(LLM_FULL_REFERENCE_DIR)/receipt.json"; then $(MAKE) llm-full-reference; fi
	$(PYTHON) tests/llm_full_numeric_test.py
	$(PYTHON) tools/build_llm_full_numeric_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --reference "$(LLM_FULL_REFERENCE_DIR)" --output "$(LLM_FULL_NUMERIC_TEST_DIR)"

LLM_FULL_HELPER_TEST_DIR ?= results/llm-native-full-helper-tests
.PHONY: llm-full-helper-tests
llm-full-helper-tests: llm-deps-fetch
	$(PYTHON) tests/llm_native_extended_test.py
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference; fi
	@if ! test -f "$(LLM_FULL_GRAPH_MODEL)"; then if test "$(LLM_FULL_GRAPH_MODEL)" = "$(LLM_GGUF_DIR)/tinyllama-q4_0.gguf"; then $(MAKE) llm-model-fetch && $(MAKE) llm-model-convert; else echo "Missing selected LLM_FULL_GRAPH_MODEL: $(LLM_FULL_GRAPH_MODEL)" >&2; exit 1; fi; fi
	$(PYTHON) tools/build_llm_full_backend_test.py --native-helper --cpu-build "$(LLM_CPU_BUILD_DIR)" --model "$(LLM_FULL_GRAPH_MODEL)" --output "$(LLM_FULL_HELPER_TEST_DIR)"

LLM_FULL_PROFILE ?= smoke
ifeq ($(LLM_NATIVE_APP_MODEL_FORMAT),original_safetensors)
ifneq ($(LLM_FULL_PROFILE),generation)
$(error original_safetensors requires LLM_FULL_PROFILE=generation)
endif
endif
LLM_FULL_ASSETS_DIR ?= $(if $(filter corpus,$(LLM_FULL_PROFILE)),results/llm-corpus-assets,$(if $(filter generation,$(LLM_FULL_PROFILE)),results/llm-generation-assets,results/llm-full-assets))
LLM_FULL_PROFILE_REFERENCE = $(if $(filter corpus,$(LLM_FULL_PROFILE)),$(LLM_GENERATION_CORPUS_REFERENCE_DIR),$(if $(filter generation,$(LLM_FULL_PROFILE)),$(LLM_GENERATION_REFERENCE_DIR),$(LLM_FULL_REFERENCE_DIR)))
.PHONY: llm-full-assets
llm-full-assets: llm-deps-fetch deps-fetch
	$(PYTHON) -c 'import sys; assert sys.argv[1] in ("smoke","generation","corpus"), "unsupported full profile"' "$(LLM_FULL_PROFILE)"
	@if ! test -f "$(LLM_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-rmsnorm-fixture; fi
	@if ! test -f "$(LLM_GETROWS_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-getrows-fixture; fi
	@if ! test -f "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_weighted_rmsnorm_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_fixture.py --rmsnorm-fixture "$(LLM_RMSNORM_FIXTURE_DIR)" --output "$(LLM_Q4_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_Q4_HALFWORD_FIXTURE_DIR)/receipt.json"; then $(PYTHON) tools/build_llm_q4_halfword_fixture.py --original-fixture "$(LLM_Q4_FIXTURE_DIR)" --output "$(LLM_Q4_HALFWORD_FIXTURE_DIR)"; fi
	@if ! test -f "$(LLM_LAYER_FIXTURE_DIR)/receipt.json"; then $(MAKE) llm-layer-fixture; fi
	@if ! test -f "$(LLM_FULL_PROFILE_REFERENCE)/receipt.json"; then $(MAKE) $(if $(filter corpus,$(LLM_FULL_PROFILE)),llm-generation-corpus-reference,$(if $(filter generation,$(LLM_FULL_PROFILE)),llm-generation-reference,llm-full-reference)); fi
	$(PYTHON) tests/public_hip_full_assets_test.py
	@if ! test -f "$(LLM_FULL_ASSETS_DIR)/receipt.json"; then $(PYTHON) tools/build_public_hip_full_assets.py --getrows "$(LLM_GETROWS_FIXTURE_DIR)" --normalized "$(LLM_RMSNORM_FIXTURE_DIR)" --weighted "$(LLM_WEIGHTED_RMSNORM_FIXTURE_DIR)" --q4 "$(LLM_Q4_HALFWORD_FIXTURE_DIR)" --layer "$(LLM_LAYER_FIXTURE_DIR)" --reference "$(LLM_FULL_PROFILE_REFERENCE)" --profile "$(LLM_FULL_PROFILE)" --output "$(LLM_FULL_ASSETS_DIR)"; fi
	$(PYTHON) -c 'import sys; from pathlib import Path; sys.path.insert(0,"tools"); from public_hip_full_assets import full_inputs; data=full_inputs(Path(sys.argv[1]).resolve()); assert data["profile"]==sys.argv[2], "full asset profile differs"' "$(LLM_FULL_ASSETS_DIR)" "$(LLM_FULL_PROFILE)"

# Explicit full-weight host short generation: both output directories must be fresh.
LLM_ORIGINAL_SOURCE_DIR ?=
LLM_ORIGINAL_GGUF ?=
LLM_ORIGINAL_REFERENCE_DIR ?=
LLM_ORIGINAL_GENERATION_DIR ?=
.PHONY: llm-original-generation-reference
llm-original-generation-reference:
	@test -n "$(LLM_ORIGINAL_SOURCE_DIR)" -a -n "$(LLM_ORIGINAL_GGUF)" -a -n "$(LLM_ORIGINAL_REFERENCE_DIR)" -a -n "$(LLM_ORIGINAL_GENERATION_DIR)" || { echo 'Set LLM_ORIGINAL_SOURCE_DIR, LLM_ORIGINAL_GGUF, LLM_ORIGINAL_REFERENCE_DIR and LLM_ORIGINAL_GENERATION_DIR explicitly' >&2; exit 2; }
	@test ! -e "$(LLM_ORIGINAL_REFERENCE_DIR)" -a ! -e "$(LLM_ORIGINAL_GENERATION_DIR)" -a "$(LLM_ORIGINAL_REFERENCE_DIR)" != "$(LLM_ORIGINAL_GENERATION_DIR)" || { echo 'Choose two distinct fresh output directories' >&2; exit 2; }
	$(MAKE) llm-original-format-cpu
	$(PYTHON) tools/build_llm_generation_reference.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)" --model "$(LLM_ORIGINAL_GGUF)" --output "$(LLM_ORIGINAL_REFERENCE_DIR)"
	$(PYTHON) tools/build_llm_generation_reference.py --cpu-build "$(LLM_ORIGINAL_CPU_BUILD_DIR)" --model "$(LLM_ORIGINAL_GGUF)" --output "$(LLM_ORIGINAL_GENERATION_DIR)" --original-source "$(LLM_ORIGINAL_SOURCE_DIR)" --reference "$(LLM_ORIGINAL_REFERENCE_DIR)"

.PHONY: llm-dense-projection-test
llm-dense-projection-test:
	@if ! test -f "$(LLM_CPU_BUILD_DIR)/receipt.json"; then $(MAKE) llm-cpu-reference LLM_CPU_BUILD_DIR="$(LLM_CPU_BUILD_DIR)"; fi
	$(PYTHON) tests/llm_dense_make_test.py
	$(PYTHON) tests/llm_dense_projection_test.py --cpu-build "$(LLM_CPU_BUILD_DIR)"
