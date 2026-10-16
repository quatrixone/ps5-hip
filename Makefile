PYTHON ?= python3
CMAKE ?= cmake
CTEST ?= ctest
UFS_REV := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["ufs2tool"]["revision"])')
ZLIB_VERSION := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["host_zlib"]["version"])')
TITLE_ID := $(shell $(PYTHON) -c 'import json; print(json.load(open("title.json"))["title_id"])')
RESULTS_DIR ?= /user/homebrew/ps5-hip
export RESULTS_DIR
.DEFAULT_GOAL := all

.PHONY: all help deps-fetch deps-check deps-build device-fixtures host-test native-core public-hip-package clean-build

all: host-test native-core public-hip-package

help:
	@echo 'make deps-fetch  Fetch and verify locked upstream sources and archives'
	@echo 'make llm-deps-fetch  Fetch the optional locked standalone inference client'
	@echo 'make llm-cpu-reference  Build host-only inference reference tools (no PS5 execution)'
	@echo 'make llm-model-profile  Verify locked model metadata/tensor headers without downloading weights'
	@echo 'make llm-model-fetch  Fetch and verify locked source weights/tokenizer (2.2 GB; fresh receipt required)'
	@echo 'make llm-conversion-env  Prepare isolated, hash-locked CPU conversion dependencies (Linux x86-64)'
	@echo 'make llm-model-convert  Convert verified source model to F16 and pure Q4_0 (fresh output required)'
	@echo 'make llm-rmsnorm-fixture  Build unchanged upstream RMSNorm device module from authenticated sources'
	@echo 'make llm-q4-halfword-fixture  Build authenticated halfword-load diagnostic from original Q4 fixture'
	@echo 'make llm-q4-fixture  Build selected upstream Q8_1 and Q4_0 kernels (fresh outputs required)'
	@echo 'make llm-q4-reference  Build authenticated GPU-format Q8/Q4 numerical references offline'
	@echo 'make llm-rmsnorm-reference  Capture genuine CPU RMSNorm tensors and frozen comparison budget (fresh output required)'
	@echo 'make llm-reference-fixtures  Capture repeated CPU logits/tokens from the locked Q4_0 model (fresh output required)'
	@echo 'make deps-build  Build SDK, zlib, UFS2Tool, Lapy helper, and controllers'
	@echo 'make device-fixtures  Compile the host suite device objects'
	@echo 'make host-test   Build and run the host test suite'
	@echo 'make public-hip-copy-gpu-package  Build captured copy/kernel graph fixture'
	@echo 'make public-hip-host-test  Build and qualify genuine HIP/ROCclr with host sanitizers'
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
	$(CMAKE) -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build build/host
	$(CTEST) --test-dir build/host --output-on-failure

# A separate fresh directory preserves every sanitizer run's evidence.
QUALIFICATION_DIR ?= results/public-hip-host

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

.PHONY: llm-rmsnorm-reference
llm-rmsnorm-reference: llm-cpu-reference
	$(PYTHON) tools/build_llm_rmsnorm_reference.py --output $(LLM_RMSNORM_REFERENCE_DIR) --cpu-build $(LLM_CPU_BUILD_DIR) --model $(LLM_GGUF_DIR)/tinyllama-q4_0.gguf

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

llm-tool-test:
	$(PYTHON) tests/llm_rmsnorm_reference_test.py
	$(PYTHON) tests/llm_reference_fixtures_test.py
	$(PYTHON) tests/llm_conversion_test.py
	$(PYTHON) tests/fetch_llm_model_test.py
	$(PYTHON) tests/profile_llm_model_test.py
	$(PYTHON) tests/llm_reference_test.py

llm-model-profile: llm-tool-test
	$(PYTHON) tools/profile_llm_model.py --output $(LLM_PROFILE_DIR) --check-lock models/tinyllama-1.1b-chat-v1.0.json

.PHONY: public-hip-host-test
public-hip-host-test: deps-build
	$(PYTHON) tools/qualify_public_hip_source.py --output $(QUALIFICATION_DIR)

Q4_QUALIFICATION_DIR ?= results/public-hip-q4
.PHONY: public-hip-q4-host-test
public-hip-q4-host-test: public-hip-host-test llm-q4-fixture
	$(PYTHON) tools/qualify_public_hip_q4.py --base $(QUALIFICATION_DIR) --fixture $(LLM_Q4_FIXTURE_DIR) --output $(Q4_QUALIFICATION_DIR)

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
	$(CMAKE) -S . -B build/native -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/ps5-toolchain.cmake -DBUILD_TESTING=OFF
	$(CMAKE) --build build/native --target ps5hip_native_memory

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
