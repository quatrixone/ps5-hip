PYTHON ?= python3
CMAKE ?= cmake
CTEST ?= ctest
UFS_REV := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["ufs2tool"]["revision"])')
ZLIB_VERSION := $(shell $(PYTHON) -c 'import json; print(json.load(open("dependencies.json"))["host_zlib"]["version"])')
TITLE_ID := $(shell $(PYTHON) -c 'import json; print(json.load(open("title.json"))["title_id"])')

.PHONY: help deps-fetch deps-check deps-build host-test native-core clean-build

help:
	@echo 'make deps-fetch  Fetch and verify locked upstream sources and archives'
	@echo 'make deps-build  Build SDK, zlib, UFS2Tool, Lapy helper, and controllers'
	@echo 'make host-test   Build and run the host test suite'
	@echo 'make native-core Cross-compile the native core library'

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

host-test:
	$(CMAKE) -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build build/host
	$(CTEST) --test-dir build/host --output-on-failure

native-core: .deps/sdk/.ps5-sdk-revision
	$(CMAKE) -S . -B build/native -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/ps5-toolchain.cmake -DBUILD_TESTING=OFF
	$(CMAKE) --build build/native --target ps5hip_native_memory

clean-build:
	$(CMAKE) -E rm -rf build/host build/native
