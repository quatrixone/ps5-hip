"""The one measured genuine-HIP title profile, distinct from handwritten canaries."""
import hashlib
import json
import struct
import re

PROVENANCE_FILES = {"llvm-pie.elf", "eboot.elf", "symbols.json", "startup-disassembly.stdout",
                    "app_crt.cpp", "ps5-pie.ld", "original-app_crt.cpp", "original-ps5-pie.ld"}

WORKERS_PROFILE = {
    "target": "gfx1013", "kernel_names": ["thread_words"],
    "client_entry": "_Z26ps5hip_workers_client_mainiPPc", "launch_entry": "threads_launch",
    "combined_object_sha256": "a51527f046165e437a0b9e2e151b43247340bea17e5497837d1091d15c3f81f3",
    "bundle_sha256": "352c3eb26097a8723e39183f0050968ae5a1d1f591d6c2638f0d53388070dab2",
    "code_sha256": "0484405bc2a5247de40c41105a170026be1ea1802d60930fb0df56099dfc3dc0",
    "code_bytes": 3696,
    "compiler_receipt_sha256": "f6c57047764720fbf6b61c7eb2d03939a914dc50a0942099d2b33a4a572f2d33",
    "dispatches": 5, "exact_values": 661, "word_comparisons": 1355,
    "template_comparisons": 1355, "created": 2, "joined": 2, "phase_checks": 17,
    "actors": [0, 1, 2, 1, 0], "counts": [1, 73, 257, 73, 257],
    "salts": [11, 101, 1009, 65521, 4294967280], "guard_words": 7,
    "data_words": 271, "kernarg_bytes": 272, "initialization_actor": "main",
}


WORKERS_INIT_PROFILE = dict(WORKERS_PROFILE, dispatches=4, exact_values=660, word_comparisons=1084,
    template_comparisons=1084, actors=[1,2,1,0], counts=[73,257,73,257],
    salts=[101,1009,65521,4294967280], initialization_actor="A", prebackend_required=True)

PROFILE = {
    "target": "gfx1013", "kernel_name": "_Z10vector_addPKfS0_Pfj",
    "combined_object_sha256": "b0c2518f45e63787221655374a0d678c02527eaf72c66a64b96fe3378333ddf0",
    "bundle_sha256": "a1a2909290d23a8e774f2e4ba5cabdc5bca683700d976aec2c5e2e6831055964",
    "code_sha256": "a6ab8bf13688777e6ca9af7048263147550b163bd365f263fb7abd2f3d2fb88b",
    "code_bytes": 4040, "dispatches": 2, "elements": 73,
    "exact_values": 146, "kernarg_bytes": 288,
    "data_logical_bytes": 16384, "interior_offset_bytes": 64,
}

# Fresh --title-entry compiler-only measurement; distinct from the bare-main proof.
ROUTES_PROFILE = {
    "target": "gfx1013", "kernel_names": ["routes_integer", "routes_lds", "routes_spill", "routes_geometry"],
    "client_entry": "_Z25ps5hip_routes_client_mainiPPc",
    "combined_object_sha256": "5c9c6741ed2f4203ea6c8dea8fe14fcd717a59d6498499524e7766e0373f1951",
    "bundle_sha256": "106c3e28a3c417bc21f3a47298f8eebadcb2e86f4822036760dddff29029e4aa",
    "code_sha256": "1f4a39915d66a8b562accf7a75e6518a6684746f2bc365423f16865b836c7ec0",
    "code_bytes": 10208,
    "compiler_receipt_sha256": "f6cefacc365bdd10bd9c84d33955992f84cb6ca17adf06c4764d6210da652bdf",
    "dispatches": 8, "sequences": 2, "outputs_per_sequence": [128, 2, 8192, 6144],
    "data_logical_bytes": 65536, "interior_offset_bytes": 64,
    "geometry_records": 384, "geometry_logical": 273, "geometry_inactive": 111,
    "static_lds_bytes": 256, "private_segment_bytes": 212,
}

GENERIC_PROFILE = {
    "target": "gfx1013", "kernel_names": ["affine_words", "block_sum"],
    "client_entry": "_Z26ps5hip_generic_client_mainiPPc",
    "combined_object_sha256": "b454e1dc47c7b97aa487fd7ddbf8ca0d3438df2a875853509be6582f7620c5c8",
    "bundle_sha256": "07875fab8b8df55f64404db805757ce45ccf11103583c6c5f67dc28727831974",
    "code_sha256": "793bea3aa8cdf8e0fbeb530f046dfedb3850bdde3533d2d5dd6a081c2b171ef7", "code_bytes": 4664,
    "compiler_receipt_sha256": "1c2ba52474b8d56705385378d29b108d12583ed093dddd7707f9a97d66756746",
    "sequence_launches": [0, 1, 3, 7], "dispatches": 11,
    "per_sequence_sizes": [1, 63, 64, 65, 127, 193, 257], "static_lds_bytes": 256,
    "zero_sequence_scope": "prebackend state within this process; not zero-work finalization",
}

MEMORY_FILL_PROFILE = {'target': 'gfx1013', 'kernel_names': ['affine_words', 'block_sum'], 'client_entry': '_Z30ps5hip_memory_fill_client_mainiPPc', 'combined_object_sha256': '7d3b52a67ddc6517662eb10b495e776280aae5478ef5c2e599ba7e4952a8e29d', 'bundle_sha256': '07875fab8b8df55f64404db805757ce45ccf11103583c6c5f67dc28727831974', 'code_sha256': '793bea3aa8cdf8e0fbeb530f046dfedb3850bdde3533d2d5dd6a081c2b171ef7', 'code_bytes': 4664, 'compiler_receipt_sha256': '24c2b61384e401c15376823b05f4861d686f6241256979aeba01e9536ee6e698', 'dispatches': 6, 'exact_values': 662, 'output_words': 746, 'input_words': 746, 'fills': 24, 'sizes': [1, 73, 257], 'rounds': 2, 'guard_words': 7}

DEVICE_COPY_PROFILE = {'target': 'gfx1013', 'kernel_names': ['affine_words', 'block_sum'], 'client_entry': '_Z30ps5hip_device_copy_client_mainiPPc', 'combined_object_sha256': '69fa078cd767a90cef2b6a5279a25af53a6288eba2caa30b4ba99aa993e3eeb7', 'bundle_sha256': '07875fab8b8df55f64404db805757ce45ccf11103583c6c5f67dc28727831974', 'code_sha256': '793bea3aa8cdf8e0fbeb530f046dfedb3850bdde3533d2d5dd6a081c2b171ef7', 'code_bytes': 4664, 'compiler_receipt_sha256': '74c5258893d5b7e7586cbdda46a0d5c87cba61c10ca04abaccf7306d176fe641', 'dispatches': 24, 'copies': 12, 'copied_words': 1324, 'copied_bytes': 5296, 'exact_values': 2648, 'readback_words': 5926, 'guard_comparisons': 630, 'allocations': 21, 'frees': 21, 'sizes': [1, 73, 257], 'rounds': 2, 'topologies': ['distinct', 'combined'], 'guard_words': 7}

HIPIFY_PROFILE = {
    "target": "gfx1013", "kernel_names": ["translated_add"],
    "client_entry": "_Z18hipify_client_mainv",
    "combined_object_sha256": "e6801b478241859f3e4961091098b89135a6f9b61024171d3b8b36de0cfbc56c",
    "bundle_sha256": "ec6228b5f8bb5d0ac6a79c5237ad715e949191dc0f6511295a83eb0008eb9612",
    "code_sha256": "86eeb4b965cf4b049990f4ea41c13301c95b864dee6078d2764de06d60985d8f",
    "code_bytes": 4024,
    "compiler_receipt_sha256": "9c9f89451323a092eeba82363c32865f2c76c67074073884350b9e67667c9014",
    "sequence_sizes": [1, 73, 257], "rounds_per_size": 2, "dispatches": 6,
    "exact_values": 662, "word_comparisons": 2238, "guard_words_each_end": 7,
    "original_cuda_sha256": "ecd50a2a6c8cfc4f58702b8725dfca2ae25dbf4049a358cc988562e8f1c2922a",
}


MATMUL_PROFILE = {'target': 'gfx1013',
 'kernel_names': ['tiled_matmul'],
 'client_entry': '_Z25ps5hip_matmul_client_mainiPPc',
 'combined_object_sha256': '9848fed87e57caedc3b0e8cc2dea66b32f1b31f69f99437d5393b9bc2c2437f3',
 'bundle_sha256': '0995fde75b6909239847ac68ffb1290cfc5d208ba64c267016e9ecfc3c5ab0db',
 'code_sha256': '7b862ae012fcde99802191b5588e35057b3f8c92a5f7f4808d859611062613f6',
 'code_bytes': 4528,
 'compiler_receipt_sha256': '82146a2290841da5ae59252d05c5000873e1133378424a67ddb257abfeba8cfd',
 'dispatches': 8,
 'cases': 4,
 'rounds': 2,
 'exact_values': 982,
 'shapes_mnk': [[8, 8, 8], [13, 21, 19], [1, 1, 17], [17, 9, 1]],
 'threads': [8, 8, 1],
 'static_lds_bytes': 512,
 'allocated_lds_bytes': 1024,
 'kernarg_bytes': 296,
 'rsrc2': 2444,
 'encoded_rsrc2': 67980,
 'guard_words': 16}


REDUCTION_PROFILE = {'target': 'gfx1013', 'kernel_names': ['_Z12block_reducePiPmS_m'], 'client_entry': 'ps5hip_reduction_companion_main', 'combined_object_sha256': '4380b8cc4b2f0820ce4d57c32131882b34b732d8a4f4bad894066ddc29c6d6c6', 'bundle_sha256': 'e456dfe0a15898ba15dc0531bc97e5454bfc82359c46dbb090e88bb031bd1409', 'code_sha256': '807b46dfaef2fbcb16be270fb76fe9b596fd3da96b0cf390e4da67ccebc7229a', 'code_bytes': 4488, 'compiler_receipt_sha256': 'd886b2ebd1c875be9156fa762f528234931e05421ed8a383e48d312cead3aa74', 'dispatches': 3, 'exact_values': 48, 'guard_comparisons': 72, 'threads': [1024, 1, 1], 'groups': [16, 1, 1], 'dynamic_lds_bytes': 4096, 'kernarg_bytes': 288}

def profile_for(kernel):
    if kernel == "hip_multi_query":
        from native_multi_query import PROFILE as MULTI_PROFILE
        return MULTI_PROFILE
    if kernel == "hip_reduction":return REDUCTION_PROFILE
    if kernel in ("hip_workers_main", "hip_workers_init"):
        return WORKERS_INIT_PROFILE if kernel == "hip_workers_init" else WORKERS_PROFILE
    if kernel == "hip_vector":
        return PROFILE
    if kernel == "hip_routes":
        return ROUTES_PROFILE
    if kernel == "hip_device_copy":
        return DEVICE_COPY_PROFILE
    if kernel == "hip_memory_fill":
        return MEMORY_FILL_PROFILE
    if kernel == "hip_generic":
        return GENERIC_PROFILE
    if kernel == "hip_matmul":
        return MATMUL_PROFILE

    if kernel == "hipify":
        return HIPIFY_PROFILE
    raise ValueError("unsupported genuine HIP title profile")


def provenance_files(kernel):
    if kernel == "hip_multi_query":
        return PROVENANCE_FILES | {"multi-a.o","multi-b.o","compiler-receipt.json"}
    profile_for(kernel)
    return PROVENANCE_FILES | ({"routes-combined.o", "compiler-receipt.json"} if kernel in ("hip_routes", "hip_reduction", "hip_generic", "hip_memory_fill", "hip_device_copy", "hip_matmul", "hipify", "hip_workers_main", "hip_workers_init") else set()) | ({"workers-agc-disassembly.stdout"} if kernel in ("hip_workers_main", "hip_workers_init") else set())


def validate_fself(container, elf):
    """Narrow plaintext format emitted by the pinned signer; no retail SELF support."""
    def require(valid):
        if not valid:
            raise ValueError("genuine HIP vector signed/ELF binding mismatch")
    require(len(container) >= 32 and len(elf) >= 64 and container[:4] == bytes.fromhex("5414f5ee"))
    count = struct.unpack_from("<H", container, 24)[0]
    table_end = 32 + 32 * count
    require(0 < count <= 128 and table_end + 64 <= len(container))
    require(elf[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<Q", elf, 32)[0] == 64)
    width, phnum = struct.unpack_from("<HH", elf, 54)
    end = 64 + width * phnum
    require(width == 56 and end <= len(elf) and table_end + end <= len(container))
    require(container[table_end:table_end + end] == elf[:end])
    headers = [struct.unpack_from("<IIQQQQQQ", elf, 64 + i * 56) for i in range(phnum)]
    selected = {i for i, h in enumerate(headers) if h[5] and h[0] in (1, 0x61000000, 0x61000010, 0x6fffff00)}
    found = set()
    for i in range(count):
        flags, offset, size, memory = struct.unpack_from("<QQQQ", container, 32 + i * 32)
        require(offset <= len(container) and size <= len(container) - offset)
        if flags & 0xfffff == 0x10004:
            continue
        require(flags & 0xfffff == 0x2804 and memory == size)
        index = flags >> 20
        require(index in selected and index not in found)
        found.add(index)
        h = headers[index]
        require(size == h[5] and h[2] <= len(elf) and size <= len(elf) - h[2])
        require(container[offset:offset + size] == elf[h[2]:h[2] + size])
    require(found == selected)


def audit_roundtrip(raw, extracted):
    """The pinned signer omits only the converter's unmapped final SIE build-id note."""
    if len(raw) < 64 or len(raw) != len(extracted) or raw[:6] != b"\x7fELF\x02\x01":
        raise ValueError("genuine HIP vector FSELF reconstruction length/header mismatch")
    width, count = struct.unpack_from("<HH", raw, 54)
    if width != 56 or struct.unpack_from("<Q", raw, 32)[0] != 64 or 64 + count * 56 > len(raw):
        raise ValueError("genuine HIP vector FSELF reconstruction program table mismatch")
    headers = [struct.unpack_from("<IIQQQQQQ", raw, 64 + i * 56) for i in range(count)]
    notes = [(i, h) for i, h in enumerate(headers) if h[0] == 4 and h[3] == 0 and h[6] == 0]
    if len(notes) != 1:
        raise ValueError("genuine HIP vector FSELF reconstruction omitted note ambiguous")
    index, h = notes[0]
    offset = h[2]
    if (index != count - 1 or h[1] != 0 or h[5] != 24 or offset + 24 != len(raw) or
            raw[offset:offset + 16] != bytes.fromhex("04000000080000000300000053494500") or
            any(other[5] and other[2] < offset + 24 and offset < other[2] + other[5]
                for i, other in enumerate(headers) if i != index) or
            extracted[offset:] != bytes(24) or raw[:offset] != extracted[:offset]):
        raise ValueError("genuine HIP vector FSELF reconstruction differs outside exact unmapped SIE note")
    return {"matches_converted_elf": False, "matches_except_unmapped_sie_note": True,
            "elf_sha256": hashlib.sha256(raw).hexdigest(),
            "extracted_sha256": hashlib.sha256(extracted).hexdigest(),
            "omitted_note": {"offset": offset, "bytes": 24, "raw_hex": raw[offset:].hex(),
                             "mapped": False, "program_header_index": index}, "native_executed": False}


def validate_profile(receipt, source_root):
    kernel = receipt.get("kernel")
    selected = profile_for(kernel)
    if (receipt.get("target") != "gfx1013" or
            receipt.get("shutdown_mode") != "shell-request-probe" or
            receipt.get("hardware_validated") is not False or
            json.dumps(receipt.get(kernel + "_profile"), sort_keys=True) != json.dumps(selected, sort_keys=True) or
            not (len(selected.get("modules",[]))==2 if kernel=="hip_multi_query" else all(selected.get(n) for n in ("combined_object_sha256", "bundle_sha256", "code_sha256", "code_bytes"))) or
            set(receipt.get("artifacts", {})) != {"eboot.bin", "lapy.elf", "sce_module/libc.prx"}):
        raise ValueError("genuine HIP vector title profile mismatch")
    sources = receipt.get("source_sha256", {})
    required = ["tools/build_native_vector_title.py", "tools/native_vector_title.py",
                "tools/hip_vector_title.cpp", "src/hip_runtime.cpp", "include/ps5hip/hip_runtime_control.h",
                "tools/native_shell_exit_probe.cpp", "tools/native_runtime.h", "evidence/offline-compatible-hip-headers.json"]
    required += (["src/hip_launch_profile.cpp"] if kernel=="hip_multi_query" else ["src/hip_launch_profile.cpp", "tests/reduction_companion.hip", "tools/probe_reduction_companion_compiler.py", "tests/reduction_companion_model.cpp", "tools/probe_reduction_companion_runtime.py", "tests/reduction_title_model.cpp", "tests/reduction_title_test.py", "tests/reduction_native_title_test.py", "tests/reduction_title_saved_test.py", "CMakeLists.txt", "tools/check_canary_log.py", "tools/stage_native_title.py", "tools/verify_native_case.py", "docs/HIP_REDUCTION_TITLE.md"] if kernel == "hip_reduction" else
                 ["tools/hip_vector_control.cpp", "tests/hip_vector_client.cpp"] if kernel == "hip_vector" else
                 ["src/hip_launch_profile.cpp", "tests/hip_threads.hip", "tools/hip_workers_client.cpp", "tools/hip_workers_client.h", "tools/hip_workers_agc.cpp", "tools/hip_workers_agc.h", "tools/probe_hip_workers_compiler.py", "tests/hip_workers_title_test.py", "tests/hip_workers_title_model.cpp", "docs/HIP_WORKERS_TITLE.md"] if kernel in ("hip_workers_main", "hip_workers_init") else
                 ["src/hip_launch_profile.cpp", "tests/cuda_vector_add.cu", "tools/probe_hipify_client.py", "tools/probe_hip_routes.py", "docs/HIPIFY_NATIVE_BUILD.md", "tests/hipify_native_title_test.py"] if kernel == "hipify" else
                 ["src/hip_launch_profile.cpp", "tests/hip_generic.hip", "tests/hip_device_copy_client.h", "tools/probe_hip_generic.py", "docs/HIP_DEVICE_COPY_TITLE.md"] if kernel == "hip_device_copy" else
                 ["src/hip_launch_profile.cpp", "tests/hip_generic.hip", "tests/hip_memory_fill_client.h", "tools/probe_hip_generic.py", "docs/HIP_MEMORY_FILL_TITLE.md"] if kernel == "hip_memory_fill" else
                 ["src/hip_launch_profile.cpp", "tests/hip_generic.hip", "tools/probe_hip_generic.py", "docs/HIP_GENERIC_TITLE.md"] if kernel == "hip_generic" else
                 ["src/hip_launch_profile.cpp", "tests/hip_tiled_matmul_blockdim.hip", "tools/probe_hip_matmul.py", "docs/HIP_MATMUL_TITLE.md"] if kernel == "hip_matmul" else
                 ["tools/hip_routes_control.cpp", "tests/hip_routes.hip", "tools/probe_hip_routes.py", "docs/HIP_ROUTES_COMPILER.md"])
    if kernel in ("hip_memory_fill","hip_device_copy"):required += ['tests/hip_generic_model.cpp', 'tests/hip_generic_title_model.cpp', 'tests/hip_generic_title_test.py', 'tests/hip_generic_title_profile_test.py', 'tools/check_canary_log.py', 'tools/stage_native_title.py', 'tools/verify_native_case.py', 'CMakeLists.txt']
    if kernel=="hip_multi_query":
        from native_multi_query import SOURCES
        required += SOURCES + ["tools/check_canary_log.py","tools/stage_native_title.py","tools/verify_native_case.py","CMakeLists.txt"]
    for name in required:
        if sources.get(name) != hashlib.sha256((source_root / name).read_bytes()).hexdigest():
            raise ValueError("genuine HIP vector title source provenance mismatch: " + name)
    if kernel in ("hip_workers_main", "hip_workers_init"):
        for name in ("tests/hip_workers_native_title_test.py", "tools/check_canary_log.py", "tools/stage_native_title.py", "tools/verify_native_case.py", "CMakeLists.txt"):
            if sources.get(name)!=hashlib.sha256((source_root/name).read_bytes()).hexdigest():
                raise ValueError("worker title source provenance mismatch: "+name)
    audits = receipt.get("vector_audits" if kernel == "hip_vector" else "routes_audits", {})
    if (set(audits) != ({"raw_pie", "converted_eboot", "startup", "fself_roundtrip"} | ({"workers_agc"} if kernel in ("hip_workers_main", "hip_workers_init") else set())) or
            any(audits[k].get("bundle_sha256") != selected["bundle_sha256"] for k in ("raw_pie", "converted_eboot")) or
            audits["startup"].get("install_order") != ["shell", "vector_finalizer", "_init", "reduction_title" if kernel == "hip_reduction" else "main"] or
            audits["startup"].get("compiler_atexit_unregister") is not True or
            audits["fself_roundtrip"].get("matches_except_unmapped_sie_note") is not True):
        raise ValueError("genuine HIP vector title embedded bundle/lifecycle audit missing")
    if kernel == "hip_vector":
        authority = json.loads((source_root / "evidence/offline-compatible-hip-headers.json").read_text())
        for suffix, key in ((".o", "combined_object_sha256"), (".hsaco", "code_sha256")):
            if authority["raw_file_hashes"]["ps5-vector-add-gfx1013-device-wchar4" + suffix]["sha256"] != selected[key]:
                raise ValueError("genuine HIP vector fixture authority changed")


def validate_reduction_compiler(raw_receipt, combined, source_root):
    from probe_hip_routes import section
    selected=REDUCTION_PROFILE
    if hashlib.sha256(raw_receipt).hexdigest()!=selected['compiler_receipt_sha256'] or hashlib.sha256(combined).hexdigest()!=selected['combined_object_sha256']:
        raise ValueError('reduction compiler authority changed')
    report=json.loads(raw_receipt)
    if report.get('classification')!='bound-declaration-only-reduction-companion-compiler' or report.get('native_linked') is not False or report.get('hardware_executed') is not False:
        raise ValueError('reduction compiler scope changed')
    from pathlib import Path
    normalized={}
    for name,digest in report['sources'].items():
        path=Path(name).resolve()
        if not path.is_relative_to(source_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('reduction source changed')
        normalized[str(path.relative_to(source_root.resolve()))]=digest
    if not normalized or not report['inputs'] or set(report['tools'])!={'clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm','llvm-config'}:raise ValueError('reduction dependency inventory')
    for name,digest in report['inputs'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('reduction input changed')
    for tool in report['tools'].values():
        if hashlib.sha256(Path(tool['path']).read_bytes()).hexdigest()!=tool['sha256']:raise ValueError('reduction tool changed')
    if hashlib.sha256(section(combined,b'.hip_fatbin')).hexdigest()!=selected['bundle_sha256']:raise ValueError('reduction bundle changed')
    # Return a relative view for the existing builder's source confinement check.
    report['sources']=normalized
    return report


def validate_compiler_dependencies(report, source_root, expected_sources=None):
    sources=report.get('sources',report.get('source_sha256',{}))
    if not sources or (expected_sources is not None and set(sources)!=set(expected_sources)):
        raise ValueError('compiler source inventory mismatch')
    for name,digest in sources.items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('compiler source changed: '+name)
    if not report.get('inputs') or set(report.get('tools',{}))!={'clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm'}:
        raise ValueError('compiler dependency inventory missing')
    from pathlib import Path
    for name,digest in report['inputs'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('compiler input changed: '+name)
    for tool in report['tools'].values():
        if hashlib.sha256(Path(tool['path']).read_bytes()).hexdigest()!=tool['sha256']:
            raise ValueError('compiler tool changed')


def validate_routes_compiler(raw_receipt, combined, source_root):
    selected = ROUTES_PROFILE
    if (hashlib.sha256(raw_receipt).hexdigest() != selected["compiler_receipt_sha256"] or
            hashlib.sha256(combined).hexdigest() != selected["combined_object_sha256"]):
        raise ValueError("genuine HIP routes compiler/combined binding mismatch")
    report = json.loads(raw_receipt)
    if report.get("title_entry") is not True or report.get("hardware_validated") is not False:
        raise ValueError("genuine HIP routes title compiler mode missing")
    for name, digest in report["source_sha256"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("genuine HIP routes compiler source changed: " + name)
    cases = [c for c in report["cases"] if c["host"] == "ps5"]
    if len(cases) != 1 or cases[0]["files"]["routes.o"]["sha256"] != selected["combined_object_sha256"]:
        raise ValueError("genuine HIP routes PS5 compiler case missing")
    files = cases[0]["files"]
    if (files["routes.bundle"]["sha256"] != selected["bundle_sha256"] or
            files["routes.hsaco"] != {"sha256": selected["code_sha256"], "bytes": selected["code_bytes"]}):
        raise ValueError("genuine HIP routes device binding mismatch")
    from probe_hip_routes import section
    if hashlib.sha256(section(combined, b".hip_fatbin")).hexdigest() != selected["bundle_sha256"]:
        raise ValueError("genuine HIP routes combined bundle mismatch")
    return report


def validate_generic_compiler(raw_receipt, combined, source_root, *, memory_fill=False, device_copy=False):
    selected=DEVICE_COPY_PROFILE if device_copy else MEMORY_FILL_PROFILE if memory_fill else GENERIC_PROFILE
    if hashlib.sha256(raw_receipt).hexdigest()!=selected["compiler_receipt_sha256"] or hashlib.sha256(combined).hexdigest()!=selected["combined_object_sha256"]:
        raise ValueError("generic title compiler/combined binding mismatch")
    report=json.loads(raw_receipt)
    from probe_hip_generic import compiler_sources, ROOT
    validate_compiler_dependencies(report,source_root,[str(p.relative_to(ROOT)) for p in compiler_sources(True,memory_fill,device_copy)])
    if report.get("schema")!=2 or report.get("compiler_only") is not True or report.get("title_entry") is not True or report.get("hardware_validated") is not False or report.get('models')!=[]:
        raise ValueError("generic title compiler mode missing")
    for name,digest in report["sources"].items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("generic title compiler source changed: "+name)
    if report.get("device_copy",False) is not device_copy:raise ValueError("generic copy variant mismatch")
    if report.get("memory_fill",False) is not memory_fill:raise ValueError("generic fill variant mismatch")
    cases=[c for c in report["cases"] if c["profile"]=="ps5"]
    if len(cases)!=1 or cases[0]["kernels"]!=selected["kernel_names"] or cases[0]["registration"]["register_function_calls"]!=2:
        raise ValueError("generic genuine compiler registration inventory")
    from probe_hip_routes import section
    from probe_hip_generic import check_profiles
    check_profiles(cases[0],selected["kernel_names"])
    artifacts=report["artifacts"]
    if (artifacts.get("ps5/routes.o")!=selected["combined_object_sha256"] or
            artifacts.get("ps5/routes.bundle")!=selected["bundle_sha256"] or
            artifacts.get("ps5/routes.hsaco")!=selected["code_sha256"] or
            hashlib.sha256(section(combined,b".hip_fatbin")).hexdigest()!=selected["bundle_sha256"]):
        raise ValueError("generic title device binding mismatch")
    return report


def validate_device_copy_compiler(raw_receipt,combined,source_root):
    return validate_generic_compiler(raw_receipt,combined,source_root,device_copy=True)

def validate_memory_fill_compiler(raw_receipt,combined,source_root):
    return validate_generic_compiler(raw_receipt,combined,source_root,memory_fill=True)


def validate_matmul_compiler(raw_receipt, combined, source_root):
    selected=MATMUL_PROFILE
    if hashlib.sha256(raw_receipt).hexdigest()!=selected["compiler_receipt_sha256"] or hashlib.sha256(combined).hexdigest()!=selected["combined_object_sha256"]:
        raise ValueError("matmul title compiler/combined binding mismatch")
    report=json.loads(raw_receipt)
    from probe_hip_matmul import compiler_sources, ROOT
    validate_compiler_dependencies(report,source_root,[str(p.relative_to(ROOT)) for p in compiler_sources()])
    if report.get("schema")!=2 or report.get("title_entry") is not True or report.get("compiler_only") is not True or report.get("hardware_validated") is not False or report.get("models")!=[]:
        raise ValueError("matmul title compiler mode mismatch")
    for name,digest in report["sources"].items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=digest:raise ValueError("matmul title source changed: "+name)
    cases=report["cases"]
    if [(c["host"],c["optimization"]) for c in cases]!=[("linux","O1"),("linux","O2"),("ps5","O1"),("ps5","O2"),("linux-model","O2")]:
        raise ValueError("matmul compiler inventory mismatch")
    from probe_hip_matmul import check_profile
    from probe_hip_routes import section
    for case in cases:
        check_profile(case)
        if case["lifecycle"]!={"modules":1,"functions":1,"atexit":1,"unregister":1,"ordinary_launch_stub":True}:raise ValueError("matmul genuine lifecycle inventory")
    artifacts=report["artifacts"]
    if (artifacts.get("ps5-O2/matmul.o")!=selected["combined_object_sha256"] or
            artifacts.get("ps5-O2/matmul.bundle")!=selected["bundle_sha256"] or
            artifacts.get("ps5-O2/matmul.hsaco")!=selected["code_sha256"] or
            hashlib.sha256(section(combined,b".hip_fatbin")).hexdigest()!=selected["bundle_sha256"]):
        raise ValueError("matmul title embedded device binding")
    return report


def validate_hipify_compiler(raw_receipt, combined, source_root):
    selected=HIPIFY_PROFILE
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    if digest(raw_receipt)!=selected["compiler_receipt_sha256"] or digest(combined)!=selected["combined_object_sha256"]:
        raise ValueError("HIPIFY title compiler/combined binding mismatch")
    report=json.loads(raw_receipt)
    from probe_hipify_runtime import compiler_sources, ROOT
    validate_compiler_dependencies(report,source_root,[str(p.relative_to(ROOT)) for p in compiler_sources()])
    if (report.get("schema")!=2 or report.get("compiler_only") is not True or report.get("models")!=[] or report.get("classification")!="hipify-title-compiler-only-pass" or
            report.get("generated_source_hand_edits")!=0 or report.get("original_main_renamed_by_flag") is not True or
            report.get("hardware_executed") is not False or report.get("native_linked") is not False):
        raise ValueError("HIPIFY original translated compiler mode missing")
    for name,value in report["sources"].items():
        if digest((source_root/name).read_bytes())!=value:
            raise ValueError("HIPIFY compiler source changed: "+name)
    if digest((source_root/"tests/cuda_vector_add.cu").read_bytes())!=selected["original_cuda_sha256"]:
        raise ValueError("HIPIFY original CUDA source changed")
    cases=[c for c in report["cases"] if c["host"]=="ps5"]
    if len(cases)!=1 or cases[0]["lifecycle"]!={"register_module_calls":1,"register_function_calls":1,"atexit_calls":1,"unregister_calls":1}:
        raise ValueError("HIPIFY genuine compiler registration inventory")
    desc=cases[0]["audit"]["descriptors"]
    if (set(desc)!={"translated_add"} or
            tuple(desc["translated_add"][n] for n in ("group_bytes","private_bytes","kernarg_bytes","rsrc1","rsrc2","rsrc3","properties","preload"))!=(0,0,288,0x60af0040,0x8c,0,0x409,0)):
        raise ValueError("HIPIFY measured descriptor changed")
    artifacts=report["artifacts"]
    if any(artifacts.get("ps5/translated"+suffix)!=selected[key] for suffix,key in ((".o","combined_object_sha256"),(".bundle","bundle_sha256"),(".hsaco","code_sha256"))):
        raise ValueError("HIPIFY compiler artifact binding mismatch")
    from probe_hipify_client import device_object
    from fat_binary_fixture_test import section
    bundle=section(combined,b".hip_fatbin");device=device_object(bundle)
    if digest(bundle)!=selected["bundle_sha256"] or digest(device)!=selected["code_sha256"] or len(device)!=selected["code_bytes"]:
        raise ValueError("HIPIFY embedded device binding mismatch")
    return report


def validate_workers_compiler(raw_receipt, combined, source_root):
    selected=WORKERS_PROFILE
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    if digest(raw_receipt)!=selected["compiler_receipt_sha256"] or digest(combined)!=selected["combined_object_sha256"]:
        raise ValueError("workers compiler/combined binding mismatch")
    report=json.loads(raw_receipt)
    if (report.get("classification")!="genuine-worker-kernel-compiler-proof" or
            report.get("hardware_executed") is not False or report.get("native_linked") is not False or report.get("kernel_source_unchanged") is not True):
        raise ValueError("workers compiler proof scope mismatch")
    for name,value in report["sources"].items():
        if digest((source_root/name).read_bytes())!=value:raise ValueError("workers compiler source changed: "+name)
    cases=[c for c in report["cases"] if c["host"]=="ps5"]
    if len(cases)!=1 or cases[0]["lifecycle"]!={"register_module_calls":1,"register_function_calls":1,"atexit_calls":1,"unregister_calls":1} or cases[0]["wrapper"]!={"payload_bytes":24,"section_bytes":24,"model_redzone_bytes":0}:
        raise ValueError("workers compiler registration/wrapper mismatch")
    descriptors=cases[0]["audit"]["descriptors"]
    if descriptors!={"thread_words":{"group_bytes":0,"private_bytes":0,"kernarg_bytes":272,"rsrc1":1622081600,"rsrc2":140,"rsrc3":0,"properties":1033,"preload":0}}:
        raise ValueError("workers measured descriptor mismatch")
    if any(report["artifacts"].get("ps5/threads"+suffix)!=selected[key] for suffix,key in ((".o","combined_object_sha256"),(".bundle","bundle_sha256"),(".hsaco","code_sha256"))):
        raise ValueError("workers compiler artifact binding mismatch")
    from probe_hipify_client import device_object
    from fat_binary_fixture_test import section
    bundle=section(combined,b".hip_fatbin");device=device_object(bundle)
    if digest(bundle)!=selected["bundle_sha256"] or digest(device)!=selected["code_sha256"] or len(device)!=selected["code_bytes"]:
        raise ValueError("workers embedded device mismatch")
    return report


def audit_workers_agc(disassembly, image):
    """Verify backend redirects and exactly one original import branch per wrapper."""
    from fat_binary_fixture_test import section
    from probe_host_registration import load_regions
    data=image.read_bytes();reader,_,_=load_regions(image)
    dynsym,dynstr=section(data,b".dynsym"),section(data,b".dynstr")
    names={i//24:dynstr[struct.unpack_from("<I",dynsym,i)[0]:].split(b"\0",1)[0].decode() for i in range(0,len(dynsym),24)}
    slots={}
    for name in (b".rela.dyn",b".rela.plt"):
        raw=section(data,name)
        for offset in range(0,len(raw),24):
            address,info,_=struct.unpack_from("<QQq",raw,offset)
            if info&0xffffffff in (6,7):slots[address]=names.get(info>>32)
    bodies=dict(re.findall(r"^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)",disassembly,re.M|re.S))
    results=[]
    for original,backend in (("sceAgcInit","_ZN6ps5hip10AgcBackendC2Ev"),("sceAgcDriverSubmitDcb","_ZN6ps5hip10AgcBackend6submitEPjj"),("sceAgcSuspendPoint","_ZN6ps5hip10AgcBackend4kickEv")):
        wrapper="__wrap_"+original
        # Some objdump versions select the C1 alias for the same constructor.
        body=bodies.get(backend,bodies.get(backend.replace("C2Ev","C1Ev"),""))
        if body.count("<"+wrapper+">")!=1 or wrapper not in bodies:raise ValueError("workers backend redirection missing: "+original)
        forwarded=[]
        for target in re.findall(r"\b(?:jmp|callq?)\s+0x([0-9a-f]+)",bodies[wrapper]):
            address=int(target,16);raw,_=reader(address,6)
            if raw[:2]==b"\xff\x25":forwarded.append(slots.get(address+6+struct.unpack_from("<i",raw,2)[0]))
        for target in re.findall(r"(?:callq?|jmp)\s+\*[^\n]+#\s+0x([0-9a-f]+)",bodies[wrapper]):forwarded.append(slots.get(int(target,16)))
        if forwarded.count(original)!=1 or any(n and n.startswith("sceAgc") and n!=original for n in forwarded):
            raise ValueError("workers wrapper original forwarding missing/duplicate: "+original)
        results.append({"backend":backend,"wrapper":wrapper,"original":original,"forward_calls":1})
    return {"calls":results,"native_executed":False}


def validate_build(receipt, source_root, provenance=None, signed=None):
    validate_profile(receipt, source_root)
    kernel = receipt["kernel"]
    selected = profile_for(kernel)
    audits = receipt["vector_audits" if kernel == "hip_vector" else "routes_audits"]
    binding = receipt.get(kernel + "_provenance", {})
    if provenance is None or signed is None or set(binding) != provenance_files(kernel):
        raise ValueError("genuine HIP vector linked artifact provenance missing")
    for name, digest in binding.items():
        path = provenance / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("genuine HIP vector linked artifact changed: " + name)
    if kernel=="hip_multi_query":
        from native_multi_query import validate_compiler
        validate_compiler((provenance/"compiler-receipt.json").read_bytes(),[(provenance/("multi-"+m+".o")).read_bytes() for m in "ab"],source_root)
    elif kernel in ("hip_workers_main", "hip_workers_init"):
        validate_workers_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                  (provenance / "routes-combined.o").read_bytes(), source_root)
        if audit_workers_agc((provenance / "workers-agc-disassembly.stdout").read_text(), provenance / "llvm-pie.elf") != audits["workers_agc"]:
            raise ValueError("workers AGC call redirection changed")
    elif kernel == "hip_reduction":
        validate_reduction_compiler((provenance / "compiler-receipt.json").read_bytes(), (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hip_routes":
        validate_routes_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                 (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hip_device_copy":
        validate_device_copy_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                  (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hip_memory_fill":
        validate_memory_fill_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                  (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hip_generic":
        validate_generic_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                  (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hip_matmul":
        validate_matmul_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                 (provenance / "routes-combined.o").read_bytes(), source_root)
    elif kernel == "hipify":
        validate_hipify_compiler((provenance / "compiler-receipt.json").read_bytes(),
                                 (provenance / "routes-combined.o").read_bytes(), source_root)
    # Reinspect actual linked bytes rather than accepting stale audit booleans.
    from probe_host_registration import audit_image, load_regions
    from build_native_vector_title import patch_startup, patch_linker, audit_startup
    symbols = json.loads((provenance / "symbols.json").read_text())
    elf = (provenance / "eboot.elf").read_bytes()
    relation = audits["fself_roundtrip"]
    expected = {"signed_sha256": hashlib.sha256(signed).hexdigest(),
                "elf_sha256": hashlib.sha256(elf).hexdigest(),
                "raw_pie_sha256": binding["llvm-pie.elf"],
                "app_crt_sha256": binding["app_crt.cpp"],
                "linker_sha256": binding["ps5-pie.ld"]}
    if any(relation.get(k) != v for k, v in expected.items()):
        raise ValueError("genuine HIP vector signed/audited digest relation mismatch")
    omitted = relation.get("omitted_note", {})
    offset = omitted.get("offset")
    if type(offset) is not int or not 0 <= offset <= len(elf) - 24:
        raise ValueError("genuine HIP vector omitted-note relation missing")
    reconstruction = elf[:offset] + bytes(24)
    if any(relation.get(k) != v for k, v in audit_roundtrip(elf, reconstruction).items()):
        raise ValueError("genuine HIP vector exact FSELF reconstruction relation mismatch")
    validate_fself(signed, elf)
    reader, _, _ = load_regions(provenance / "eboot.elf")
    bundle, _ = reader(symbols["__ps5hip_bundle_begin"], symbols["__ps5hip_bundle_end"] - symbols["__ps5hip_bundle_begin"])
    if hashlib.sha256(bundle).hexdigest() != selected["bundle_sha256"]:
        raise ValueError("genuine HIP vector linked bundle changed")
    for key, name, native in (("raw_pie", "llvm-pie.elf", False), ("converted_eboot", "eboot.elf", True)):
        if kernel=="hip_multi_query":
            from native_multi_query import audit_modules
            from fat_binary_fixture_test import section
            actual=audit_modules(provenance/name,symbols,[section((provenance/("multi-"+m+".o")).read_bytes(),b".hip_fatbin") for m in "ab"],(provenance/"startup-disassembly.stdout").read_text(),native=native)
        else:
            actual=audit_image(provenance/name,symbols,bundle,native=native)
        if actual != audits[key]:
            raise ValueError("genuine HIP vector recomputed image audit differs")
    if audit_startup((provenance / "startup-disassembly.stdout").read_text(), provenance / "llvm-pie.elf", kernel) != audits["startup"]:
        raise ValueError("genuine HIP vector startup audit differs")
    native_expected = json.loads((source_root / "evidence/offline-lds-native-readiness.json").read_text())["profiles"][0]["native_tool_source_sha256"]
    for name, patch in (("app_crt.cpp", patch_startup), ("ps5-pie.ld", patch_linker)):
        original = provenance / ("original-" + name)
        patched = patch(original.read_text(), kernel)
        if (hashlib.sha256(original.read_bytes()).hexdigest() != native_expected[name] or
                patched != (provenance / name).read_text()):
            raise ValueError("genuine HIP vector private native patch binding mismatch")
