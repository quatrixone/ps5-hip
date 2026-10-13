"""The one measured genuine-HIP title profile, distinct from handwritten canaries."""
import hashlib
import json
import struct

PROVENANCE_FILES = {"llvm-pie.elf", "eboot.elf", "symbols.json", "startup-disassembly.stdout",
                    "app_crt.cpp", "ps5-pie.ld", "original-app_crt.cpp", "original-ps5-pie.ld"}


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
    "compiler_receipt_sha256": "283c064b5af9d57df2014663b5407222649132567ec5271c8124894326ee6362",
    "dispatches": 8, "sequences": 2, "outputs_per_sequence": [128, 2, 8192, 6144],
    "data_logical_bytes": 65536, "interior_offset_bytes": 64,
    "geometry_records": 384, "geometry_logical": 273, "geometry_inactive": 111,
    "static_lds_bytes": 256, "private_segment_bytes": 212,
}


def profile_for(kernel):
    if kernel == "hip_vector":
        return PROFILE
    if kernel == "hip_routes":
        return ROUTES_PROFILE
    raise ValueError("unsupported genuine HIP title profile")


def provenance_files(kernel):
    profile_for(kernel)
    return PROVENANCE_FILES | ({"routes-combined.o", "compiler-receipt.json"} if kernel == "hip_routes" else set())


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
            not all(selected.get(n) for n in ("combined_object_sha256", "bundle_sha256", "code_sha256", "code_bytes")) or
            set(receipt.get("artifacts", {})) != {"eboot.bin", "lapy.elf", "sce_module/libc.prx"}):
        raise ValueError("genuine HIP vector title profile mismatch")
    sources = receipt.get("source_sha256", {})
    required = ["tools/build_native_vector_title.py", "tools/native_vector_title.py",
                "tools/hip_vector_title.cpp", "src/hip_runtime.cpp", "include/ps5hip/hip_runtime_control.h",
                "tools/native_shell_exit_probe.cpp", "tools/native_runtime.h", "evidence/offline-compatible-hip-headers.json"]
    required += (["tools/hip_vector_control.cpp", "tests/hip_vector_client.cpp"] if kernel == "hip_vector" else
                 ["tools/hip_routes_control.cpp", "tests/hip_routes.hip", "tools/probe_hip_routes.py", "docs/HIP_ROUTES_COMPILER.md"])
    for name in required:
        if sources.get(name) != hashlib.sha256((source_root / name).read_bytes()).hexdigest():
            raise ValueError("genuine HIP vector title source provenance mismatch: " + name)
    audits = receipt.get("vector_audits" if kernel == "hip_vector" else "routes_audits", {})
    if (set(audits) != {"raw_pie", "converted_eboot", "startup", "fself_roundtrip"} or
            any(audits[k].get("bundle_sha256") != selected["bundle_sha256"] for k in ("raw_pie", "converted_eboot")) or
            audits["startup"].get("install_order") != ["shell", "vector_finalizer", "_init", "main"] or
            audits["startup"].get("compiler_atexit_unregister") is not True or
            audits["fself_roundtrip"].get("matches_except_unmapped_sie_note") is not True):
        raise ValueError("genuine HIP vector title embedded bundle/lifecycle audit missing")
    if kernel == "hip_vector":
        authority = json.loads((source_root / "evidence/offline-compatible-hip-headers.json").read_text())
        for suffix, key in ((".o", "combined_object_sha256"), (".hsaco", "code_sha256")):
            if authority["raw_file_hashes"]["ps5-vector-add-gfx1013-device-wchar4" + suffix]["sha256"] != selected[key]:
                raise ValueError("genuine HIP vector fixture authority changed")


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
    if kernel == "hip_routes":
        validate_routes_compiler((provenance / "compiler-receipt.json").read_bytes(),
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
        if audit_image(provenance / name, symbols, bundle, native=native) != audits[key]:
            raise ValueError("genuine HIP vector recomputed image audit differs")
    if audit_startup((provenance / "startup-disassembly.stdout").read_text(), provenance / "llvm-pie.elf", kernel) != audits["startup"]:
        raise ValueError("genuine HIP vector startup audit differs")
    native_expected = json.loads((source_root / "evidence/offline-lds-native-readiness.json").read_text())["profiles"][0]["native_tool_source_sha256"]
    for name, patch in (("app_crt.cpp", patch_startup), ("ps5-pie.ld", patch_linker)):
        original = provenance / ("original-" + name)
        patched = patch(original.read_text(), kernel) if name == "app_crt.cpp" else patch(original.read_text())
        if (hashlib.sha256(original.read_bytes()).hexdigest() != native_expected[name] or
                patched != (provenance / name).read_text()):
            raise ValueError("genuine HIP vector private native patch binding mismatch")
