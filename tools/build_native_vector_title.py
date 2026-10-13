#!/usr/bin/env python3
"""Offline title build for the one genuine HIP vector client; never deploys/runs it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fat_binary_fixture_test import section
from native_import_libraries_test import read_imports, nid
from probe_host_registration import audit_image, load_regions
import struct
from native_vector_title import PROFILE, PROVENANCE_FILES, validate_build, audit_roundtrip


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch_startup(source):
    declaration = 'extern "C" void ps5hip_install_shell_exit_probe();'
    call = '    ps5hip_install_shell_exit_probe();\n#endif\n    _init();'
    require(source.count(declaration) == 1 and source.count(call) == 1, "pinned shell startup changed")
    return source.replace(declaration, declaration + '\n#ifdef PS5HIP_VECTOR_TITLE\nextern "C" void ps5hip_install_vector_title_finalizer();\n#endif').replace(call,
        '    ps5hip_install_shell_exit_probe();\n#endif\n#ifdef PS5HIP_VECTOR_TITLE\n    ps5hip_install_vector_title_finalizer();\n#endif\n    _init();')


def patch_linker(source):
    ro = '    .rodata : { *(.rodata .rodata.*) } : ro'
    rw = '        *(.data.rel.ro .data.rel.ro.*)'
    require(source.count(ro) == 1 and source.count(rw) == 1, "native linker source changed")
    return source.replace(ro, '''    .rodata : {
        HIDDEN(__ps5hip_names_begin = .);
        *(.rodata .rodata.*)
        HIDDEN(__ps5hip_names_end = .);
        . = ALIGN(4096);
        HIDDEN(__ps5hip_bundle_begin = .);
        KEEP(*(.hip_fatbin))
        HIDDEN(__ps5hip_bundle_end = .);
    } : ro''').replace(rw, '''        HIDDEN(__ps5hip_relro_begin = .);
        *(.data.rel.ro .data.rel.ro.*)
        . = ALIGN(8);
        HIDDEN(__ps5hip_wrappers_begin = .);
        KEEP(*(.hipFatBinSegment))
        HIDDEN(__ps5hip_wrappers_end = .);
        HIDDEN(__ps5hip_relro_end = .);''')


def audit_startup(disassembly, image=None):
    def body(name):
        match = re.search(r"^[0-9a-f]+ <" + re.escape(name) + r">:\n(.*?)(?=^[0-9a-f]+ <|\Z)", disassembly, re.M | re.S)
        require(match is not None, "missing linked startup symbol " + name)
        return match.group(1)
    start = body("_start")
    order = ["ps5hip_install_shell_exit_probe", "ps5hip_install_vector_title_finalizer", "_init", "main"]
    positions = [start.find("<" + name + ">") for name in order]
    require(min(positions) >= 0 and positions == sorted(set(positions)), "linked startup installation order changed")
    ctor, finalizer = body("__hip_module_ctor"), body("ps5hip_install_vector_title_finalizer")
    require("<__hipUnregisterFatBinary>" in body("__hip_module_dtor"), "genuine compiler unregister missing")
    result = {"install_order": ["shell", "vector_finalizer", "_init", "main"],
              "compiler_atexit_unregister": True, "native_executed": False}
    if image is None:
        require("<atexit>" in ctor and "<atexit>" in finalizer, "atexit registration missing")
        return result
    data = image.read_bytes()
    dynsym, dynstr = section(data, b".dynsym"), section(data, b".dynstr")
    indices = []
    for i in range(0, len(dynsym), 24):
        offset = struct.unpack_from("<I", dynsym, i)[0]
        if dynstr[offset:].split(b"\0", 1)[0] == b"atexit":
            indices.append(i // 24)
    require(len(indices) == 1, "atexit dynamic symbol missing/ambiguous")
    slots = []
    for name in (b".rela.dyn", b".rela.plt"):
        raw = section(data, name)
        for offset in range(0, len(raw), 24):
            address, info, _ = struct.unpack_from("<QQq", raw, offset)
            if info >> 32 == indices[0] and info & 0xffffffff in (6, 7):
                slots.append(address)
    require(1 <= len(slots) <= 2 and len(set(slots)) == len(slots), "atexit relocation missing/ambiguous")
    reader, _, _ = load_regions(image)
    branches = re.findall(r"\b(?:jmp|callq?)\s+0x([0-9a-f]+)", ctor)
    matched = []
    for target in branches:
        address = int(target, 16)
        raw, _ = reader(address, 6)
        if raw[:2] == b"\xff\x25" and address + 6 + struct.unpack_from("<i", raw, 2)[0] in slots:
            matched.append(address)
    require(len(matched) == 1 and "<__hip_module_dtor>" in ctor, "compiler destructor not passed to actual atexit PLT")
    got_calls = re.findall(r"callq?\s+\*[^\n]+#\s+0x([0-9a-f]+)", finalizer)
    require(sum(got_calls.count(format(slot, "x")) for slot in slots) == 1, "title finalizer not passed to actual atexit GOT")
    result.update(atexit_got=slots, atexit_plt=matched[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/native/gfx1013-hip_vector",
                        help="Fresh output under build/native; never overwrites a prior attempt")
    args = parser.parse_args()
    ref = args.reference_root.resolve()
    out = args.output.resolve()
    require(out.parent == (ROOT / "build/native").resolve() and out.name.startswith("gfx1013-hip_vector"), "invalid fresh native output")
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, *command):
        argv = [str(c) for c in command]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180)
        (out / (label + ".stdout")).write_text(result.stdout)
        (out / (label + ".stderr")).write_text(result.stderr)
        commands.append({"label": label, "argv": argv, "exit_code": result.returncode})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        require(result.returncode == 0, label + " failed: " + result.stderr[-3000:])
        return result.stdout

    deps = json.loads((ROOT / "dependencies.json").read_text())
    sdk = ROOT / ".deps/sdk"
    require((sdk / ".ps5-sdk-revision").read_text().strip() == deps["sdk"]["revision"], "SDK revision changed")
    require(run("llvm-version", "llvm-config-18", "--version").strip() == deps["llvm"]["version"], "LLVM version changed")
    fixture_root = ref / "results/hip-host-abi-spike/rocm-6.2.4"
    authority = ROOT / "evidence/offline-compatible-hip-headers.json"
    evidence = json.loads(authority.read_text())
    fixture_manifest = fixture_root / "compatible-emitted-abi.json"
    require(sha(fixture_manifest) == evidence["raw_file_hashes"][fixture_manifest.name]["sha256"], "fixture manifest changed")
    combined = fixture_root / "ps5-vector-add-gfx1013-device-wchar4.o"
    require(sha(combined) == PROFILE["combined_object_sha256"], "genuine combined object changed")
    selected = out / "ps5-combined.o"
    shutil.copyfile(combined, selected)
    bundle = section(selected.read_bytes(), b".hip_fatbin")
    require(hashlib.sha256(bundle).hexdigest() == PROFILE["bundle_sha256"], "embedded bundle changed")
    (out / "ps5.bundle").write_bytes(bundle)
    headers, include_dirs, header_inputs = {}, [], {}
    for key, archive, subdir in (("HIP", "HIP.tar.gz", "include"), ("clr", "clr.tar.gz", "hipamd/include")):
        package = fixture_root / archive
        require(sha(package) == evidence["input_sources"][key]["archive_sha256"], "header archive changed")
        with tarfile.open(package) as tar:
            prefix = tar.getmembers()[0].name.split("/")[0]
            directory = fixture_root / prefix
            for member in tar.getmembers():
                relative = member.name[len(prefix) + 1:]
                if member.isfile() and relative.startswith(subdir + "/"):
                    path = directory / relative
                    require(path.read_bytes() == tar.extractfile(member).read(), "pinned header tree changed")
                    headers[key + "/" + relative] = sha(path)
                    header_inputs[str(path)] = sha(path)
        include_dirs.append(directory / subdir)
    generated = fixture_root / "package-build/include/hip/hip_version.h"
    require(sha(generated) == evidence["raw_file_hashes"]["package-stage/include/hip/hip_version.h"]["sha256"], "generated HIP version changed")
    fixture_inputs = {str(p): sha(p) for p in (fixture_manifest, combined, generated,
                      fixture_root / "HIP.tar.gz", fixture_root / "clr.tar.gz")}
    fixture_inputs.update(header_inputs)
    tool_inputs = {n: {"path": str(Path(shutil.which(n)).resolve()), "sha256": sha(Path(shutil.which(n)))}
                   for n in ("clang++-18", "ld.lld-18", "llvm-config-18", "llvm-nm-18", "llvm-objdump-18")}
    sdk_wrapper_before = sha(sdk / "bin/prospero-clang++")
    include_dirs += [generated.parents[1], ROOT / "include"]
    native_source = ref / ".deps" / ("native-tools-" + deps["native_tools"]["revision"])
    native_authority = ROOT / "evidence/offline-lds-native-readiness.json"
    expected_native = json.loads(native_authority.read_text())["profiles"][0]["native_tool_source_sha256"]
    for name, digest in expected_native.items():
        require(sha(native_source / name) == digest, "pinned native export changed: " + name)
    native_before = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob("*") if p.is_file()}
    native = out / "native-tools"
    shutil.copytree(native_source, native)
    (native / "app_crt.cpp").write_text(patch_startup((native / "app_crt.cpp").read_text()))
    (native / "ps5-pie.ld").write_text(patch_linker((native / "ps5-pie.ld").read_text()))
    elevation_source = ref / ".deps/elevation"
    elevation_receipt = json.loads((elevation_source / "receipt.json").read_text())
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    require(title == "PPSA99014" and elevation_receipt["title_id"] == title and
            elevation_receipt["inputs"] == json.loads((ROOT / "elevation-inputs.json").read_text()) and
            elevation_receipt["sdk_revision"] == deps["sdk"]["revision"] and
            elevation_receipt["llvm_version"] == deps["llvm"]["version"], "elevation provenance mismatch")
    elevation = out / "elevation"
    for name, digest in elevation_receipt["source_sha256"].items():
        require(sha(elevation_source / name) == digest, "elevation source changed")
        destination = elevation / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(elevation_source / name, destination)
    helper = elevation_source / "lapy/build" / ("owned_root_helper-" + title) / "lapy.elf"
    require(sha(helper) == elevation_receipt["helper_sha256"], "elevation helper changed")
    include_dirs.append(elevation / "client")
    core = [ROOT / "src" / n for n in ("code_object.cpp", "fat_binary.cpp", "host_registration.cpp", "host_launch.cpp", "hip_runtime.cpp",
            "launch.cpp", "native_memory.cpp", "native_data.cpp", "native_launch.cpp", "submission.cpp", "agc_backend.cpp")]
    clients = [ROOT / n for n in ("tools/hip_vector_control.cpp", "tests/hip_vector_client.cpp",
               "tools/hip_vector_title.cpp", "tools/native_shell_exit_probe.cpp")]
    tracked = [*core, *clients, *sorted((ROOT / "include/ps5hip").glob("*.h")),
        *[ROOT / n for n in ("tools/build_native_vector_title.py", "tools/native_vector_title.py",
            "tools/probe_host_registration.py", "tools/native_runtime.h", "tools/ps5-clangxx",
            "tests/native_vector_title_test.py", "tests/fat_binary_fixture_test.py",
            "tests/native_import_libraries_test.py", "tests/native_agc_imports_test.py",
            "dependencies.json", "elevation-inputs.json", "title.json")],
        authority, native_authority]
    source_before = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    common = ["-O2", "-g", "-Wall", "-Wextra", "-Werror", "-fPIC", "-fvisibility=hidden",
              "-DPS5HIP_WITH_ELEVATION=1", "-DPS5HIP_VECTOR_TITLE=1", "-DPS5HIP_SHELL_EXIT_PROBE=1",
              *[part for directory in include_dirs for part in ("-I", directory)]]
    objects = [selected]
    for source in [*core, *clients, native / "app_crt.cpp", elevation / "client/elevation.cpp"]:
        obj = out / (source.stem + ".o")
        options = ["-std=c++20" if source.name in ("hip_vector_title.cpp", "elevation.cpp") else "-std=c++17"]
        if source.name == "hip_vector_client.cpp":
            options.append("-Dmain=ps5hip_vector_client_main")
        if source.name == "app_crt.cpp":
            options.append("-fno-inline")
        run("compile-" + source.stem, ROOT / "tools/ps5-clangxx", *options, *common, "-c", source, "-o", obj)
        objects.append(obj)
    stubs = []
    for library, declarations in {
        "libSceAgc": "int sceAgcInit(unsigned){return -1;} int sceAgcSuspendPoint(){return -1;}",
        "libSceAgcDriver": "int sceAgcDriverSubmitDcb(void*){return -1;}"}.items():
        source = out / (library + ".cpp")
        source.write_text('extern "C" {' + declarations + '}\n')
        obj, stub = out / (library + ".o"), out / (library + ".so")
        run("stub-compile-" + library, ROOT / "tools/ps5-clangxx", "-fPIC", "-c", source, "-o", obj)
        run("stub-link-" + library, "ld.lld-18", "-shared", "-soname", library + ".prx", obj, "-o", stub)
        stubs.append(stub)
    (out / "symbols.map").write_text("{ local: *; };\n")
    unwind = [p for n in ("eh_frame", "eh_frame_hdr") for p in
        ("--defsym", f"__{n}_start=ADDR(.{n})", "--defsym", f"__{n}_end=ADDR(.{n})+SIZEOF(.{n})")]
    libs = sdk / "target/lib"
    sdk_names = ("libc++.a", "libc++abi.a", "libunwind.a", "libc.a", "libps5platform.a",
                 "libkernel.so", "libSceLibcInternal.so", "libSceNet.so", "libSceSystemService.so")
    sdk_before = {n: sha(libs / n) for n in sdk_names}
    pie, elf = out / "llvm-pie.elf", out / "eboot.elf"
    run("pie-link", "ld.lld-18", "-pie", "-T", native / "ps5-pie.ld", "--eh-frame-hdr", *unwind,
        "-z", "max-page-size=0x4000", "--hash-style=gnu", "--version-script", out / "symbols.map",
        "--exclude-libs=ALL", "-L", libs, "-e", "_start", "-o", pie, *objects, *stubs,
        "--start-group", *[libs / n for n in sdk_names[:5]], "--end-group", "--as-needed",
        *[libs / n for n in sdk_names[5:]])
    zlib = ref / ".deps" / ("host-zlib-" + deps["host_zlib"]["version"])
    zlib_receipt = json.loads((zlib / "receipt.json").read_text())
    require(zlib_receipt["source_sha256"] == deps["host_zlib"]["sha256"] and
            zlib_receipt["llvm_version"] == deps["llvm"]["version"], "zlib provenance mismatch")
    for name, digest in zlib_receipt["artifacts"].items():
        require(sha(zlib / "root" / name) == digest, "zlib artifact changed")
    converter = out / "ps5-native-tool"
    run("converter-build", "clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        *[native / n for n in ("native_app_builder.cpp", "elf_object.cpp", "sce_module_writer.cpp", "self_container.cpp")],
        "-I", zlib / "root/include", zlib / "root/lib/libz.a", "-o", converter)
    run("native-convert", converter, "link", "--in", pie, "--out", elf, "--stub-dir", libs,
        "--stub", stubs[0], "--stub", stubs[1], "--file-name", "eboot.elf",
        "--import-library", "libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil")
    lines = run("native-symbols", "llvm-nm-18", "-n", pie).splitlines()
    symbols = {p[2]: int(p[0], 16) for line in lines if len(p := line.split()) == 3}
    (out / "symbols.json").write_text(json.dumps(symbols, indent=2) + "\n")
    audits = {"raw_pie": audit_image(pie, symbols, bundle),
              "converted_eboot": audit_image(elf, symbols, bundle, native=True)}
    startup = run("startup-disassembly", "llvm-objdump-18", "-d", "--no-show-raw-insn",
        "--disassemble-symbols=_start,__hip_module_ctor,__hip_module_dtor,ps5hip_install_vector_title_finalizer", pie)
    audits["startup"] = audit_startup(startup, pie)
    modules, libraries, imports = read_imports(elf)
    names = {p[1]: imports[nid(p[1])] for line in lines if len(p := line.split()) == 2 and p[0] == "U" and nid(p[1]) in imports}
    require(len(names) == len(imports) and not any(n.startswith(("hip", "__hip")) for n in names), "unexpected native import identity")
    (out / "native-imports.json").write_text(json.dumps({"modules": modules, "libraries": libraries, "symbols": imports, "names": names}, indent=2) + "\n")
    run("agc-provider-audit", "python3", ROOT / "tests/native_agc_imports_test.py", elf)
    candidate = out / "candidate"
    candidate.mkdir()
    run("fself-sign", converter, "self", "--sign", "--in", elf, "--out", candidate / "eboot.bin")
    run("fself-inspect", converter, "self", "--inspect", "--file", candidate / "eboot.bin")
    extracted = out / "fself-extracted.elf"
    run("fself-extract", converter, "self", "--extract", "--file", candidate / "eboot.bin", "--out", extracted)
    audits["fself_roundtrip"] = audit_roundtrip(elf.read_bytes(), extracted.read_bytes())
    audits["fself_roundtrip"].update(signed_sha256=sha(candidate / "eboot.bin"), raw_pie_sha256=sha(pie),
        app_crt_sha256=sha(native / "app_crt.cpp"), linker_sha256=sha(native / "ps5-pie.ld"))
    libc_builder = out / "libc-builder"
    run("libc-builder", "clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", native / "libc_builder.cpp", "-o", libc_builder)
    libc_raw = out / "libc.raw.elf"
    run("libc-raw", libc_builder, native / "runtime/api-surface.txt", native / "runtime/imports.txt", libc_raw)
    require(sha(libc_raw) == "8ee6e124993e1af26420cb455890fd002f5d6c7e78883c860ce45734e7d002bb", "loader companion raw mismatch")
    (candidate / "sce_module").mkdir()
    run("libc-sign", converter, "self", "--sign", "--in", libc_raw, "--out", candidate / "sce_module/libc.prx")
    require(sha(candidate / "sce_module/libc.prx") == "e6ff45d16adf687855cc3b33b0c8a4132b6504360b221e0a34c7e99fb3ba0036", "loader companion signed mismatch")
    shutil.copyfile(helper, candidate / "lapy.elf")
    provenance = out / "provenance"
    provenance.mkdir()
    for name in PROVENANCE_FILES:
        source = (native_source / name[len("original-"):]) if name.startswith("original-") else (
            native / name if name in ("app_crt.cpp", "ps5-pie.ld") else out / name)
        shutil.copyfile(source, provenance / name)
    require(native_before == {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob("*") if p.is_file()}, "shared native inputs changed")
    require(sdk_before == {n: sha(libs / n) for n in sdk_names}, "shared SDK artifacts changed")
    require(all(sha(Path(p)) == h for p, h in fixture_inputs.items()), "consumed HIP fixture/header inputs changed")
    require(all(sha(Path(item["path"])) == item["sha256"] for item in tool_inputs.values()) and
            sha(sdk / "bin/prospero-clang++") == sdk_wrapper_before, "consumed tool binaries changed")
    require(all(sha(zlib / "root" / n) == h for n, h in zlib_receipt["artifacts"].items()), "shared zlib artifacts changed")
    require(all(sha(elevation_source / n) == h for n, h in elevation_receipt["source_sha256"].items()) and sha(helper) == elevation_receipt["helper_sha256"], "shared elevation inputs changed")
    require(source_before == {str(p.relative_to(ROOT)): sha(p) for p in tracked}, "source changed during recipe")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_changes = subprocess.check_output(["git", "status", "--porcelain", "--", *source_before], cwd=ROOT, text=True)
    receipt = {"schema": 1, "classification": "Offline genuine HIP vector native title link/FSELF only",
        "target": "gfx1013", "kernel": "hip_vector", "shutdown_mode": "shell-request-probe",
        "hip_vector_profile": PROFILE, "vector_audits": audits, "source_sha256": source_before,
        "hardware_validated": False, "native_executed": False, "gpu_calls": 0,
        "source_commit": source_commit, "source_tree_clean": not bool(source_changes),
        "fixture_manifest_sha256": sha(fixture_manifest), "header_hashes": headers,
        "generated_version_sha256": sha(generated), "native_source_revision": deps["native_tools"]["revision"],
        "native_source_authority_sha256": sha(native_authority), "native_inputs_before_after": native_before,
        "elevation": elevation_receipt, "host_zlib": zlib_receipt, "sdk_inputs": sdk_before,
        "tools": tool_inputs, "fixture_inputs_before_after": fixture_inputs,
        "sdk_wrapper_sha256": sdk_wrapper_before,
        "hip_vector_provenance": {n: sha(provenance / n) for n in sorted(PROVENANCE_FILES)},
        "artifacts": {n: sha(candidate / n) for n in ("eboot.bin", "lapy.elf", "sce_module/libc.prx")},
        "raw_artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        "limits": ["Not executed or hardware qualified.",
                   "Installed SDK header/static archive internals are not fully inventoried; not hermetic.",
                   "Constructor stdout precedes redirection; main reports current registration without replay.",
                   "Static startup and atexit audits do not qualify native loader teardown."]}
    validate_build(receipt, ROOT, provenance, (candidate / "eboot.bin").read_bytes())
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(out), "native_executed": False, "gpu_calls": 0}))


if __name__ == "__main__":
    main()
