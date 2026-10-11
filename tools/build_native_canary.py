#!/usr/bin/env python3
"""Offline native title link experiment. Does not install, launch or deploy a title."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
from build_canaries import require_device_linker, require_tool_versions

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def verified_device_object(root, target, version, wave_directory=None, fixture="canary", lds_directory=None):
    if fixture not in ("canary", "resources") or (wave_directory is not None and fixture != "canary"):
        raise ValueError("unsupported device fixture selection")
    if lds_directory is not None and (wave_directory is not None or fixture != "canary"):
        raise ValueError("LDS probes cannot be combined with other fixtures")
    probe = "lds" if lds_directory is not None else "wave"
    probe_directory = lds_directory if lds_directory is not None else wave_directory
    directory = probe_directory if probe_directory is not None else root / "build/canaries"
    report_name = "build.json" if fixture == "canary" else "resources-build.json"
    report = json.loads((directory / ("comparison.json" if probe_directory is not None else report_name)).read_text())
    inputs = [("recipe_sha256", root / "tools/build_canaries.py"),
              ("source_sha256", root / "kernels" / (fixture + ".hip"))]
    if probe_directory is not None:
        inputs = [("recipe_sha256", root / "tools" / ("probe_" + probe + "_compiler.py")),
                  ("shared_recipe_sha256", root / "tools/build_canaries.py"),
                  ("source_sha256", root / "kernels" / (probe + ".hip"))]
    for field, path in inputs:
        if report.get(field) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("device recipe or kernel changed; rebuild device objects")
    require_tool_versions(report["tool_versions"], version)
    profiles = ([p for p in report["cases"] if p["target"] == target and p["optimization"] == "-O2"]
                if probe_directory is not None else [p for p in report["profiles"] if p["target"] == target])
    if len(profiles) != 1:
        raise ValueError("missing or duplicate device target")
    profile = profiles[0]
    if lds_directory is not None:
        expected = {name: {"group_segment_fixed_size": 0 if name == "lds_dynamic" else 256,
                           "private_segment_fixed_size": 0, "kernarg_segment_size": 16 if name == "lds_static" else 20,
                           "wavefront_size": 32} for name in ("lds_static", "lds_dynamic", "lds_mixed")}
        actual = profile.get("profiles", {})
        if actual != expected or any(type(value) is not int for fields in actual.values() for value in fields.values()):
            raise ValueError("LDS probe resource profile changed")
    profile_directory = directory / (target + "-O2" if probe_directory is not None else target)
    if fixture == "resources":
        profile_directory /= fixture
    data = (profile_directory / (probe + ".hsaco" if probe_directory is not None else fixture + ".hsaco")).read_bytes()
    trace = (profile_directory / "driver.txt").read_bytes()
    if (hashlib.sha256(data).hexdigest() != profile["code_sha256"] or
            len(data) != profile["bytes"] or
            hashlib.sha256(trace).hexdigest() != profile.get("driver_sha256")):
        raise ValueError("device object or driver trace changed")
    linker = shutil.which("ld.lld-18")
    if not linker:
        raise ValueError("pinned device linker unavailable")
    require_device_linker(trace.decode(), linker)
    return data, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=pathlib.Path, required=True)
    parser.add_argument("--target", choices=["gfx1013", "gfx1030"], required=True)
    parser.add_argument("--kernel", choices=["runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "register_spill", "lds_static", "lds_dynamic", "lds_mixed"], required=True)
    parser.add_argument("--wave-probes", type=pathlib.Path, help="Verified probe_wave_compiler.py output; selects the O2 profile")
    parser.add_argument("--lds-probes", type=pathlib.Path, help="Verified probe_lds_compiler.py output; selects O2")
    parser.add_argument("--shell-exit-probe", action="store_true", help="Finite canary application finalization and shell closure")
    args = parser.parse_args()
    wave = args.kernel in ("wave_full", "wave_partial")
    if wave != (args.wave_probes is not None):
        parser.error("--wave-probes is required only for wave kernels")
    lds = args.kernel in ("lds_static", "lds_dynamic", "lds_mixed")
    if lds != (args.lds_probes is not None):
        parser.error("--lds-probes is required only for LDS kernels")
    lock = json.loads((ROOT / "dependencies.json").read_text())
    revision = lock["native_tools"]["revision"]
    sdk = ROOT / ".deps/sdk"
    if (sdk / ".ps5-sdk-revision").read_text().strip() != lock["sdk"]["revision"]:
        parser.error("prepare the pinned SDK first")
    version = subprocess.check_output(["llvm-config-18", "--version"], text=True).strip()
    if version != lock["llvm"]["version"]:
        parser.error("compiler version does not match dependency lock")
    fixture = "resources" if args.kernel == "register_spill" else "canary"
    device_data, device_report = (verified_device_object(ROOT, args.target, version, args.wave_probes, fixture, args.lds_probes)
                                  if args.kernel != "runtime" else (None, None))
    elevation = ROOT / ".deps/elevation"
    elevation_receipt = json.loads((elevation / "receipt.json").read_text())
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    if (elevation_receipt["inputs"] != json.loads((ROOT / "elevation-inputs.json").read_text()) or
            elevation_receipt["title_id"] != title or elevation_receipt["sdk_revision"] != lock["sdk"]["revision"] or
            elevation_receipt["llvm_version"] != version):
        parser.error("prepare the pinned elevation helper first")
    for name, expected in elevation_receipt["source_sha256"].items():
        if hashlib.sha256((elevation / name).read_bytes()).hexdigest() != expected:
            parser.error("elevation source hash mismatch")
    helper = elevation / "lapy/build" / ("owned_root_helper-" + title) / "lapy.elf"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != elevation_receipt["helper_sha256"]:
        parser.error("elevation helper hash mismatch")
    zlib = ROOT / ".deps" / ("host-zlib-" + lock["host_zlib"]["version"])
    zlib_receipt = json.loads((zlib / "receipt.json").read_text())
    if zlib_receipt["source_sha256"] != lock["host_zlib"]["sha256"] or zlib_receipt["llvm_version"] != version:
        parser.error("prepare the pinned host zlib first")
    for name, digest in zlib_receipt["artifacts"].items():
        if hashlib.sha256((zlib / "root" / name).read_bytes()).hexdigest() != digest:
            parser.error("host zlib artifact digest mismatch")
    native = ROOT / ".deps" / ("native-tools-" + revision)
    native.mkdir(parents=True, exist_ok=True)
    files = ["native_app_builder.cpp", "elf_object.cpp", "elf_object.hpp", "hash.hpp",
             "sce_module_writer.cpp", "sce_module_writer.hpp", "self_container.cpp",
             "self_container.hpp", "app_crt.cpp", "ps5-pie.ld", "libc_builder.cpp",
             "runtime/api-surface.txt", "runtime/imports.txt"]
    # Export exact blobs, never the reference checkout's uncommitted files.
    for name in files:
        (native / name).parent.mkdir(parents=True, exist_ok=True)
        (native / name).write_bytes(subprocess.check_output(
            ["git", "-C", str(args.native_source), "show", f"{revision}:tooling/native/{name}"]))
    for name in ("LICENSE", "NOTICE.md"):
        (native / name).write_bytes(subprocess.check_output(
            ["git", "-C", str(args.native_source), "show", f"{revision}:{name}"]))
    import_patch = ROOT / "patches/native-import-libraries.patch"
    run("patch", "--batch", "--fuzz=0", "-p1", "-d", native, "-i", import_patch)
    shell_patch = ROOT / "patches/native-shell-exit-probe.patch"
    if args.shell_exit_probe:
        run("patch", "--batch", "--fuzz=0", "-p1", "-d", native, "-i", shell_patch)
    build = ROOT / "build/native" / f"{args.target}-{args.kernel}"
    build.mkdir(parents=True, exist_ok=True)
    tool = build / "ps5-native-tool"
    run("clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        *[native / name for name in files[:1] + ["elf_object.cpp", "sce_module_writer.cpp", "self_container.cpp"]],
        "-I", zlib / "root/include", zlib / "root/lib/libz.a", "-o", tool)
    compiler = ROOT / "tools/ps5-clangxx"
    objects = []
    sources = [native / "app_crt.cpp", ROOT / "tools/native_canary.cpp", elevation / "client/elevation.cpp"]
    if args.kernel == "runtime" or args.shell_exit_probe:
        sources.append(ROOT / "tools" / ("native_shell_exit_probe.cpp" if args.shell_exit_probe else "native_shutdown_probe.cpp"))
    sources += [ROOT / "src" / name for name in
                ["code_object.cpp", "launch.cpp", "native_memory.cpp", "submission.cpp", "agc_backend.cpp"]]
    for source in sources:
        output = build / (source.stem + ".o")
        options = [f'-DPS5HIP_CANARY_TARGET="{args.target}"', "-DPS5HIP_WITH_ELEVATION=1", "-I", elevation / "client"]
        if args.kernel == "vector_add":
            options.append("-DPS5HIP_CANARY_VECTOR_ADD=1")
        if args.kernel == "vector_add_twice":
            options.append("-DPS5HIP_CANARY_VECTOR_ADD_TWICE=1")
        if wave or lds:
            options.append("-DPS5HIP_CANARY_" + args.kernel.upper() + "=1")
        if args.kernel == "register_spill":
            options.append("-DPS5HIP_CANARY_REGISTER_SPILL=1")
        if args.kernel == "runtime":
            options.append("-DPS5HIP_RUNTIME_ONLY=1")
        if args.shell_exit_probe:
            options.append("-DPS5HIP_SHELL_EXIT_PROBE=1")
        run(compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
            "-fvisibility=hidden", "-I", ROOT / "include", *options, "-c", source, "-o", output)
        objects.append(output)
    stubs = []
    for library, declarations in {
        "libSceAgc": "int sceAgcInit(unsigned v) { return -1; } int sceAgcSuspendPoint(void) { return -1; }",
        "libSceAgcDriver": "int sceAgcDriverSubmitDcb(void *p) { return -1; }",
    }.items():
        source = build / (library + ".cpp")
        # Host-link symbols only. These implementations are never staged.
        source.write_text('extern "C" { ' + declarations + ' }\n')
        obj, stub = build / (library + ".o"), build / (library + ".so")
        run(compiler, "-fPIC", "-c", source, "-o", obj)
        run("ld.lld-18", "-shared", "-soname", library + ".prx", obj, "-o", stub)
        stubs.append(stub)
    # Hide static runtime definitions from the native converter's import table.
    symbols = build / "symbols.map"
    symbols.write_text("{ local: *; };\n")
    pie = build / "llvm-pie.elf"
    unwind_symbols = []
    for section in ("eh_frame", "eh_frame_hdr"):
        unwind_symbols += ["--defsym", f"__{section}_start=ADDR(.{section})",
                           "--defsym", f"__{section}_end=ADDR(.{section})+SIZEOF(.{section})"]
    run("ld.lld-18", "-pie", "-T", native / "ps5-pie.ld", "--eh-frame-hdr",
        *unwind_symbols,
        "-z", "max-page-size=0x4000", "--hash-style=gnu", "--version-script", symbols,
        "--exclude-libs=ALL", "-L", sdk / "target/lib", "-e", "_start", "-o", pie, *objects, *stubs,
        "--start-group", *[sdk / "target/lib" / name for name in
                             ["libc++.a", "libc++abi.a", "libunwind.a", "libc.a", "libps5platform.a"]],
        "--end-group", "--as-needed", *[sdk / "target/lib" / name for name in
                                         ["libkernel.so", "libSceLibcInternal.so", "libSceNet.so"]],
        *([sdk / "target/lib/libSceSystemService.so"] if args.shell_exit_probe else []))
    elf = build / "eboot.elf"
    run(tool, "link", "--in", pie, "--out", elf, "--stub-dir", sdk / "target/lib",
        "--stub", stubs[0], "--stub", stubs[1], "--file-name", "eboot.elf",
        *(["--import-library", "libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil"]
          if args.shell_exit_probe else []))
    candidate = build / "candidate"
    candidate.mkdir(exist_ok=True)
    run(tool, "self", "--sign", "--in", elf, "--out", candidate / "eboot.bin")
    run(tool, "self", "--inspect", "--file", candidate / "eboot.bin")
    # The native loader requires its libc companion even with a static C++ runtime.
    # This emitter is independently authored; it is not a console/SDK runtime dump.
    libc_builder = build / "libc-builder"
    run("clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        native / "libc_builder.cpp", "-o", libc_builder)
    libc_raw = build / "libc.raw.elf"
    run(libc_builder, native / "runtime/api-surface.txt", native / "runtime/imports.txt", libc_raw)
    if hashlib.sha256(libc_raw.read_bytes()).hexdigest() != "8ee6e124993e1af26420cb455890fd002f5d6c7e78883c860ce45734e7d002bb":
        raise ValueError("clean-room loader companion raw hash mismatch")
    (candidate / "sce_module").mkdir(exist_ok=True)
    companion = candidate / "sce_module/libc.prx"
    run(tool, "self", "--sign", "--in", libc_raw, "--out", companion)
    run(tool, "self", "--inspect", "--file", companion)
    if hashlib.sha256(companion.read_bytes()).hexdigest() != "e6ff45d16adf687855cc3b33b0c8a4132b6504360b221e0a34c7e99fb3ba0036":
        raise ValueError("clean-room loader companion signed hash mismatch")
    shutil.copyfile(helper, candidate / "lapy.elf")
    artifacts = ["eboot.bin", "lapy.elf", "sce_module/libc.prx"]
    if args.kernel != "runtime":
        (candidate / "canary.hsaco").write_bytes(device_data)
        artifacts.append("canary.hsaco")
    receipt = {"hardware_validated": False, "target": args.target, "kernel": args.kernel,
               "shutdown_mode": "shell-request-probe" if args.shell_exit_probe else "platform-exit",
               "native_tools_revision": revision, "sdk_revision": lock["sdk"]["revision"],
               "native_tool_source_sha256": {name: hashlib.sha256((native / name).read_bytes()).hexdigest()
                                             for name in files},
               "llvm_version": version,
               "host_zlib": zlib_receipt, "elevation": elevation_receipt,
               "device_build": device_report,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources[1:] + sorted((ROOT / "include/ps5hip").glob("*.h")) +
                                 ([shell_patch] if args.shell_exit_probe else []) +
                                 [ROOT / "tools/build_canaries.py"] +
                                 ([ROOT / "tools" / ("probe_" + ("lds" if lds else "wave") + "_compiler.py")] if wave or lds else []) +
                                 ([ROOT / "tools/native_lds.h"] if lds else []) +
                                 ([ROOT / "kernels" / ("lds.hip" if lds else "wave.hip" if wave else fixture + ".hip")] if args.kernel != "runtime" else []) +
                                 [pathlib.Path(__file__).resolve(), import_patch, ROOT / "tools/native_runtime.h", ROOT / "dependencies.json",
                                  ROOT / "elevation-inputs.json", ROOT / "title.json", ROOT / "tools/prepare_elevation.py"] +
                                 [elevation / name for name in elevation_receipt["source_sha256"]]},
               "artifacts": {name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
                             for name in artifacts}}
    if args.kernel == "vector_add_twice":
        receipt["dispatches"] = 2
    if lds:
        receipt["lds"] = {"groups": 257, "static_bytes": 0 if args.kernel == "lds_dynamic" else 256,
                          "dynamic_bytes": 0 if args.kernel == "lds_static" else 1280,
                          "allocated_bytes": 1024 if args.kernel == "lds_static" else 2048,
                          "stride": 0 if args.kernel == "lds_static" else 5}
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Built unqualified native candidate:", candidate)
    print("No title metadata, installation or console execution performed.")


if __name__ == "__main__":
    main()
