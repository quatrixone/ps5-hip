#!/usr/bin/env python3
"""Build the native public-HIP startup title from this checkout and locked sources."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from native_vector_title import audit_roundtrip

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
SOURCE_UNITS = (
    "hipamd/src/fixme.cpp", "rocclr/compiler/lib/utils/options.cpp",
    "rocclr/device/appprofile.cpp", "rocclr/device/blitcl.cpp",
    "rocclr/device/device.cpp", "rocclr/device/devkernel.cpp",
    "rocclr/device/devprogram.cpp", "rocclr/elf/elf.cpp",
    "rocclr/os/alloc.cpp", "rocclr/os/os.cpp", "rocclr/os/os_posix.cpp",
    "rocclr/platform/activity.cpp", "rocclr/platform/agent.cpp",
    "rocclr/platform/command.cpp", "rocclr/platform/commandqueue.cpp",
    "rocclr/platform/context.cpp", "rocclr/platform/interop_gl.cpp",
    "rocclr/platform/kernel.cpp", "rocclr/platform/memory.cpp",
    "rocclr/platform/ndrange.cpp", "rocclr/platform/program.cpp",
    "rocclr/platform/runtime.cpp", "rocclr/thread/monitor.cpp",
    "rocclr/thread/semaphore.cpp", "rocclr/thread/thread.cpp",
    "rocclr/utils/debug.cpp", "rocclr/utils/flags.cpp",
)
PROJECT_UNITS = (
    "ports/clr/ps5-clr-tls.cpp", "ports/clr/ps5-device.cpp",
    "tools/public_hip_app_crt.cpp", "tools/clr_gpu/watchdog.cpp",
    "src/code_object.cpp", "src/fat_binary.cpp", "src/host_launch.cpp",
    "src/launch.cpp", "src/native_memory.cpp", "src/native_data.cpp",
    "src/native_launch.cpp", "src/submission.cpp", "src/hip_launch_profile.cpp",
    "src/agc_backend.cpp", "tools/native_shell_exit_probe.cpp",
    "tools/public_hip_bootstrap_console_title.cpp",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_inputs():
    names = {"dependencies.json", "elevation-inputs.json", "controller-inputs.json",
             "title.json", "assets/param.json", "assets/icon0.png",
             "tools/build_public_hip.py", "tools/prepare_clr_gpu_source.py",
             "tools/prepare_hip_admission.py", "tools/guard_native_hip_gl.py",
             "tools/native_runtime.h", "tools/native_shell_exit_probe.cpp",
             "tools/public_hip_app_crt.cpp", "tools/public_hip_bootstrap_console_title.cpp",
             "tools/ps5-clangxx", "patches/native-import-libraries.patch",
             "tests/native_import_libraries_test.py", "tests/public_hip_imports_test.py",
             "tools/public_hip_memory_console_title.cpp", "tools/prepare_public_hip_memory.py",
             "tools/public_hip_capture_gpu_console_title.cpp", "tools/public_hip_graph_gpu_fixture.h", "tools/public_hip_graph_gpu_console_title.cpp",
             "tools/public_hip_stream_gpu_fixture.h", "tools/public_hip_stream_console_title.cpp",
             "tools/public_hip_graph_fixture.h", "tools/public_hip_graph_console_title.cpp", "tools/public_hip_fill_console_title.cpp", "tools/public_hip_copy_console_title.cpp", "tools/prepare_public_hip_copy.py",
             "tools/prepare_public_hip_module.py", "tools/prepare_public_hip_symbols.py",
             "tools/prepare_public_hip_capture.py", "tools/prepare_public_hip_fill.py", "tools/prepare_public_hip_rect.py", "tools/prepare_public_hip_aot.py", "tools/prepare_public_hip_launch.py", "tools/public_hip_module_console_title.cpp", "tools/public_hip_launch_console_title.cpp",
             "kernels/thread_words.hip",
             "tools/build_canaries.py", "kernels/canary.hip"}
    for folder in ("src", "include", "ports/clr", "ports/hip", "tools/clr_gpu"):
        names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).rglob("*")
                     if path.is_file() and path.suffix in
                     (".cpp", ".h", ".hpp", ".inc", ".patch", ".json", ".ld"))
    return {name: sha(ROOT / name) for name in sorted(names)}


def run(output, label, argv):
    process = subprocess.run([str(arg) for arg in argv], cwd=ROOT, capture_output=True,
                             text=True, timeout=300,
                             env=dict(os.environ, DOTNET_ROLL_FORWARD="Major",
                                      DOTNET_CLI_TELEMETRY_OPTOUT="1"))
    (output / (label + ".stdout")).write_text(process.stdout)
    (output / (label + ".stderr")).write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"{label} failed ({process.returncode}): {process.stderr[-3000:]}")


def source_tree(output, fixture="bootstrap"):
    base = output / "source-base"
    public = output / "source-public"
    if not (base / "receipt.json").exists():
        if base.exists():
            raise RuntimeError(f"incomplete source preparation: {base}")
        run(output, "prepare-clr", [sys.executable, ROOT / "tools/prepare_clr_gpu_source.py",
                                    "--output", base])
    if not (public / "public-hip-preparation.json").exists():
        if public.exists():
            raise RuntimeError(f"incomplete public HIP preparation: {public}")
        run(output, "prepare-public", [sys.executable, ROOT / "tools/prepare_hip_admission.py",
                                       "--base-source", base / "source",
                                       "--hip-source", DEPS / "sources/clr", "--output", public])
    correlation = output / "hip-correlation.sha256"
    if not correlation.exists():
        patch = ROOT / "ports/clr/ps5-hip-correlation.patch"
        run(output, "correlation", ["patch", "--batch", "--forward", "-p1",
                                    "-d", public, "-i", patch])
        correlation.write_text(sha(public / "hipamd/src/hip_prof_api.h") + "\n")
    if correlation.read_text().strip() != sha(public / "hipamd/src/hip_prof_api.h"):
        raise RuntimeError("HIP correlation source changed")
    guard = output / "hip-gl-guard.sha256"
    if not guard.exists():
        run(output, "guard-hip-gl", [sys.executable, ROOT / "tools/guard_native_hip_gl.py", public])
        guard.write_text(sha(public / "hipamd/src/hip_gl.cpp") + "\n")
    if guard.read_text().strip() != sha(public / "hipamd/src/hip_gl.cpp"):
        raise RuntimeError("guarded HIP GL source changed")
    run(output, "prepare-module", [sys.executable, ROOT / "tools/prepare_public_hip_aot.py", public])
    run(output, "prepare-symbols", [sys.executable, ROOT / "tools/prepare_public_hip_symbols.py", public])
    run(output, "prepare-rect", [sys.executable, ROOT / "tools/prepare_public_hip_rect.py", public])
    run(output, "prepare-launch", [sys.executable, ROOT / "tools/prepare_public_hip_launch.py", public])
    run(output, "prepare-capture", [sys.executable, ROOT / "tools/prepare_public_hip_capture.py", public])
    return public


def compiler_flags(source):
    paths = [
        "opencl/khronos/headers/opencl2.2/CL", "opencl/khronos/headers/opencl2.2",
        "opencl/khronos/headers", "opencl/khronos", "opencl", "opencl/amdocl",
        "rocclr", "rocclr/include", "rocclr/compiler/lib",
        "rocclr/compiler/lib/include", "rocclr/compiler/lib/backends/common",
        "rocclr/device", "rocclr/elf", "rocclr/platform", "hipamd/include",
        "hipamd/src",
    ]
    includes = [source / name for name in paths]
    includes += [DEPS / "sources/hip-headers/include", ROOT / "ports/hip", ROOT / "include",
                 ROOT / "ports/clr", ROOT / "tools", DEPS / "elevation/client"]
    if not all(path.is_dir() for path in includes):
        raise RuntimeError("native compiler include path is missing")
    defines = [
        "LITTLEENDIAN_CPU", "HAVE_CL2_HPP", "OPENCL_MAJOR=2", "OPENCL_MINOR=1",
        "OPENCL_C_MAJOR=2", "OPENCL_C_MINOR=0", "CL_TARGET_OPENCL_VERSION=220",
        "CL_USE_DEPRECATED_OPENCL_1_0_APIS", "CL_USE_DEPRECATED_OPENCL_1_1_APIS",
        "CL_USE_DEPRECATED_OPENCL_1_2_APIS", "CL_USE_DEPRECATED_OPENCL_2_0_APIS",
        "__HIP_PLATFORM_AMD__", "PS5HIP_CLR_PLATFORM", "PS5HIP_CLR_NO_GRAPHICS",
        "WITH_PS5_DEVICE", "PS5HIP_WITH_ELEVATION=1", "PS5HIP_SHELL_EXIT_PROBE=1",
        "PS5HIP_CLR_GPU_TITLE=1", "PS5HIP_PUBLIC_HIP", "DISABLE_DIRECT_DISPATCH",
    ]
    return [ROOT / "tools/ps5-clangxx", *["-I" + str(p) for p in includes],
            "-std=c++20", "-O0", "-fPIC", "-pthread", "-ferror-limit=0",
            "-ffunction-sections", "-fdata-sections", *["-D" + d for d in defines]]


def compile_units(output, source, fixture="bootstrap"):
    object_dir = output / "objects"
    object_dir.mkdir(exist_ok=True)
    units = [source / name for name in SOURCE_UNITS]
    units += [ROOT / name for name in PROJECT_UNITS]
    if fixture != "bootstrap":
        units[units.index(ROOT / "tools/public_hip_bootstrap_console_title.cpp")] = ROOT / ("tools/public_hip_" + fixture + "_console_title.cpp")
    units += [DEPS / "elevation/client/elevation.cpp"]
    hip = sorted((source / "hipamd/src").glob("*.cpp"))
    hip = [path for path in hip if path.stem not in ("fixme", "hip_runtime")]
    if len(units) != 44 or len(hip) != 30:
        raise RuntimeError(f"expected 44 platform and 30 HIP units; got {len(units)} and {len(hip)}")
    units += hip
    flags = compiler_flags(source)

    def compile_one(pair):
        index, path = pair
        obj = object_dir / (f"{index:02d}-" + path.stem + ".o")
        extra = (["-include", "hip/hip_runtime.h", "-include", "hip/hip_deprecated.h",
                  "-include", "hip/amd_detail/amd_hip_gl_interop.h"]
                 if path.stem in ("hip_api_trace", "hip_table_interface",
                                  "hip_table_interface_c") else [])
        run(output, f"compile-{index:02d}", [*flags, *extra, "-c", path, "-o", obj])
        return {"source": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "source_sha256": sha(path), "object": str(obj.relative_to(output)),
                "object_sha256": sha(obj)}

    # Fail quickly on the selected client before spending time on the upstream closure.
    title_index = units.index(ROOT / ("tools/public_hip_" + fixture + "_console_title.cpp"))
    title_record = compile_one((title_index, units[title_index]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(compile_one, ((index, path) for index, path in enumerate(units)
                                              if index != title_index)))
    records.insert(title_index, title_record)
    if len(records) != 74 or len({record["object"] for record in records}) != 74:
        raise RuntimeError("not a clean 74-object closure")
    return records


def native_stubs(output):
    names = {"libSceAgc": 'extern "C" {int sceAgcInit(unsigned){return -1;} int sceAgcSuspendPoint(){return -1;}}\n',
             "libSceAgcDriver": 'extern "C" {int sceAgcDriverSubmitDcb(void*){return -1;}}\n'}
    for name, text in names.items():
        source = output / (name + ".cpp")
        source.write_text(text)
        obj = output / (name + ".o")
        run(output, "stub-compile-" + name,
            [ROOT / "tools/ps5-clangxx", "-fPIC", "-c", source, "-o", obj])
        run(output, "stub-link-" + name,
            ["ld.lld-18", "-shared", "-soname", name + ".prx", obj,
             "-o", output / (name + ".so")])


def link_title(output, units, fixture="bootstrap"):
    sdk = DEPS / "sdk/target/lib"
    static = ["libc++.a", "libc++abi.a", "libunwind.a", "libc.a",
              "libps5platform.a", "libdl.a", "libpthread.a"]
    imports = [sdk / (name + ".so") for name in
               ("libkernel", "libSceLibcInternal", "libSceNet", "libSceSystemService")]
    imports += [output / "libSceAgc.so", output / "libSceAgcDriver.so"]
    (output / "symbols.map").write_text("{ local: *; };\n")
    unwind = [argument for section in ("eh_frame", "eh_frame_hdr")
              for argument in ("--defsym", "__" + section + "_start=ADDR(." + section + ")",
                               "--defsym", "__" + section + "_end=ADDR(." + section + ")+SIZEOF(." + section + ")")]
    command = ["ld.lld-18", "-pie", "--no-undefined", "--gc-sections", "-T",
               ROOT / "ports/clr/ps5-pie.ld", "--eh-frame-hdr", *unwind,
               "-z", "max-page-size=0x4000", "--hash-style=gnu", "--version-script",
               output / "symbols.map", "--exclude-libs=ALL", "-L", sdk, "-e", "_start",
               "-o", output / "title.elf", *[output / item["object"] for item in units],
               "--start-group", *[sdk / name for name in static], "--end-group",
               "--as-needed", *imports]
    run(output, "link", command)
    symbols = subprocess.check_output(["llvm-readelf-18", "-Ws", str(output / "title.elf")], text=True)
    required = ["hipInit", "hipGetDeviceCount", "ps5hip_clr_admit_device"]
    if fixture in ("memory", "copy", "launch", "fill", "stream", "graph_gpu", "capture_gpu"):
        required += ["hipMalloc", "hipFree"]
    if fixture in ("copy", "launch", "fill", "stream", "graph_gpu", "capture_gpu"):
        required += ["hipMemcpy"]
    if fixture in ("module", "launch", "stream", "graph_gpu", "capture_gpu"):
        required += ["hipModuleLoad", "hipModuleGetFunction", "hipModuleUnload"]
    if fixture == "launch":
        required += ["hipModuleLaunchKernel", "hipDeviceSynchronize"]
    if fixture == "stream":
        required += ["hipModuleLaunchKernel", "hipMemcpyAsync", "hipMemset", "hipStreamCreateWithFlags",
                     "hipStreamWaitEvent", "hipStreamSynchronize", "hipStreamQuery", "hipStreamDestroy",
                     "hipEventCreateWithFlags", "hipEventRecord", "hipEventSynchronize", "hipEventQuery",
                     "hipEventDestroy", "hipLaunchHostFunc"]
    if fixture == "fill":
        required += ["hipMemset", "hipMemsetD16", "hipMemsetD32Async", "hipMemsetAsync", "hipDeviceSynchronize"]
    if fixture in ("graph_gpu", "capture_gpu"):
        required += ["hipGraphAddKernelNode", "hipGraphAddHostNode", "hipGraphInstantiate", "hipGraphLaunch",
                     "hipGraphExecKernelNodeSetParams", "hipGraphNodeSetEnabled", "hipGraphDestroy", "hipGraphExecDestroy",
                     "hipStreamCreateWithFlags", "hipStreamSynchronize", "hipStreamDestroy"]
    if fixture == "capture_gpu":
        required += ["hipModuleLaunchKernel", "hipStreamBeginCapture", "hipStreamEndCapture", "hipLaunchHostFunc"]
    if fixture == "graph":
        required += ["hipGraphInstantiate", "hipGraphLaunch", "hipGraphExecHostNodeSetParams", "hipStreamBeginCapture", "hipStreamEndCapture", "hipGraphDestroy", "hipGraphExecDestroy"]
    for name in required:
        if name not in symbols:
            raise RuntimeError(f"missing public bootstrap symbol {name}")


def native_tool(output):
    upstream = DEPS / "sources/native-tools/tooling/native"
    vendor = output / "native-tool-source"
    if vendor.exists():
        raise RuntimeError("native converter source output already exists; use a fresh build")
    shutil.copytree(upstream, vendor)
    run(output, "native-import-patch", ["patch", "--batch", "--fuzz=0", "-p1", "-d", vendor,
                                       "-i", ROOT / "patches/native-import-libraries.patch"])
    zlib = DEPS / "host-zlib-1.3.2/root"
    tool = output / "ps5-native-tool"
    sources = [vendor / name for name in ("native_app_builder.cpp", "self_container.cpp",
                                         "elf_object.cpp", "sce_module_writer.cpp")]
    run(output, "native-tool", ["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra",
                                "-Werror", "-I", zlib / "include", *sources,
                                zlib / "lib/libz.a", "-o", tool])
    return tool


def clean_room_libc(output, tool):
    vendor = DEPS / "sources/native-tools/tooling/native"
    builder = output / "libc-builder"
    run(output, "libc-builder", ["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra",
                                 "-Werror", vendor / "libc_builder.cpp", "-o", builder])
    raw = output / "libc.raw.elf"
    signed = output / "libc.prx"
    run(output, "libc-raw", [builder, vendor / "runtime/api-surface.txt",
                             vendor / "runtime/imports.txt", raw])
    run(output, "libc-sign", [tool, "self", "--sign", "--in", raw, "--out", signed])
    expected = (DEPS / "sources/native-tools/runtime/libc.prx.sha256").read_text().split()[0]
    if sha(signed) != expected:
        raise RuntimeError("clean-room libc differs from locked native-tools digest")
    return signed


def convert_title(output, tool, fixture="bootstrap"):
    sdk = DEPS / "sdk/target/lib"
    converted = output / "eboot.elf"
    signed = output / "eboot.bin"
    extracted = output / "extracted.elf"
    run(output, "convert", [tool, "link", "--in", output / "title.elf", "--out", converted,
                            "--stub-dir", sdk, "--stub", output / "libSceAgc.so",
                            "--stub", output / "libSceAgcDriver.so", "--file-name", "eboot.elf",
                            "--import-library", "libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil"])
    run(output, "verify-sce-imports", [sys.executable, ROOT / "tests/public_hip_imports_test.py", converted,
                                     *(["--no-gpu-imports"] if fixture in ("module", "fill", "copy", "graph") else [])])
    run(output, "sign", [tool, "self", "--sign", "--in", converted, "--out", signed])
    run(output, "extract", [tool, "self", "--extract", "--file", signed,
                            "--out", extracted])
    relation = audit_roundtrip(converted.read_bytes(), extracted.read_bytes())
    if not relation["matches_except_unmapped_sie_note"]:
        raise RuntimeError("FSELF roundtrip differs outside the unmapped SIE note")
    return signed, relation


def inventory(folder):
    if any(path.is_symlink() for path in folder.rglob("*")):
        raise RuntimeError("package contains a symlink")
    return {path.relative_to(folder).as_posix(): sha(path) for path in folder.rglob("*")
            if path.is_file()}


def validate_results_directory(results_dir):
    parts = results_dir[1:].split("/") if results_dir.startswith("/") else []
    if (not parts or not 2 <= len(results_dir) <= 240 or results_dir.endswith("/") or
            any(part in ("", ".", "..") for part in parts) or
            any(not 32 <= ord(char) <= 126 or char == "\\" for char in results_dir)):
        raise ValueError("results directory must be a bounded absolute console path")


def package_title(output, image, libc, results_dir, code_object=None):
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    validate_results_directory(results_dir)
    stage = output / "package"
    app = stage / "app"
    sources = {
        "eboot.bin": image,
        "lapy.elf": DEPS / "elevation/lapy/build" / ("owned_root_helper-" + title) / "lapy.elf",
        "sce_module/libc.prx": libc,
        "sce_sys/param.json": ROOT / "assets/param.json",
        "sce_sys/icon0.png": ROOT / "assets/icon0.png",
    }
    if code_object is not None:
        sources[code_object.name] = code_object
    if app.exists():
        expected = {name: sha(source) for name, source in sources.items()}
        expected["results-dir.txt"] = hashlib.sha256(results_dir.encode()).hexdigest()
        if inventory(app) != expected:
            raise RuntimeError("existing package stage differs from locked inputs")
    else:
        for name, source in sources.items():
            destination = app / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (app / "results-dir.txt").write_text(results_dir)
    if json.loads((app / "sce_sys/param.json").read_text())["titleId"] != title:
        raise RuntimeError("package title identity differs from title.json")
    before = inventory(app)
    version = json.loads((ROOT / "dependencies.json").read_text())["ufs2tool"]["revision"]
    ufs = DEPS / ("ufs2tool-" + version) / "bin/UFS2Tool.dll"
    command = ["dotnet", ufs]
    with tempfile.TemporaryDirectory(prefix="pack-", dir=stage) as temporary:
        temporary = Path(temporary)
        packed = temporary / (title + ".ffpkg")
        extracted = temporary / "extracted"
        run(output, "makefs", [*command, "makefs", "-S", "4096", "-b", "20%", "-t", "ffs",
                               "-o", "version=2,bsize=32768,fsize=4096,minfree=0,softupdates=0,optimization=space",
                               packed, app])
        run(output, "fsck", [*command, "fsck_ufs", "-f", "-n", packed])
        run(output, "package-extract", [*command, "extract", packed, extracted])
        if inventory(extracted) != before:
            raise RuntimeError("UFS extraction inventory differs from staged title")
        data = packed.read_bytes()
        if data[0x1055c:0x10560] != bytes.fromhex("19015419"):
            raise RuntimeError("package format marker differs")
        final = stage / (title + ".ffpkg")
        with final.open("xb") as stream:
            stream.write(data)
    return final, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/public-hip")
    parser.add_argument("--fixture", choices=("bootstrap", "memory", "copy", "module", "launch", "fill", "graph", "stream", "graph_gpu", "capture_gpu"), default="bootstrap")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "/user/homebrew/ps5-hip"))
    args = parser.parse_args()
    validate_results_directory(args.results_dir)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "build").resolve()) and not output.is_relative_to((ROOT / ".local").resolve()):
        parser.error("output must stay within build/ or .local/")
    output.mkdir(parents=True, exist_ok=True)
    cached = output / "link-receipt.json"
    if cached.exists() and json.loads(cached.read_text()).get("fixture", "bootstrap") != args.fixture:
        raise RuntimeError("cached fixture differs; use a fresh output")
    source = source_tree(output, args.fixture)
    if cached.exists():
        previous = json.loads(cached.read_text())
        if previous.get("fixture", "bootstrap") != args.fixture:
            raise RuntimeError("cached fixture differs; use a fresh output")
        units = previous["units"]
        if (len(units) != 74 or previous["title_elf_sha256"] != sha(output / "title.elf") or
                any(sha(ROOT / unit["source"]) != unit["source_sha256"] or
                    sha(output / unit["object"]) != unit["object_sha256"] for unit in units)):
            raise RuntimeError("cached native link or its source changed; use a fresh output")
        if "package_sha256" in previous:
            if (previous.get("input_sha256") != build_inputs() or
                    previous.get("results_dir") != args.results_dir or
                    sha(output / "package" / (json.loads((ROOT / "title.json").read_text())["title_id"] + ".ffpkg")) != previous["package_sha256"] or
                    inventory(output / "package/app") != previous["package_files"]):
                raise RuntimeError("cached package differs from current inputs; use a fresh output")
            print(cached)
            return
    else:
        if args.fixture in ("module", "launch", "stream", "graph_gpu", "capture_gpu"):
            run(output, "build-code-object", [sys.executable, ROOT / "tools/build_canaries.py",
                "--fixture", "thread_words" if args.fixture in ("launch", "stream", "graph_gpu", "capture_gpu") else "canary",
                "--output", output / "canaries"])
        native_stubs(output)
        units = compile_units(output, source, args.fixture)
        link_title(output, units, args.fixture)
    tool = native_tool(output)
    libc = clean_room_libc(output, tool)
    image, roundtrip = convert_title(output, tool, args.fixture)
    package, files = package_title(output, image, libc, args.results_dir,
                                   output / "canaries/gfx1013/canary.hsaco" if args.fixture == "module" else
                                   output / "canaries/gfx1013/thread_words/thread_words.hsaco" if args.fixture in ("launch", "stream", "graph_gpu", "capture_gpu") else None)
    receipt = {"classification": "offline-public-hip-native-link", "source_commit":
               subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "fixture": args.fixture, "units": units, "object_count": len(units), "title_elf_sha256": sha(output / "title.elf"),
               "fself_sha256": sha(image), "fself_roundtrip": roundtrip,
               "package_sha256": sha(package), "package_files": files,
               "input_sha256": build_inputs(), "results_dir": args.results_dir,
               "native_executed": False}
    (output / "link-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output / "link-receipt.json")


if __name__ == "__main__":
    main()
