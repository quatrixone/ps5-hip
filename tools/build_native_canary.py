#!/usr/bin/env python3
"""Offline native title link experiment. Does not install, launch or deploy a title."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=pathlib.Path, required=True)
    parser.add_argument("--target", choices=["gfx1013", "gfx1030"], required=True)
    parser.add_argument("--kernel", choices=["runtime", "indexed_store", "vector_add"], required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "dependencies.json").read_text())
    revision = lock["native_tools"]["revision"]
    sdk = ROOT / ".deps/sdk"
    if (sdk / ".ps5-sdk-revision").read_text().strip() != lock["sdk"]["revision"]:
        parser.error("prepare the pinned SDK first")
    version = subprocess.check_output(["llvm-config-18", "--version"], text=True).strip()
    if version != lock["llvm"]["version"]:
        parser.error("compiler version does not match dependency lock")
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
             "self_container.hpp", "app_crt.cpp", "ps5-pie.ld"]
    # Export exact blobs, never the reference checkout's uncommitted files.
    for name in files:
        (native / name).write_bytes(subprocess.check_output(
            ["git", "-C", str(args.native_source), "show", f"{revision}:tooling/native/{name}"]))
    for name in ("LICENSE", "NOTICE.md"):
        (native / name).write_bytes(subprocess.check_output(
            ["git", "-C", str(args.native_source), "show", f"{revision}:{name}"]))
    build = ROOT / "build/native" / f"{args.target}-{args.kernel}"
    build.mkdir(parents=True, exist_ok=True)
    tool = build / "ps5-native-tool"
    run("clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        *[native / name for name in files[:1] + ["elf_object.cpp", "sce_module_writer.cpp", "self_container.cpp"]],
        "-I", zlib / "root/include", zlib / "root/lib/libz.a", "-o", tool)
    compiler = ROOT / "tools/ps5-clangxx"
    objects = []
    sources = [native / "app_crt.cpp", ROOT / "tools/native_canary.cpp"]
    sources += [ROOT / "src" / name for name in
                ["code_object.cpp", "launch.cpp", "native_memory.cpp", "submission.cpp", "agc_backend.cpp"]]
    for source in sources:
        output = build / (source.stem + ".o")
        options = [f'-DPS5HIP_CANARY_TARGET="{args.target}"']
        if args.kernel == "vector_add":
            options.append("-DPS5HIP_CANARY_VECTOR_ADD=1")
        if args.kernel == "runtime":
            options.append("-DPS5HIP_RUNTIME_ONLY=1")
        run(compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
            "-fvisibility=hidden", "-I", ROOT / "include", *options, "-c", source, "-o", output)
        objects.append(output)
    stubs = []
    for library, declarations in {
        "libSceAgc": "int sceAgcInit(unsigned v) { return -1; }",
        "libSceAgcDriver": "int sceAgcDriverSubmitDcb(void *p) { return -1; } int sceAgcSuspendPoint(void) { return -1; }",
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
                                         ["libkernel.so", "libSceLibcInternal.so", "libSceNet.so"]])
    elf = build / "eboot.elf"
    run(tool, "link", "--in", pie, "--out", elf, "--stub-dir", sdk / "target/lib",
        "--stub", stubs[0], "--stub", stubs[1], "--file-name", "eboot.elf")
    candidate = build / "candidate"
    candidate.mkdir(exist_ok=True)
    run(tool, "self", "--sign", "--in", elf, "--out", candidate / "eboot.bin")
    run(tool, "self", "--inspect", "--file", candidate / "eboot.bin")
    artifacts = ["eboot.bin"]
    if args.kernel != "runtime":
        shutil.copyfile(ROOT / "build/canaries" / args.target / "canary.hsaco", candidate / "canary.hsaco")
        artifacts.append("canary.hsaco")
    receipt = {"hardware_validated": False, "target": args.target, "kernel": args.kernel,
               "native_tools_revision": revision, "sdk_revision": lock["sdk"]["revision"],
               "llvm_version": version,
               "host_zlib": zlib_receipt,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources[1:] + sorted((ROOT / "include/ps5hip").glob("*.h")) +
                                 [pathlib.Path(__file__).resolve(), ROOT / "tools/native_runtime.h", ROOT / "dependencies.json"]},
               "artifacts": {name: hashlib.sha256((candidate / name).read_bytes()).hexdigest()
                             for name in artifacts}}
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Built unqualified native candidate:", candidate)
    print("No title metadata, installation or console execution performed.")


if __name__ == "__main__":
    main()
