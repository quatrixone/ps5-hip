#!/usr/bin/env python3
"""Compile real device-only HIP objects and retain independent LLVM inspection.

No console traffic. These artifacts are compiler probes, not hardware-qualified
code. The headers/runtime and device libraries are deliberately not needed by the
two initial arithmetic fixtures.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode:
        result.check_returncode()
    return result.stdout


def isa_inventory(disassembly, expected):
    """Diagnostic instruction inventory; this is not a binary ISA verifier."""
    blocks = re.split(r"\n[0-9a-fA-F]+ <([^>]+)>:\n", disassembly)
    result = {}
    for name, body in zip(blocks[1::2], blocks[2::2]):
        if name not in expected:
            continue
        opcodes = re.findall(r"^\s+(\w+)\s+.*// [0-9A-F]+:", body, re.MULTILINE)
        if not opcodes:
            raise RuntimeError(f"No decoded instructions for {name}")
        result[name] = {"opcodes": sorted(set(opcodes)),
                        "flat_scratch_setup": "HW_REG_FLAT_SCR" in body}
    if set(result) != expected:
        raise RuntimeError("Disassembly kernel inventory mismatch")
    return result


def require_tool_versions(versions, expected):
    for name, text in versions.items():
        match = re.search(r"(?:version\s+|LLD\s+)(\d+\.\d+\.\d+)(?![\w.+-])", text)
        if not match or match[1] != expected:
            raise ValueError(f"{name} does not match pinned LLVM {expected}: {text.splitlines()[0] if text else 'no version'}")


def require_device_linker(trace, expected):
    commands = [shlex.split(line) for line in trace.splitlines() if line.lstrip().startswith('"')]
    linkers = [command[0] for command in commands if 'elf64_amdgpu' in command and '--whole-archive' in command]
    if len(linkers) != 1 or pathlib.Path(linkers[0]).resolve() != pathlib.Path(expected).resolve():
        raise ValueError(f"HIP device linker differs from pinned executable: {linkers}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "build/canaries")
    parser.add_argument("--fixture", choices=("canary", "resources", "thread_words"), default="canary")
    args = parser.parse_args()
    commands = {name: shutil.which(name) for name in
                ("clang++-18", "ld.lld-18", "llvm-readobj-18", "llvm-objdump-18")}
    if not all(commands.values()):
        parser.error("Required tools: " + ", ".join(k for k, v in commands.items() if not v))
    versions = {name: run([path, "--version"]) for name, path in commands.items()}
    expected_version = json.loads((ROOT / "dependencies.json").read_text())["llvm"]["version"]
    require_tool_versions(versions, expected_version)
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "kernels" / (args.fixture + ".hip")
    report = {"hardware_validated": False,
              "recipe_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tool_versions": {name: text.splitlines()[0] for name, text in versions.items()}, "profiles": []}
    for target in ("gfx1013", "gfx1030"):
        directory = args.output / target
        if args.fixture != "canary":
            directory = directory / args.fixture
        directory.mkdir(parents=True, exist_ok=True)
        code = directory / (args.fixture + ".hsaco")
        compile_args = [commands["clang++-18"], "-x", "hip", "--cuda-device-only",
                        "--no-gpu-bundle-output", "-B" + str(pathlib.Path(commands["ld.lld-18"]).resolve().parent),
                        "-nogpuinc", "-nogpulib", "--offload-arch=" + target,
                        "-mcode-object-version=5", "-mno-wavefrontsize64", "-O2",
                        "-std=c++17", "-fno-fast-math", "-c", str(source), "-o", str(code)]
        trace = subprocess.run(compile_args + ['-###'], text=True, capture_output=True, check=True).stderr
        require_device_linker(trace, commands["ld.lld-18"])
        (directory / "driver.txt").write_text(trace)
        run(compile_args)
        inspection = run([commands["llvm-readobj-18"],
            "--file-headers", "--program-headers", "--sections", "--symbols", "--relocations",
            "--notes", str(code)])
        (directory / "readobj.txt").write_text(inspection)
        expected = ({"thread_words"} if args.fixture == "thread_words" else
                    {"indexed_store", "vector_add"} if args.fixture == "canary" else
                    {"reduce_serial", "reduce_lds", "atomic_count", "private_array", "private_explicit", "register_spill"})
        actual = set(re.findall(r"^\s+\.name:\s+(\w+)\s*$", inspection, re.MULTILINE))
        if actual != expected:
            raise RuntimeError(f"Kernel inventory changed: expected {expected}, found {actual}")
        disassembly = run([commands["llvm-objdump-18"],
            "--disassemble", "--mcpu=" + target, str(code)])
        (directory / "disassembly.txt").write_text(disassembly)
        isa = isa_inventory(disassembly, expected)
        if args.fixture == "resources":
            if "s_barrier" not in isa["reduce_lds"]["opcodes"]:
                raise RuntimeError("LDS reduction lost barriers")
            if "global_atomic_add" not in isa["atomic_count"]["opcodes"]:
                raise RuntimeError("Atomic probe lost its atomic operation")
            for name in ("private_explicit", "register_spill"):
                opcodes = isa[name]["opcodes"]
                if isa[name]["flat_scratch_setup"] or any(op.startswith("scratch_") for op in opcodes):
                    raise RuntimeError(f"{name} still uses flat scratch")
                if not all(any(op.startswith(prefix) for op in opcodes)
                           for prefix in ("buffer_load_", "buffer_store_")):
                    raise RuntimeError(f"{name} did not exercise buffer scratch loads and stores")
        report["profiles"].append({"target": target, "compile_command": compile_args,
                                    "driver_sha256": hashlib.sha256(trace.encode()).hexdigest(),
                                    "flat_scratch_setup_present": "HW_REG_FLAT_SCR" in disassembly,
                                    "kernel_inventory": sorted(actual),
                                    "isa_inventory": isa,
                                    "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                                    "bytes": code.stat().st_size})
    report_name = "build.json" if args.fixture == "canary" else args.fixture + "-build.json"
    (args.output / report_name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
