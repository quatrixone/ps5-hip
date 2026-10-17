"""Build the host HIP interpreter and compare three real decoder requests to the CPU oracle."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
import build_llm_executed_graph as provenance
from build_llm_full_model_reference import authenticate_full_model_reference

ROOT = Path(__file__).resolve().parents[1]
ABS_TOL, REL_TOL, L2_TOL = 2e-5, 2e-4, 2e-4
FILES = {f"run-{r}-pos-{p}-{name}.f32le": 128000 if name == "logits" else 8192
         for r, p in ((0, 0), (0, 1), (1, 0))
         for name in [*(f"layer-{i}" for i in range(22)), "norm", "logits"]}


def compare(raw, reference):
    raw, reference = Path(raw), Path(reference)
    if raw.is_symlink() or {p.name for p in raw.iterdir()} != set(FILES):
        raise ValueError("numerical output inventory")
    rows = []
    for name, size in FILES.items():
        paths = (raw / name, reference / name)
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("numerical output file")
        actual, expected = (p.read_bytes() for p in paths)
        if len(actual) != size or len(expected) != size:
            raise ValueError("numerical output size")
        a, b = (struct.unpack("<" + str(size // 4) + "f", x) for x in (actual, expected))
        if not all(math.isfinite(x) for x in (*a, *b)):
            raise ValueError("nonfinite numerical output")
        differences = [abs(x-y) for x, y in zip(a, b)]
        bad = sum(d > ABS_TOL + REL_TOL * abs(y) for d, y in zip(differences, b))
        relative = math.sqrt(sum(d*d for d in differences) / max(sum(y*y for y in b), 1e-30))
        rows.append({"file": name, "max_abs": max(differences), "outside_element_budget": bad,
                     "relative_l2": relative, "pass": bad == 0 and relative <= L2_TOL})
    for name in FILES:
        if name.startswith("run-1-") and (raw / name).read_bytes() != (raw / name.replace("run-1-", "run-0-")).read_bytes():
            raise ValueError("numerical reset differs")
    return rows


def status(rows):
    if len(rows) != len(FILES):
        raise ValueError("incomplete comparison")
    return ("PASS", 0) if all(row["pass"] for row in rows) else ("NUMERICAL_FAIL", 1)


def build(cpu, model, reference, output, same_input_oracle=False):
    cpu, model, reference, output = (Path(p).resolve() for p in (cpu, model, reference, output))
    if output.exists() or not output.is_relative_to(ROOT / "results"):
        raise ValueError("fresh results output required")
    reference_data, inputs = authenticate_full_model_reference(reference)
    inputs = dict(inputs)
    for path, digest in provenance.cpu_inputs(cpu, model).items():
        provenance.merge(inputs, path, digest)
    if reference_data["model"] != {"path": str(model), "bytes": model.stat().st_size, "sha256": inputs[str(model)]}:
        raise ValueError("reference model identity")
    compiler = Path("/usr/bin/clang++-18")
    for path in (Path(__file__), ROOT / "tests/llm_full_numeric_test.py", ROOT / "tests/llm_rms_numeric_mock_test.cpp", Path(provenance.__file__), compiler, Path("/usr/bin/ldd")):
        provenance.merge(inputs, path, provenance.sha(path))
    output.mkdir(parents=True); raw = output / "raw"; raw.mkdir()
    commands = []
    excluded = {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX"}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LD_", "GGML_", "OMP_")) and k not in excluded}
    env.update(ASAN_OPTIONS="halt_on_error=1:detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1", GGML_CPU_DISABLE_FUSION="1")

    def run(label, argv):
        argv = list(map(str, argv)); start = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=600)
        (output / (label + ".stdout")).write_bytes(result.stdout)
        (output / (label + ".stderr")).write_bytes(result.stderr)
        commands.append({"label": label, "argv": argv, "returncode": result.returncode, "seconds": time.monotonic()-start})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise ValueError(label + " failed")
        return result.stdout.decode()

    client = ROOT / ".deps/sources/llama-client"; lib = cpu / "cmake/bin"
    includes = [ROOT / "ports/ggml", ROOT / "ports/hip", ROOT / "tests", ROOT / ".deps/sources/hip-headers/include",
                ROOT / ".deps/sources/clr/hipamd/include", client / "include", client / "ggml/include"]
    flags = [compiler, "-std=c++17", "-O1", "-gline-tables-only", "-Wall", "-Wextra", "-Werror",
             "-Wno-misleading-indentation", "-fsanitize=address,undefined", "-D__HIP_PLATFORM_AMD__",
             *("-I"+str(p) for p in includes), "-isystem", client / "ggml/src"]
    if same_input_oracle:
        flags.append("-DPS5HIP_SAME_INPUT_ORACLE")
    units = [ROOT / "tests/llm_full_numeric_test.cpp", ROOT / "ports/ggml/ps5_backend.cpp"]
    objects = []
    for i, unit in enumerate(units):
        dep = output / f"headers-{i}.d"
        run(f"headers-{i}", [*flags, "-M", "-MF", dep, unit])
        for path in provenance.dependencies(dep):
            provenance.merge(inputs, path, provenance.sha(path))
    for i, unit in enumerate(units):
        obj, dep = output / f"unit-{i}.o", output / f"compiled-{i}.d"
        run(f"compile-{i}", [*flags, "-MD", "-MF", dep, "-c", unit, "-o", obj])
        if provenance.dependencies(dep) != provenance.dependencies(output / f"headers-{i}.d"):
            raise ValueError("compiler dependency drift")
        objects.append(obj)
    binary = output / "full-numeric"
    run("link", [*flags, *objects, "-L"+str(lib), "-Wl,-rpath,"+str(lib), "-Wl,--no-as-needed",
                 "-lllama", "-lggml", "-lggml-cpu", "-lggml-base", "-ldl", "-o", binary])
    for path, digest in provenance.runtime_inputs(run("ldd", ["/usr/bin/ldd", binary]), lib, inputs).items():
        provenance.merge(inputs, path, digest)
    binary_hash = provenance.sha(binary)
    run("execute", [binary, model, raw])
    if provenance.sha(binary) != binary_hash:
        raise ValueError("binary changed")
    rows = compare(raw, reference / "raw"); classification, exit_code = status(rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    provenance.recheck(inputs)
    receipt = {"status": classification, "classification": "full-model-host-HIP-numerical-comparison",
               "gpu_executed": False, "same_input_oracle": same_input_oracle, "raw_files": 72, "requests": 3, "inputs": inputs, "commands": commands,
               "reference_receipt": reference_data["receipt"], "reference_sha256": reference_data["receipt_sha256"],
               "model": reference_data["model"], "tolerances": {"absolute": ABS_TOL, "relative": REL_TOL, "relative_l2": L2_TOL},
               "artifacts": {str(p.relative_to(output)): provenance.sha(p) for p in output.rglob("*") if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(classification)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu-build", "model", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--same-input-oracle", action="store_true", help="Also check identical-input Q8 bytes and Q4 FP64 error bounds; does not relax the CPU gate")
    args = parser.parse_args()
    raise SystemExit(build(args.cpu_build, args.model, args.reference, args.output, args.same_input_oracle))
