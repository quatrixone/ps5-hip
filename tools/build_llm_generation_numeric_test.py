"""Build the actual-scheduler HIP interpreter and compare greedy feedback to the CPU oracle."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
import build_llm_executed_graph as provenance
from build_llm_generation_reference import authenticate_generation_reference, loads

ROOT = Path(__file__).resolve().parents[1]
ABS_TOL, REL_TOL, L2_TOL = 2e-5, 2e-4, 2e-4
def compare_generation(raw, reference):
    raw, reference = Path(raw), Path(reference)
    expected = loads((reference / "generation.json").read_text())
    actual = loads((raw / "generation.json").read_text())
    if actual != expected or (raw / "steps.jsonl").read_bytes() != (reference / "steps.jsonl").read_bytes():
        raise ValueError("generation feedback/token metadata differs")
    runs = expected["runs"]
    if len(runs) != 2 or runs[0]["processed"] != runs[1]["processed"]:
        raise ValueError("generation reset history differs")
    files = {f"run-{run['run']}-pos-{p}-{stage}.f32le": 128000 if stage == "logits" else 8192
             for run in runs for p in range(len(run["processed"]))
             for stage in [*(f"layer-{i}" for i in range(22)), "norm", "logits"]}
    if raw.is_symlink() or {p.name for p in raw.iterdir()} != set(files) | {"generation.json", "steps.jsonl", "backend.json"}:
        raise ValueError("generation output inventory")
    for name in ("generation.json", "steps.jsonl", "backend.json"):
        if (raw / name).is_symlink() or not (raw / name).is_file():
            raise ValueError("generation metadata kind")
    winners = {}
    rows = []
    for name, size in files.items():
        path = raw / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("generation boundary size/kind")
        content = path.read_bytes(); values = struct.unpack("<" + str(size // 4) + "f", content)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite generation boundary")
        if name.startswith("run-1-") and content != (raw / name.replace("run-1-", "run-0-")).read_bytes():
            raise ValueError("generation boundary reset differs")
        if name.endswith("-logits.f32le"):
            winners[name] = max(range(32000), key=lambda i: values[i])
        target = reference / name
        if target.is_symlink() or not target.is_file() or target.stat().st_size != size:
            raise ValueError("generation reference boundary")
        wanted = struct.unpack("<" + str(size // 4) + "f", target.read_bytes())
        if not all(math.isfinite(x) for x in wanted):
            raise ValueError("nonfinite reference boundary")
        differences = [abs(x-y) for x,y in zip(values,wanted)]
        bad = sum(d > ABS_TOL + REL_TOL * abs(y) for d,y in zip(differences,wanted))
        relative = math.sqrt(sum(d*d for d in differences) / max(sum(y*y for y in wanted), 1e-30))
        rows.append(dict(file=name, max_abs=max(differences), outside_element_budget=bad, relative_l2=relative, pass_budget=bad == 0 and relative <= L2_TOL))
    prompt_count = len(expected["prompt_tokens"])
    for run in runs:
        for i, token in enumerate(run["generated"]):
            name = f"run-{run['run']}-pos-{prompt_count-1+i}-logits.f32le"
            if winners.get(name) != token:
                raise ValueError("actual logits greedy argmax differs")
    counters = loads((raw / "backend.json").read_text()); requests = sum(len(run["processed"]) for run in runs)
    keys = {"requests", "nodes", "launches", "CPU_arithmetic", "scheduler_forwarded", "allocations", "frees", "instances_created", "instances_freed"}
    if set(counters) != keys or any(type(value) is not int for value in counters.values()):
        raise ValueError("generation backend counter types/keys")
    if (counters.get("requests"), counters.get("nodes"), counters.get("launches"), counters.get("CPU_arithmetic"), counters.get("scheduler_forwarded")) != (requests, requests*798, requests*733, 0, requests):
        raise ValueError("generation backend counters")
    for a,b in (("allocations","frees"),("instances_created","instances_freed")):
        if type(counters.get(a)) is not int or counters[a] <= 0 or counters[a] != counters.get(b):
            raise ValueError("generation backend ownership")
    return rows, files, requests


def build(cpu, model, reference, output, native_helper=False):
    cpu, model, reference, output = (Path(p).resolve() for p in (cpu, model, reference, output))
    if output.exists() or not output.is_relative_to(ROOT / "results"):
        raise ValueError("fresh results output required")
    reference_data, inputs = authenticate_generation_reference(reference)
    inputs = dict(inputs)
    for path, digest in provenance.cpu_inputs(cpu, model).items():
        provenance.merge(inputs, path, digest)
    if reference_data["model"] != {"path": str(model), "bytes": model.stat().st_size, "sha256": inputs[str(model)]}:
        raise ValueError("reference model identity")
    compiler = Path("/usr/bin/clang++-18")
    for path in (Path(__file__), ROOT / "tests/llm_generation_numeric_test.py", ROOT / "tests/llm_rms_numeric_mock_test.cpp", Path(provenance.__file__), compiler, Path("/usr/bin/ldd")):
        provenance.merge(inputs, path, provenance.sha(path))
    if native_helper:
        for path in (ROOT / "tools/source_llm_generation_log.py", ROOT / "tests/native_generation_transcript_test.py"):
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
        if result.returncode or any(marker in result.stderr for marker in (b"AddressSanitizer", b"UndefinedBehaviorSanitizer", b"LeakSanitizer", b"runtime error:")):
            raise ValueError(label + " failed")
        return result.stdout.decode()

    client = ROOT / ".deps/sources/llama-client"; lib = cpu / "cmake/bin"
    includes = [ROOT / "ports/ggml", ROOT / "ports/hip", ROOT / "tests", ROOT / ".deps/sources/hip-headers/include",
                ROOT / ".deps/sources/clr/hipamd/include", client / "include", client / "ggml/include"]
    flags = [compiler, "-std=c++17", "-O1", "-gline-tables-only", "-Wall", "-Wextra", "-Werror",
             "-Wno-misleading-indentation", "-fsanitize=address,undefined", "-D__HIP_PLATFORM_AMD__",
             *("-I"+str(p) for p in includes), "-isystem", client / "ggml/src"]
    units = [ROOT / ("tests/llm_native_generation_numeric_test.cpp" if native_helper else "tests/llm_generation_numeric_test.cpp"), ROOT / "ports/ggml/ps5_backend.cpp"]
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
    executed = run("execute", [binary, model, raw])
    marker = "GENERATION_NUMERIC_PASS actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false"
    if executed.splitlines().count(marker) != 1:
        raise ValueError("generation completion marker missing/duplicated")
    if provenance.sha(binary) != binary_hash:
        raise ValueError("binary changed")
    if native_helper:
        from source_llm_generation_log import check_fragment
        policy = loads((reference / "raw/generation.json").read_text())
        metadata_sizes = {name: (raw / name).stat().st_size for name in ("generation.json", "steps.jsonl")}
        # The exact marker follows the production helper; no arbitrary log filtering.
        if not executed.endswith(marker + "\n"):
            raise ValueError("generation helper terminal marker")
        timing = check_fragment(executed[:-len(marker)-1].encode('ascii'), policy, metadata_sizes)
        (output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    rows, files, requests = compare_generation(raw, reference / "raw")
    classification = "PASS" if all(row["pass_budget"] for row in rows) else "NUMERICAL_FAIL"
    exit_code = 0 if classification == "PASS" else 1
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    provenance.recheck(inputs)
    receipt = {"status": classification, "classification": "generation-host-HIP-numerical-comparison",
               "gpu_executed": False, "native_executed": False, "native_helper": native_helper, "limits": ["Numerical HIP interpreter only; no GPU execution", "CPU numerical comparison covers every layer, final normalization and complete logits; no native device arithmetic is established"], "raw_files": len(files), "requests": requests, "inputs": inputs, "commands": commands,
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
    parser.add_argument("--native-helper", action="store_true", help="Execute the production native generation helper under numerical HIP interpretation")
    args = parser.parse_args()
    raise SystemExit(build(args.cpu_build, args.model, args.reference, args.output, args.native_helper))
