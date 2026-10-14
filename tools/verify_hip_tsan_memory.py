"""Preserve one targeted TSan result when the failed-join leak reporter stalls."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
build = args.build.resolve()
output = args.output.resolve()
root = Path(__file__).resolve().parents[1]
assert not output.exists(), "Use a fresh result directory"
commands = json.loads((build / "commands.json").read_text())
assert any(c["label"] == "link" and c["exit_code"] == 0 for c in commands)
assert sum(c["label"].startswith("hip_") and c["label"] not in ("hip_api_trace-headers",) and c["exit_code"] == 0 for c in commands) >= 20
output.mkdir(parents=True)
environment = dict(os.environ, TSAN_OPTIONS="halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
command = [str(build / "model"), "memory"]
result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
(output / "stdout").write_text(result.stdout)
(output / "stderr").write_text(result.stderr)
assert result.returncode == 0 and not result.stderr
assert "HIP_PUBLIC_PASS mode=memory gpu_execution=false" in result.stdout
assert "HIP_PUBLIC_FINAL queues=1 retired=1 gate_destroyed=1 retained=0" in result.stdout
receipt = {
    "classification": "targeted-public-hip-memory-tsan-model",
    "inputs": {str(path): sha(path) for path in (
        Path(__file__), root / "tests/hip_admission_model.cpp",
        root / "tools/qualify_hip_admission.py", root / "results/source-v3/hip-memory-composition.json",
        build / "model", build / "commands.json")},
    "command": command,
    "environment": {"TSAN_OPTIONS": environment["TSAN_OPTIONS"], "UBSAN_OPTIONS": environment["UBSAN_OPTIONS"]},
    "exit_code": result.returncode,
    "stdout_sha256": sha(output / "stdout"),
    "stderr_sha256": sha(output / "stderr"),
    "full_tsan_qualification": False,
    "gpu_executed": False,
}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(sha(output / "receipt.json"))
