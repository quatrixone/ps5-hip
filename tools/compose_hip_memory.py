"""Compose the reviewed public HIP, SVM memory and profiling patches exactly."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

if not __debug__:
    raise RuntimeError("Validation requires ordinary Python")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.admission_source.resolve()
    output = args.output.resolve()
    manifest = json.loads((root / "ports/clr/ps5-public-hip.json").read_text())
    assert sha(root / "ports/clr/ps5-public-hip.patch") == manifest["patch_sha256"]
    for name, entry in manifest["files"].items():
        assert sha(source / name) == entry["result_sha256"], name
    changes = {
        "rocclr/platform/memory.cpp": (
            "fe9384d6a8a7e763238164ecbf9e0c7cf0b42e0d09f511ec5a975395723e40b8",
            "7cb0364bb313e3fb60292511dbeca5dae6375a6aa5e4d3d0429c2b8633b4f215",
        ),
        "hipamd/src/hip_memory.cpp": (
            "833ff2bf952d77cdea083751022d4b9989c0d5e17e566213c95e92369674cb4b",
            "897238c92852724fa84f126090a10b4129c3a8848a85243383af3ee85927f75f",
        ),
        "hipamd/src/hip_prof_api.h": (
            "1f91d5bb347d228b60490785b09ce7724078e005f48b87616402cfdd6a1afaa0",
            "57a603b91235d4cf2e9d1683190316952cbe3b3b9bc57339e9dfdaa5f70094ed",
        ),
    }
    for name, (before, _) in changes.items():
        assert sha(source / name) == before, name
    patches = {
        "ps5-public-memory.patch": "e12744ddeab27cc4ea390c7eca063ac02277724892ff5838a94bc10c3c41b23a",
        "ps5-hip-correlation.patch": "959b04dbd7779e285e36e26ed93c378c9b39d963e8505836cf1cd8348ea860cb",
    }
    for name, digest in patches.items():
        assert sha(root / "ports/clr" / name) == digest, name
    assert not output.exists(), "Use a fresh output directory"
    shutil.copytree(source, output)
    for name in patches:
        subprocess.run(["git", "apply", "--check", str(root / "ports/clr" / name)], cwd=output, check=True)
        subprocess.run(["git", "apply", str(root / "ports/clr" / name)], cwd=output, check=True)
    for name, (_, after) in changes.items():
        assert sha(output / name) == after, name
    receipt = {
        "classification": "exact-public-hip-memory-composition",
        "admission_manifest_sha256": sha(root / "ports/clr/ps5-public-hip.json"),
        "patches": patches,
        "changed_sources": {name: {"before": pair[0], "after": pair[1]} for name, pair in changes.items()},
        "gpu_executed": False,
    }
    (output / "hip-memory-composition.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(sha(output / "hip-memory-composition.json"))

if __name__ == "__main__":
    main()
