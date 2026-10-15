#!/usr/bin/env python3
"""Prepare a private SDK copy from pinned source and a verified cached release.

Runs under WSL/Linux. This tool performs no download and never contacts a console.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-source", type=pathlib.Path, required=True)
    parser.add_argument("--release-archive", type=pathlib.Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "dependencies.json").read_text())
    sdk = lock["sdk"]
    if hashlib.sha256(args.release_archive.read_bytes()).hexdigest() != sdk["release_sha256"]:
        parser.error("SDK release digest does not match dependencies.json")
    subprocess.run(["git", "-C", str(args.sdk_source), "cat-file", "-e", sdk["revision"] + "^{commit}"], check=True)
    env = dict(os.environ, LLVM_CONFIG=shutil.which("llvm-config-18") or "llvm-config-18")
    version = subprocess.check_output(["llvm-config-18", "--version"], text=True).strip()
    if version != lock["llvm"]["version"]:
        parser.error("Native compiler version does not match dependencies.json")
    deps = ROOT / ".deps"
    exports = deps / ("sdk-source-" + sdk["revision"])
    cache = deps / "sdk-cache"
    cache.mkdir(parents=True, exist_ok=True)
    # The setup script may replace its SDK/cache output; all destinations stay here.
    for destination in (exports, cache, deps / "sdk"):
        if not destination.resolve().is_relative_to(deps.resolve()):
            parser.error("SDK output resolves outside this repository's .deps directory")
    if not (exports / ".export-complete").exists():
        exports.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(["git", "-C", str(args.sdk_source), "archive", sdk["revision"]], stdout=subprocess.PIPE)
        try:
            subprocess.run(["tar", "-x", "-C", str(exports)], stdin=process.stdout, check=True)
        finally:
            process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError("SDK source export failed")
        (exports / ".export-complete").write_text(sdk["revision"] + "\n")
    archive = cache / "ps5-payload-sdk.zip"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != sdk["release_sha256"]:
        shutil.copyfile(args.release_archive, archive)
    subprocess.run(["bash", str(exports / "platform/tools/setup-sdk.sh"), str(deps / "sdk"),
                    sdk["revision"], str(cache)], env=env, check=True)
    (deps / "sdk/.ps5-sdk-revision").touch()
    print("Prepared pinned SDK at", deps / "sdk")


if __name__ == "__main__":
    main()
