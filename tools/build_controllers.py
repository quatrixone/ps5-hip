#!/usr/bin/env python3
"""Compile frozen title-aware controllers offline; this tool has no send operation."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-source", type=pathlib.Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((ROOT / "controller-inputs.json").read_text())
    lock = json.loads((ROOT / "dependencies.json").read_text())
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    if not re.fullmatch(r"PPSA[0-9]{5}", title):
        parser.error("invalid dedicated title identity")
    sdk = ROOT / ".deps/sdk"
    if (sdk / ".ps5-sdk-revision").read_text().strip() != lock["sdk"]["revision"]:
        parser.error("SDK revision mismatch")
    if subprocess.check_output(["llvm-config-18", "--version"], text=True).strip() != lock["llvm"]["version"]:
        parser.error("compiler version mismatch")
    source = ROOT / ".deps" / ("protocol-" + inputs["revision"])
    build = ROOT / "build/controllers" / title
    source.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LLVM_CONFIG=shutil.which("llvm-config-18"))
    receipt = {"title_id": title, "protocol_revision": inputs["revision"],
               "sdk_revision": lock["sdk"]["revision"], "llvm_version": lock["llvm"]["version"],
               "hardware_validated": False, "sources": {}, "artifacts": {}}
    for name in ["LICENSE"] + [f"scripts/controllers/{mode}.c" for mode in inputs["controllers"]]:
        data = subprocess.check_output(["git", "-C", str(args.protocol_source), "show", f"{inputs['revision']}:{name}"])
        output = source / pathlib.PurePosixPath(name).name
        output.write_bytes(data)
        receipt["sources"][output.name] = hashlib.sha256(data).hexdigest()
    for mode in inputs["controllers"]:
        output = build / (mode + ".elf")
        libraries = ["-lSceSystemService"] + (["-lSceUserService"] if mode == "launch" else [])
        subprocess.run([str(sdk / "bin/prospero-clang"), "-Wall", "-Werror",
                        f'-DBOOTSTRAP_TITLE_ID="{title}"', *libraries, str(source / (mode + ".c")),
                        "-o", str(output)], env=env, check=True)
        receipt["artifacts"][output.name] = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                           "bytes": output.stat().st_size}
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Built launch/close controllers for", title, "without console access.")


if __name__ == "__main__":
    main()
