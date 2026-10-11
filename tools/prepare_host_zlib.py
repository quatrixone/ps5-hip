#!/usr/bin/env python3
"""Build the native converter's pinned static zlib from a cached archive, offline."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "dependencies.json").read_text())
    zlib = lock["host_zlib"]
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != zlib["sha256"]:
        parser.error("zlib archive digest does not match dependencies.json")
    if subprocess.check_output(["llvm-config-18", "--version"], text=True).strip() != lock["llvm"]["version"]:
        parser.error("host compiler version does not match dependency lock")
    directory = ROOT / ".deps" / ("host-zlib-" + zlib["version"])
    if not directory.resolve().is_relative_to((ROOT / ".deps").resolve()):
        parser.error("zlib output resolves outside repository dependencies")
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "receipt.json"
    receipt.unlink(missing_ok=True)
    with tarfile.open(args.archive, "r:gz") as archive:
        archive.extractall(directory, filter="data")
    source = directory / ("zlib-" + zlib["version"])
    install = directory / "root"
    env = dict(os.environ, CC="clang-18", AR="llvm-ar-18", RANLIB="llvm-ranlib-18")
    with (directory / "build.log").open("w") as log:
        for command in (["sh", "configure", "--static", "--prefix=" + str(install)],
                        ["make", "-j4"], ["make", "install"]):
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    artifacts = ["lib/libz.a", "include/zlib.h", "include/zconf.h"]
    receipt.write_text(json.dumps({"source_sha256": zlib["sha256"], "llvm_version": lock["llvm"]["version"],
                                  "artifacts": {name: hashlib.sha256((install / name).read_bytes()).hexdigest()
                                                for name in artifacts}}, indent=2) + "\n")
    print("Prepared pinned host zlib:", install)


if __name__ == "__main__":
    main()
