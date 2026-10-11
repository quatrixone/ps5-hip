#!/usr/bin/env python3
"""Export pinned Eden client and Lapy helper, then build locally without networking."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eden-source", required=True, type=pathlib.Path)
    parser.add_argument("--lapy-source", required=True, type=pathlib.Path)
    parser.add_argument("--logging-header", required=True, type=pathlib.Path)
    args = parser.parse_args()
    lock = json.loads((ROOT / "elevation-inputs.json").read_text())
    deps = json.loads((ROOT / "dependencies.json").read_text())
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    sdk = ROOT / ".deps/sdk"
    if (sdk / ".ps5-sdk-revision").read_text().strip() != deps["sdk"]["revision"]:
        parser.error("SDK revision mismatch")
    if sha(args.logging_header) != lock["logging_header_sha256"]:
        parser.error("logging header hash mismatch")
    if subprocess.check_output(["llvm-config-18", "--version"], text=True).strip() != deps["llvm"]["version"]:
        parser.error("LLVM version mismatch")
    destination = ROOT / ".deps/elevation"
    files = {}
    exports = [(args.eden_source, lock["client"]["revision"],
                lock["client"]["source_path"] + "/" + name, "client/" + name)
               for name in ("elevation.cpp", "elevation.hpp", "protocol.hpp", "validate-helper.py")]
    exports += [(args.lapy_source, lock["helper"]["revision"], name, "lapy/" + name)
                for name in ("LICENSE", "tools/build_owned_daemon.py",
                             *["source/" + name for name in ("owned_root_daemon.c", "donor_transaction.c",
                                  "donor_transaction.h", "vnode_ref_probe.c", "vnode_ref_probe.h",
                                  "lapy_elevation_protocol.h")])]
    for repository, revision, source, relative in exports:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", str(repository), "show", revision + ":" + source]))
        files[relative] = sha(path)
    logging = destination / "logging"
    logging.mkdir(exist_ok=True)
    shutil.copyfile(args.logging_header, logging / "ps5log.h")
    files["logging/ps5log.h"] = sha(logging / "ps5log.h")
    env = dict(os.environ, LLVM_CONFIG=shutil.which("llvm-config-18"))
    subprocess.run(["python3", str(destination / "lapy/tools/build_owned_daemon.py"),
                    "--sdk", str(sdk), "--logging-client", str(logging),
                    "--elf-helper", "--title", title], env=env, check=True)
    output = destination / "lapy/build" / ("owned_root_helper-" + title)
    subprocess.run(["python3", str(destination / "client/validate-helper.py"),
                    str(output / "lapy.elf")], check=True)
    receipt = {"inputs": lock, "sdk_revision": deps["sdk"]["revision"],
               "llvm_version": deps["llvm"]["version"], "title_id": title,
               "source_sha256": files, "helper_sha256": sha(output / "lapy.elf"),
               "hardware_validated": False}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
