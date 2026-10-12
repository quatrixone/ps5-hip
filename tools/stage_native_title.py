#!/usr/bin/env python3
"""Verify and stage one native title folder offline. Never deploys or overwrites."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import struct
import math_native_fixture as math_fixture

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_results_directory(path):
    if (not isinstance(path, str) or not 2 <= len(path) <= 240 or not path.startswith("/") or
            any(ord(c) < 32 or ord(c) > 126 or c == "\\" for c in path) or
            any(part in ("", ".", "..") for part in path[1:].split("/"))):
        raise ValueError("results directory must be a normalized absolute path, 2..240 ASCII bytes")
    return path


def stage(source_root, build, output, identity, source_commit, results_directory):
    validate_results_directory(results_directory)
    title_id, title_name = identity["title_id"], identity["title_name"]
    if not re.fullmatch(r"PPSA[0-9]{5}", title_id) or not isinstance(title_name, str) or not title_name.strip() or any(ord(c) < 32 for c in title_name):
        raise ValueError("invalid title identity")
    receipt_bytes = (build / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    target, kernel = receipt["target"], receipt["kernel"]
    if target not in ("gfx1013", "gfx1030") or kernel not in ("runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "fp32_math"):
        raise ValueError("unsupported candidate profile")
    if kernel == "vector_add_twice" and (type(receipt.get("dispatches")) is not int or receipt["dispatches"] != 2):
        raise ValueError("vector_add_twice requires exactly two dispatches")
    expected = {"eboot.bin", "lapy.elf", "sce_module/libc.prx"}
    if kernel != "runtime":
        expected.add("canary.hsaco")
    if set(receipt["artifacts"]) != expected or receipt.get("hardware_validated") is not False:
        raise ValueError("invalid offline candidate receipt")
    if not receipt["source_sha256"]:
        raise ValueError("missing source provenance")
    for name, expected_hash in receipt["source_sha256"].items():
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("invalid source path in receipt")
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root.resolve()) or digest(path.read_bytes()) != expected_hash:
            raise ValueError("source changed since build: " + name)
    files = {}
    for name in sorted(expected):
        data = (build / "candidate" / name).read_bytes()
        if digest(data) != receipt["artifacts"][name]:
            raise ValueError("candidate digest mismatch: " + name)
        files[name] = data
    if kernel == "fp32_math":
        math_fixture.validate_build(receipt, files["canary.hsaco"], source_root)
    metadata = {
        "ageLevel": {"default": 0}, "applicationCategoryType": 0,
        "applicationDrmType": "free", "attribute": 0, "attribute2": 0, "attribute3": 0,
        "conceptId": title_id[4:], "contentBadgeType": 1,
        "contentId": f"UP9000-{title_id}_00-PS5HIPCOMPUTE001",
        "contentVersion": "01.000.000", "downloadDataSize": 0,
        "gameIntent": {"permittedIntents": [{"intentType": "launchActivity"}]},
        "localizedParameters": {"defaultLanguage": "en-US", "en-US": {"titleName": title_name}},
        "masterVersion": "01.00", "requiredSystemSoftwareVersion": "0x0000000000000000",
        "sdkVersion": "0x0000000000000000", "titleId": title_id, "versionFileUri": "",
    }
    files["sce_sys/param.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    icon = (source_root / "assets/icon0.png").read_bytes()
    if (len(icon) < 33 or icon[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or
            struct.unpack_from(">II", icon, 16) != (512, 512)):
        raise ValueError("title icon must be a 512x512 PNG")
    files["sce_sys/icon0.png"] = icon
    files["results-dir.txt"] = results_directory.encode("ascii")
    manifest = {
        "schema": 1, "title_id": title_id, "target": target, "kernel": kernel,
        "source_commit": source_commit, "build_receipt_sha256": digest(receipt_bytes),
        "stager_sha256": digest(pathlib.Path(__file__).read_bytes()),
        "hardware_validated": False,
        "expected_log": f"{results_directory}/ps5-hip-{target}-{kernel}.log",
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in files.items()},
    }
    if kernel == "vector_add_twice":
        manifest["dispatches"] = 2
    if kernel == "fp32_math":
        manifest["math_profile"] = receipt["math_profile"]
    # Validate everything before creating output. A failure never replaces a prior candidate.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output / "app" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "build-receipt.json").write_bytes(receipt_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["gfx1013", "gfx1030"], required=True)
    parser.add_argument("--kernel", choices=["runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "fp32_math"], required=True)
    parser.add_argument("--results-dir", required=True, help="precreated directory accessible after elevation")
    args = parser.parse_args()
    identity = json.loads((ROOT / "title.json").read_text())
    build = ROOT / "build/native" / f"{args.target}-{args.kernel}"
    receipt = json.loads((build / "receipt.json").read_text())
    if (receipt["target"], receipt["kernel"]) != (args.target, args.kernel):
        parser.error("build receipt profile mismatch")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain", "--", *receipt["source_sha256"],
                                       "title.json", "tools/stage_native_title.py", "assets/icon0.png"], cwd=ROOT, text=True)
    if changes:
        parser.error("commit compiled sources, title identity and staging tool before staging")
    # Every input that can change staged bytes contributes to the candidate key.
    validate_results_directory(args.results_dir)
    key = digest((build / "receipt.json").read_bytes() + json.dumps(identity, sort_keys=True).encode() +
                 pathlib.Path(__file__).read_bytes() + args.results_dir.encode("ascii") +
                 (ROOT / "assets/icon0.png").read_bytes())[:12]
    output = ROOT / "dist" / identity["title_id"] / f"{args.target}-{args.kernel}-{key}"
    if not output.resolve().is_relative_to(ROOT.resolve()):
        parser.error("staging output resolves outside this repository")
    manifest = stage(ROOT, build, output, identity, source_commit, args.results_dir)
    print("Staged", manifest["title_id"], "at", output)
    print("No installation or console access performed.")


if __name__ == "__main__":
    main()
