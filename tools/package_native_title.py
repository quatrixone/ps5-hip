#!/usr/bin/env python3
"""Create an offline UFS2 FFPKG and verify every file after extraction."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_tree(root, files):
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks are not permitted in title images")
        if path.is_file():
            data = path.read_bytes()
            actual[path.relative_to(root).as_posix()] = {"sha256": digest(data), "bytes": len(data)}
    if actual != files:
        raise ValueError("title file inventory or digest mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=pathlib.Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if not stage.is_relative_to((ROOT / "dist").resolve()):
        parser.error("package only a stage in this repository's dist directory")
    manifest_bytes = (stage / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    identity = json.loads((ROOT / "title.json").read_text())
    if manifest["title_id"] != identity["title_id"]:
        parser.error("staged title identity mismatch")
    if digest((stage / "build-receipt.json").read_bytes()) != manifest["build_receipt_sha256"]:
        parser.error("staged build receipt changed")
    verify_tree(stage / "app", manifest["files"])
    lock = json.loads((ROOT / "dependencies.json").read_text())["ufs2tool"]
    dependency = ROOT / ".deps" / ("ufs2tool-" + lock["revision"])
    tool_receipt_bytes = (dependency / "receipt.json").read_bytes()
    tool_receipt = json.loads(tool_receipt_bytes)
    if tool_receipt["revision"] != lock["revision"] or tool_receipt["dotnet_sdk"] != lock["dotnet_sdk"]:
        parser.error("image tool provenance mismatch")
    for name, expected in tool_receipt["artifacts"].items():
        if digest((dependency / "bin" / name).read_bytes()) != expected:
            parser.error("image tool artifact changed")
    output = stage / (manifest["title_id"] + ".ffpkg")
    if output.exists() or (stage / "package-receipt.json").exists():
        parser.error("package already exists; never overwrite a frozen image")
    command = ["dotnet", str(dependency / "bin/UFS2Tool.dll")]
    env = dict(os.environ, DOTNET_ROLL_FORWARD="Major", DOTNET_CLI_TELEMETRY_OPTOUT="1")
    with tempfile.TemporaryDirectory(prefix=".pack-", dir=stage) as temporary:
        temporary = pathlib.Path(temporary)
        image, extracted = temporary / "title.ffpkg", temporary / "extracted"
        with (stage / "package.log").open("w") as log:
            for arguments in (["makefs", "-S", "4096", "-b", "20%", "-t", "ffs", "-o",
                               "version=2,bsize=32768,fsize=4096,minfree=0,softupdates=0,optimization=space",
                               str(image), str(stage / "app")],
                              ["fsck_ufs", "-f", "-n", str(image)],
                              ["extract", str(image), str(extracted)]):
                subprocess.run(command + arguments, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_tree(extracted, manifest["files"])
        image_bytes = image.read_bytes()
        if image_bytes[0x1055c:0x10560] != bytes.fromhex("19015419"):
            raise ValueError("missing UFS2 superblock magic")
        # Publish only after filesystem checks and a complete byte-for-byte round trip.
        with output.open("xb") as destination:
            destination.write(image_bytes)
    receipt = {"schema": 1, "hardware_validated": False, "title_id": manifest["title_id"],
               "manifest_sha256": digest(manifest_bytes), "tool_receipt_sha256": digest(tool_receipt_bytes),
               "tool_revision": lock["revision"], "image": {"file": output.name,
               "sha256": digest(image_bytes), "bytes": len(image_bytes)},
               "filesystem_check": "passed", "extracted_files_match": True}
    (stage / "package-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified offline package:", output)


if __name__ == "__main__":
    main()
