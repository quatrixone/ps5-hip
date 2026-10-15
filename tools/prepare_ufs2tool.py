#!/usr/bin/env python3
"""Build the pinned UFS2 image tool with locked NuGet package hashes."""
import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "dependencies.json").read_text())["ufs2tool"]
    version = subprocess.check_output(["dotnet", "--version"], text=True).strip()
    if version != lock["dotnet_sdk"]:
        parser.error("installed .NET SDK does not match dependency lock")
    directory = ROOT / ".deps" / ("ufs2tool-" + lock["revision"])
    if not directory.resolve().is_relative_to((ROOT / ".deps").resolve()):
        parser.error("UFS2 output resolves outside dependencies")
    source, binary = directory / "source", directory / "bin"
    for path in (source, binary):
        path.mkdir(parents=True, exist_ok=True)
    (directory / "receipt.json").unlink(missing_ok=True)
    data = subprocess.check_output(["git", "-C", str(args.source), "archive", lock["revision"]])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(source, filter="data")
    package_lock = ROOT / "locks/ufs2tool-packages.lock.json"
    shutil.copyfile(package_lock, source / "packages.lock.json")
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", DOTNET_ROLL_FORWARD="Major")
    project = source / "UFS2Tool.csproj"
    with (directory / "build.log").open("w") as log:
        for command in (["dotnet", "restore", str(project), "--locked-mode",
                         "--source", "https://api.nuget.org/v3/index.json", "-p:NuGetAudit=false"],
                        ["dotnet", "build", str(project), "--no-restore", "-c", "Release", "-o", str(binary)]):
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    artifacts = sorted(binary.glob("*.dll")) + sorted(binary.glob("*.json"))
    (directory / "receipt.json").write_text(json.dumps({
        "revision": lock["revision"], "dotnet_sdk": version,
        "package_lock_sha256": hashlib.sha256(package_lock.read_bytes()).hexdigest(),
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
    }, indent=2) + "\n")
    print("Prepared locked UFS2 tool:", binary / "UFS2Tool.dll")


if __name__ == "__main__":
    main()
