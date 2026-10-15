#!/usr/bin/env python3
"""Fetch the source and archives named by the committed dependency locks."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"


def read(name):
    return json.loads((ROOT / name).read_text())


def sources():
    deps = read("dependencies.json")
    elevation = read("elevation-inputs.json")
    controllers = read("controller-inputs.json")
    clr = read("ports/clr/manifest.json")
    return {
        "sdk": deps["sdk"],
        "ufs2tool": deps["ufs2tool"],
        "native-tools": deps["native_tools"],
        "hip-headers": deps["hip_headers"],
        "eden": elevation["client"],
        "lapy": elevation["helper"],
        "ps5log": elevation["logging_source"],
        "protocol": controllers,
        "clr": {"source_url": clr["upstream_repository"], "revision": clr["upstream_revision"]},
    }


def git(*args):
    return subprocess.check_output(("git", *map(str, args)), text=True).strip()


def source(name, spec, fetch):
    base = DEPS / "sources"
    destination = base / name
    if not destination.exists():
        if not fetch:
            raise RuntimeError(f"missing {destination}; run make deps-fetch")
        base.mkdir(parents=True, exist_ok=True)
        partial = base / (name + ".partial")
        if partial.exists():
            raise RuntimeError(f"incomplete prior clone at {partial}; inspect it before retrying")
        git("clone", "--no-checkout", "--filter=blob:none", spec["source_url"], partial)
        if "source_ref" in spec:
            git("-C", partial, "fetch", "origin", "refs/heads/" + spec["source_ref"])
        git("-C", partial, "checkout", "--detach", spec["revision"])
        partial.rename(destination)
    if git("-C", destination, "remote", "get-url", "origin") != spec["source_url"]:
        raise RuntimeError(f"{name}: remote URL differs from lock")
    if git("-C", destination, "status", "--porcelain"):
        raise RuntimeError(f"{name}: locked source has local changes")
    if git("-C", destination, "rev-parse", "HEAD") != spec["revision"]:
        if not fetch:
            raise RuntimeError(f"{name}: checkout differs from lock; run make deps-fetch")
        git("-C", destination, "fetch", "origin",
            "refs/heads/" + spec["source_ref"] if "source_ref" in spec else spec["revision"])
        git("-C", destination, "checkout", "--detach", spec["revision"])
        if git("-C", destination, "status", "--porcelain"):
            raise RuntimeError(f"{name}: checkout produced local changes")
    return destination


def archive(name, url, expected, fetch):
    destination = DEPS / "downloads" / name
    if not destination.exists():
        if not fetch:
            raise RuntimeError(f"missing {destination}; run make deps-fetch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not url.startswith("https://"):
            raise RuntimeError(f"{name}: only HTTPS archives are allowed")
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = pathlib.Path(temporary.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)
                if hashlib.sha256(temporary_path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"{name}: downloaded SHA-256 differs from lock")
                temporary_path.rename(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name}: cached SHA-256 differs from lock")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download missing pinned inputs")
    args = parser.parse_args()
    for name, spec in sources().items():
        path = source(name, spec, args.fetch)
        print(name, spec["revision"], path)
    elevation = read("elevation-inputs.json")
    header = DEPS / "sources/ps5log" / elevation["logging_source"]["source_path"]
    if hashlib.sha256(header.read_bytes()).hexdigest() != elevation["logging_header_sha256"]:
        raise RuntimeError("ps5log header differs from lock")
    deps = read("dependencies.json")
    for name, spec, suffix in (("sdk", deps["sdk"], "zip"),
                               ("zlib", deps["host_zlib"], "tar.gz")):
        path = archive(name + "." + suffix,
                       spec["release_url"] if name == "sdk" else spec["source_url"],
                       spec["release_sha256"] if name == "sdk" else spec["sha256"],
                       args.fetch)
        print(name + "-archive", path)


if __name__ == "__main__":
    main()
