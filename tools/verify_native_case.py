#!/usr/bin/env python3
"""Verify frozen local experiment bytes before console traffic. Does not deploy."""
import argparse
import hashlib
import json
import pathlib
import math_native_fixture as math_fixture
import visibility_fixture
import geometry_fixture
import native_vector_title as vector_title

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fingerprint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular file: {path}")
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify_case(stage, controllers, packages, controller_evidence, title):
    manifest = read_json(stage / "manifest.json")
    package = read_json(stage / "package-receipt.json")
    control = read_json(controllers / "receipt.json")
    # Committed evidence is the authority; adjacent generated receipts alone are not.
    if {"manifest": manifest, "package": package} not in packages["candidates"]:
        raise ValueError("candidate does not match frozen package evidence")
    if package["image"]["sha256"] in packages.get("revoked_images", {}):
        raise ValueError("candidate revoked: " + packages["revoked_images"][package["image"]["sha256"]])
    if control != controller_evidence["controller_build"]:
        raise ValueError("controller receipt does not match frozen evidence")
    if any(item["title_id"] != title for item in (manifest, package, control)):
        raise ValueError("experiment title mismatch")
    if fingerprint(stage / "manifest.json")["sha256"] != package["manifest_sha256"]:
        raise ValueError("manifest bytes changed")
    if fingerprint(stage / "build-receipt.json")["sha256"] != manifest["build_receipt_sha256"]:
        raise ValueError("build receipt bytes changed")
    build = read_json(stage / "build-receipt.json")
    if "hip_vector" in (manifest["kernel"], build.get("kernel")):
        if (manifest["kernel"] != build.get("kernel") or manifest["target"] != build.get("target") or
                manifest.get("hip_vector_profile") != build.get("hip_vector_profile") or
                manifest.get("source_commit") != build.get("source_commit") or build.get("source_tree_clean") is not True or
                "canary.hsaco" in manifest["files"]):
            raise ValueError("genuine HIP vector manifest/build profile mismatch")
        vector_title.validate_build(build, ROOT, stage / "hip-vector-provenance", (stage / "app/eboot.bin").read_bytes())
    if "vector_add_twice" in (manifest["kernel"], build.get("kernel")):
        if (build.get("kernel") != manifest["kernel"] or build.get("target") != manifest["target"] or
                any(type(item.get("dispatches")) is not int or item["dispatches"] != 2 for item in (manifest, build))):
            raise ValueError("vector_add_twice requires matching two-dispatch provenance")
    if "fp32_math" in (manifest["kernel"], build.get("kernel")):
        if (manifest["kernel"] != build.get("kernel") or manifest["target"] != build.get("target") or
                manifest.get("math_profile") != build.get("math_profile")):
            raise ValueError("math manifest/build profile mismatch")
        math_fixture.validate_build(build, (stage / "app/canary.hsaco").read_bytes(), ROOT)
    if manifest["kernel"] in geometry_fixture.PROFILES or build.get("kernel") in geometry_fixture.PROFILES:
        if manifest["kernel"] != build.get("kernel") or manifest["target"] != build.get("target") or manifest.get("geometry_profile") != build.get("geometry_profile"):
            raise ValueError("geometry manifest/build mismatch")
        geometry_fixture.validate_build(build, (stage / "app/canary.hsaco").read_bytes(), ROOT)
    if "dependent_visibility" in (manifest["kernel"], build.get("kernel")):
        if manifest["kernel"] != build.get("kernel") or manifest["target"] != build.get("target") or manifest.get("visibility_profile") != build.get("visibility_profile"):
            raise ValueError("visibility manifest/build mismatch")
        visibility_fixture.validate_build(build, (stage / "app/canary.hsaco").read_bytes(), ROOT)
    shutdown_mode = build.get("shutdown_mode", "platform-exit")
    if shutdown_mode not in ("platform-exit", "shell-request-probe"):
        raise ValueError("unsupported shutdown mode")
    actual = {}
    for path in (stage / "app").rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks forbidden in staged application")
        if path.is_file():
            actual[path.relative_to(stage / "app").as_posix()] = fingerprint(path)
    if actual != manifest["files"]:
        raise ValueError("staged application inventory or bytes changed")
    image = package["image"]
    if image["file"] != title + ".ffpkg":
        raise ValueError("unexpected image filename")
    if fingerprint(stage / image["file"]) != {key: image[key] for key in ("bytes", "sha256")}:
        raise ValueError("package image bytes changed")
    if set(control["artifacts"]) != {"launch.elf", "close.elf"}:
        raise ValueError("unexpected controller inventory")
    for name, expected in control["artifacts"].items():
        if fingerprint(controllers / name) != expected:
            raise ValueError(f"controller bytes changed: {name}")
    result = {"local_artifacts_verified": True, "hardware_gate_passed": False,
            "title_id": title, "target": manifest["target"], "kernel": manifest["kernel"],
            "shutdown_mode": shutdown_mode,
            "source_commit": manifest["source_commit"], "expected_log": manifest["expected_log"],
            "image": image, "controllers": control["artifacts"],
            "unproven": ["console idle state", "remote bytes", "registration", "log freshness",
                         "native execution", "clean teardown", "service health"]}
    if manifest["kernel"] == "vector_add_twice":
        result["dispatches"] = 2
    if manifest["kernel"] == "hip_vector":
        result["hip_vector_profile"] = manifest["hip_vector_profile"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=pathlib.Path)
    args = parser.parse_args()
    title = read_json(ROOT / "title.json")["title_id"]
    result = verify_case(args.stage, ROOT / "build/controllers" / title,
                         read_json(ROOT / "evidence/offline-native-packages.json"),
                         read_json(ROOT / "evidence/offline-controllers.json"), title)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
