"""Exact experimental compiler binding; metadata alone never qualifies buffer calls."""
import hashlib
import json
import pathlib

LOCK = "experimental/private-buffer-profile.json"
CONTRACT = {"experimental": True, "entry": "nested_private", "groups": 257,
            "threads": 64, "wave_size": 32, "invocations": 16448, "user_sgprs": 8,
            "private_bytes": 304, "root_frame_bytes": 144, "middle_frame_bytes": 160,
            "leaf_frame_bytes": 0, "wave_stride_bytes": 10240,
            "ring_bytes": 13107200, "tmpring": 0xa500}
COUNTS = {"ordinary_audit": 8, "extension_audit": 48, "guard_controls": 14,
          "register_controls": 4, "alias_linked_objects": 6}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("invalid compiler artifact path")
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("invalid compiler artifact path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("compiler artifact outside output or missing")
    return path


def load_lock(root):
    raw = (root / LOCK).read_bytes()
    lock = json.loads(raw)
    if (lock.get("schema") != 1 or lock.get("profile") != "private_calls_buffer" or
            not isinstance(lock.get("reproduction_receipt_sha256"), str) or
            len(lock["reproduction_receipt_sha256"]) != 64):
        raise ValueError("private buffer compiler binding is not frozen")
    profiles = lock.get("selected_profiles", [])
    if (len(profiles) != 4 or {(p["target"], p["optimization"]) for p in profiles} !=
            {(t, o) for t in ("gfx1013", "gfx1030") for o in ("O1", "O2")}):
        raise ValueError("private buffer lock profile set mismatch")
    return lock, digest(raw)


def binding(root, target):
    lock, lock_hash = load_lock(root)
    selected = [p for p in lock["selected_profiles"] if p["target"] == target and p["optimization"] == "O2"]
    if len(selected) != 1:
        raise ValueError("private buffer target mismatch")
    return {"kind": "experimental-private-buffer", "reproduction_receipt_sha256": lock["reproduction_receipt_sha256"],
            "profile_lock_sha256": lock_hash, "selected_profile": selected[0]}


def verified_private_buffer_object(root, output, target):
    lock, _ = load_lock(root)
    raw = contained_file(output, "reproduction-receipt.json").read_bytes()
    if digest(raw) != lock["reproduction_receipt_sha256"]:
        raise ValueError("private buffer reproduction receipt digest mismatch")
    receipt = json.loads(raw)
    if (type(receipt.get("schema")) is not int or receipt["schema"] != 1 or
            receipt.get("selected_recipe_reproduction_passed") is not True or
            receipt.get("counts") != COUNTS or
            any(type(v) is not int for v in receipt["counts"].values()) or
            receipt.get("selected_profiles") != lock["selected_profiles"]):
        raise ValueError("private buffer reproduction contract mismatch")
    for section in ("artifacts", "receipts"):
        records = receipt.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError("missing compiler artifact inventory")
        for name, expected in records.items():
            if digest(contained_file(output, name).read_bytes()) != expected:
                raise ValueError("compiler artifact digest mismatch: " + name)
    for profile in receipt["selected_profiles"]:
        data = contained_file(output, profile["path"]).read_bytes()
        if (type(profile["bytes"]) is not int or len(data) != profile["bytes"] or
                digest(data) != profile["code_sha256"] or
                receipt["artifacts"].get(profile["path"]) != profile["code_sha256"] or
                profile["input_sha256"] not in receipt["artifacts"].values()):
            raise ValueError("selected private buffer object/input mismatch")
    report = binding(root, target)
    return contained_file(output, report["selected_profile"]["path"]).read_bytes(), report


def validate_build_binding(root, build):
    expected = binding(root, build.get("target"))
    if (build.get("kernel") != "private_calls_buffer" or
            build.get("private_buffer") != CONTRACT or
            any(type(build["private_buffer"][k]) is not type(v) for k, v in CONTRACT.items()) or
            build.get("device_build") != expected or
            build.get("artifacts", {}).get("canary.hsaco") != expected["selected_profile"]["code_sha256"]):
        raise ValueError("private buffer requires exact experimental compiler provenance")
    for name in (LOCK, "tools/private_buffer_fixture.py"):
        if build.get("source_sha256", {}).get(name) != digest((root / name).read_bytes()):
            raise ValueError("private buffer helper/lock provenance missing")
    return expected
