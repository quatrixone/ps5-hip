#!/usr/bin/env python3
"""Check a captured application log. This does not certify deployment or teardown."""
import argparse
import hashlib
import json
import pathlib


def check_log(data, target, kernel, shutdown_probe=False, shell_exit_probe=False):
    if shutdown_probe and kernel != "runtime":
        raise ValueError("shutdown probe belongs to the CPU runtime case")
    if target not in ("gfx1013", "gfx1030") or kernel not in ("runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "register_spill", "lds_static", "lds_dynamic", "lds_mixed", "private_calls_buffer"):
        raise ValueError("unsupported canary profile")
    if len(data) > 65536:
        raise ValueError("canary log exceeds bounded result size")
    lines = data.decode("utf-8", errors="strict").splitlines()
    expected = [f"PS5HIP START target={target} kernel={kernel}",
                "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass"]
    if kernel == "runtime":
        expected.append("PS5HIP PASS kernel=runtime gpu_api_calls=0")
    elif kernel == "vector_add_twice":
        for dispatch in (1, 2):
            expected += [f"PS5HIP DISPATCH index={dispatch}/2 kernel=vector_add",
                         f"PS5HIP SUBMIT target={target} kernel=vector_add count=73 groups=2 timeout_ms=2000",
                         "PS5HIP PASS kernel=vector_add exact_values=73 guards=unchanged allocations=closed"]
        expected.append("PS5HIP PASS kernel=vector_add_twice dispatches=2 exact_values=146 guards=unchanged allocations=closed")
    elif kernel in ("wave_full", "wave_partial"):
        expected += [f"PS5HIP SUBMIT target={target} kernel={kernel} invocations=64 output_words=256 groups=1 timeout_ms=2000",
                     f"PS5HIP PASS kernel={kernel} exact_values=256 guards=unchanged allocations=closed"]
    elif kernel in ("lds_static", "lds_dynamic", "lds_mixed"):
        static = 0 if kernel == "lds_dynamic" else 256
        dynamic, allocated, stride = (0, 1024, 0) if kernel == "lds_static" else (1280, 2048, 5)
        expected += [f"PS5HIP SUBMIT target={target} kernel={kernel} invocations=16448 output_words=257 groups=257 static_lds_bytes={static} dynamic_lds_bytes={dynamic} allocated_lds_bytes={allocated} stride={stride} timeout_ms=2000",
                     f"PS5HIP PASS kernel={kernel} exact_values=257 inputs=unchanged guards=unchanged allocations=closed"]
    elif kernel == "private_calls_buffer":
        expected += [f"PS5HIP SUBMIT target={target} kernel=private_calls_buffer invocations=16448 groups=257 user_sgprs=8 private_bytes=304 scratch_bytes=13107200 timeout_ms=2000",
                     "PS5HIP SCRATCH payload=modified guards=unchanged",
                     "PS5HIP PASS kernel=private_calls_buffer exact_values=16448 guards=unchanged allocations=closed"]
    elif kernel == "register_spill":
        expected += [f"PS5HIP SUBMIT target={target} kernel=register_spill invocations=64 output_words=4096 groups=1 private_bytes=212 scratch_bytes=9175040 timeout_ms=2000",
                     "PS5HIP SCRATCH payload=modified guards=unchanged",
                     "PS5HIP PASS kernel=register_spill exact_values=4096 guards=unchanged allocations=closed"]
    else:
        expected += [f"PS5HIP SUBMIT target={target} kernel={kernel} count=73 groups=2 timeout_ms=2000",
                     f"PS5HIP PASS kernel={kernel} exact_values=73 guards=unchanged allocations=closed"]
    if shutdown_probe or shell_exit_probe:
        expected += ["PS5HIP SHUTDOWN main-return=0", "PS5HIP SHUTDOWN atexit=observed",
                     "PS5HIP SHUTDOWN static-dtor=observed"]
    if shell_exit_probe:
        expected += ["PS5HIP SHELL_EXIT application-fini=completed", "PS5HIP SHELL_EXIT request=begin",
                     "PS5HIP SHELL_EXIT request-return=0"]
    if any("QUARANTINED" in line for line in lines):
        result = "quarantined"
    elif any("PS5HIP FAIL" in line or "PS5HIP mismatch" in line for line in lines):
        result = "failed"
    elif data == ("\n".join(expected) + "\n").encode("utf-8"):
        result = "application-pass"
    else:
        result = "incomplete-or-unexpected"
    return {"application_result": result, "target": target, "kernel": kernel,
            "log_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "hardware_gate_passed": False,
            "unproven": ["log freshness", "remote artifact identity", "native process status",
                         "title teardown", "service health"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--target", choices=["gfx1013", "gfx1030"], required=True)
    parser.add_argument("--kernel", choices=["runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "register_spill", "lds_static", "lds_dynamic", "lds_mixed", "private_calls_buffer"], required=True)
    parser.add_argument("--shutdown-probe", action="store_true")
    parser.add_argument("--shell-exit-probe", action="store_true")
    args = parser.parse_args()
    with args.log.open("rb") as source:
        data = source.read(65537)
    result = check_log(data, args.target, args.kernel, args.shutdown_probe, args.shell_exit_probe)
    print(json.dumps(result, indent=2))
    return 0 if result["application_result"] == "application-pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
