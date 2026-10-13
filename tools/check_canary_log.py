#!/usr/bin/env python3
"""Check a captured application log. This does not certify deployment or teardown."""
import argparse
import hashlib
import json
import pathlib
import math_native_fixture as math_fixture


def check_log(data, target, kernel, shutdown_probe=False, shell_exit_probe=False):
    if shutdown_probe and kernel != "runtime":
        raise ValueError("shutdown probe belongs to the CPU runtime case")
    if target not in ("gfx1013", "gfx1030") or kernel not in ("runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "fp32_math", "hip_vector", "hip_routes", "hip_generic", "hipify"):
        raise ValueError("unsupported canary profile")
    if kernel in ("hip_vector", "hip_routes", "hip_generic", "hipify") and (target != "gfx1013" or not shell_exit_probe):
        raise ValueError("HIP vector requires the measured gfx1013 shell-managed profile")
    if len(data) > 65536:
        raise ValueError("canary log exceeds bounded result size")
    lines = data.decode("utf-8", errors="strict").splitlines()
    expected = [f"PS5HIP START target={target} kernel={kernel}",
                "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass"]
    if kernel == "hip_vector":
        expected.insert(1, "HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0")
        expected += ["VECTOR_COMPLETE sequence=1 n=73 blocking=true",
                     "VECTOR_COMPLETE sequence=2 n=73 blocking=true",
                     "VECTOR_CLIENT_PASS exact_values=146 immutable_inputs=true full_guards=true blocking=true hardware_validated=false",
                     "HIP_TITLE COMPUTE complete=true dispatches=2 exact_values=146 finalization=pending"]
    elif kernel == "hip_routes":
        expected.insert(1, "HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0")
        for round in (1, 2):
            for route, name in enumerate(("routes_integer", "routes_lds", "routes_spill", "routes_geometry")):
                sequence = (round - 1) * 4 + route + 1
                expected += [f"RUNTIME_COMPLETE sequence={sequence} kernel={name} blocking=true",
                             f"ROUTES_COMPLETE round={round} route={route} blocking=true full_guards=true inputs=unchanged"]
        expected += ["ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true",
                     "HIP_TITLE COMPUTE complete=true dispatches=8 word_comparisons=393216 finalization=pending"]
    elif kernel == "hip_generic":
        expected.insert(1,"HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=2 constructor_gpu_calls=0")
        completed=0
        for sequence,count in enumerate((0,1,3,7)):
            for iteration in range(count):
                completed+=1
                name="affine_words" if iteration%2==0 else "block_sum"
                expected.append(f"RUNTIME_COMPLETE sequence={completed} kernel={name} blocking=true")
            expected.append(f"GENERIC_SEQUENCE index={sequence} requested={count} observed={count} completed={completed} prebackend={int(sequence<2)}")
        expected += ["GENERIC_CLIENT_PASS launches=11 sequences=0,1,3,7 variable_sizes=true full_guards=true inputs_unchanged=true",
                     "HIP_TITLE COMPUTE complete=true dispatches=11 sequences=0,1,3,7 finalization=pending"]
    elif kernel == "hipify":
        expected.insert(1,"HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0")
        expected += [f"RUNTIME_COMPLETE sequence={i} kernel=translated_add blocking=true" for i in range(1,7)]
        expected += ["TRANSLATED_CLIENT_PASS launches=6 exact_u32=true inputs_unchanged=true full_guards=true",
                     "HIP_TITLE COMPUTE complete=true dispatches=6 exact_values=662 word_comparisons=2238 finalization=pending"]
    elif kernel == "runtime":
        expected.append("PS5HIP PASS kernel=runtime gpu_api_calls=0")
    elif kernel == "fp32_math":
        contract = math_fixture.profile(target)["contract_sha256"]
        expected += [f"PS5HIP SUBMIT target={target} kernel=fp32_math inputs=257 output_words=1542 groups=5 kernarg_bytes=280 timeout_ms=2000",
                     f"PS5HIP PASS kernel=fp32_math numeric_values=1542 contract={contract} inputs=unchanged guards=unchanged allocations=closed"]
    elif kernel == "vector_add_twice":
        for dispatch in (1, 2):
            expected += [f"PS5HIP DISPATCH index={dispatch}/2 kernel=vector_add",
                         f"PS5HIP SUBMIT target={target} kernel=vector_add count=73 groups=2 timeout_ms=2000",
                         "PS5HIP PASS kernel=vector_add exact_values=73 guards=unchanged allocations=closed"]
        expected.append("PS5HIP PASS kernel=vector_add_twice dispatches=2 exact_values=146 guards=unchanged allocations=closed")
    elif kernel in ("wave_full", "wave_partial"):
        expected += [f"PS5HIP SUBMIT target={target} kernel={kernel} invocations=64 output_words=256 groups=1 timeout_ms=2000",
                     f"PS5HIP PASS kernel={kernel} exact_values=256 guards=unchanged allocations=closed"]
    else:
        expected += [f"PS5HIP SUBMIT target={target} kernel={kernel} count=73 groups=2 timeout_ms=2000",
                     f"PS5HIP PASS kernel={kernel} exact_values=73 guards=unchanged allocations=closed"]
    if shutdown_probe or shell_exit_probe:
        expected += ["PS5HIP SHUTDOWN main-return=0", "PS5HIP SHUTDOWN atexit=observed",
                     "PS5HIP SHUTDOWN static-dtor=observed"]
    if shell_exit_probe:
        if kernel == "hip_vector":
            expected += ["VECTOR_UNREGISTER after_main=true active=0",
                         "VECTOR_FINAL registered=1 removed=1 completed=2 active=0 terminal=0 main_finished=1 hardware_validated=false",
                         "HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed exact_values=146"]
        elif kernel == "hip_routes":
            expected += ["VECTOR_UNREGISTER after_main=true active=0",
                         "VECTOR_FINAL registered=1 removed=1 completed=8 active=0 terminal=0 main_finished=1 hardware_validated=false",
                         "HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=8"]
        elif kernel == "hip_generic":
            expected += ["VECTOR_UNREGISTER after_main=true active=0",
                         "VECTOR_FINAL registered=1 removed=1 completed=11 active=0 terminal=0 main_finished=1 hardware_validated=false",
                         "HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=11"]
        elif kernel == "hipify":
            expected += ["VECTOR_UNREGISTER after_main=true active=0",
                         "VECTOR_FINAL registered=1 removed=1 completed=6 active=0 terminal=0 main_finished=1 hardware_validated=false",
                         "HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=6"]
        expected += ["PS5HIP SHELL_EXIT application-fini=completed", "PS5HIP SHELL_EXIT request=begin",
                     "PS5HIP SHELL_EXIT request-return=0"]
    if any("QUARANTINED" in line for line in lines):
        result = "quarantined"
    elif any("PS5HIP FAIL" in line or "PS5HIP mismatch" in line or
             (kernel in ("hip_vector", "hip_routes", "hip_generic", "hipify") and line.startswith(("VECTOR_ERROR", "VECTOR_CLIENT_REJECT", "ROUTES_CLIENT_REJECT", "GENERIC_CLIENT_FAILURE", "TRANSLATED_CLIENT_FAIL", "VECTOR_CLEANUP_FAILURE", "HIP_TITLE FAIL")))
             for line in lines):
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
    parser.add_argument("--kernel", choices=["runtime", "indexed_store", "vector_add", "vector_add_twice", "wave_full", "wave_partial", "fp32_math", "hip_vector", "hip_routes", "hip_generic", "hipify"], required=True)
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
