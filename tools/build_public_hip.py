#!/usr/bin/env python3
"""Build the native public-HIP startup title from this checkout and locked sources."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile

from native_vector_title import audit_roundtrip
from prepare_public_hip_resources import prepared as resources_prepared
from prepare_public_hip_embedded import prepared as embedded_prepared
from embedded_module_linker import patch_linker as embedded_linker
from qualify_public_hip_aggregate import compiler_inputs as aggregate_compiler_inputs
from qualify_public_hip_rmsnorm import fixture_inputs as rmsnorm_fixture_inputs, checked_inputs, checked_tools
import build_llm_rmsnorm_reference as rms_reference
from public_hip_q4_assets import assets as q4_assets
from public_hip_weighted_qkv_assets import assets as weighted_qkv_assets

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
DEPS = ROOT / ".deps"
SOURCE_UNITS = (
    "hipamd/src/fixme.cpp", "rocclr/compiler/lib/utils/options.cpp",
    "rocclr/device/appprofile.cpp", "rocclr/device/blitcl.cpp",
    "rocclr/device/device.cpp", "rocclr/device/devkernel.cpp",
    "rocclr/device/devprogram.cpp", "rocclr/elf/elf.cpp",
    "rocclr/os/alloc.cpp", "rocclr/os/os.cpp", "rocclr/os/os_posix.cpp",
    "rocclr/platform/activity.cpp", "rocclr/platform/agent.cpp",
    "rocclr/platform/command.cpp", "rocclr/platform/commandqueue.cpp",
    "rocclr/platform/context.cpp", "rocclr/platform/interop_gl.cpp",
    "rocclr/platform/kernel.cpp", "rocclr/platform/memory.cpp",
    "rocclr/platform/ndrange.cpp", "rocclr/platform/program.cpp",
    "rocclr/platform/runtime.cpp", "rocclr/thread/monitor.cpp",
    "rocclr/thread/semaphore.cpp", "rocclr/thread/thread.cpp",
    "rocclr/utils/debug.cpp", "rocclr/utils/flags.cpp",
)
PROJECT_UNITS = (
    "ports/clr/ps5-clr-tls.cpp", "ports/clr/ps5-device.cpp", "ports/clr/ps5-embedded.cpp",
    "tools/public_hip_app_crt.cpp", "tools/clr_gpu/watchdog.cpp",
    "src/code_object.cpp", "src/fat_binary.cpp", "src/host_registration.cpp", "src/host_launch.cpp",
    "src/launch.cpp", "src/native_memory.cpp", "src/native_data.cpp",
    "src/native_launch.cpp", "src/submission.cpp", "src/hip_launch_profile.cpp",
    "src/agc_backend.cpp", "tools/native_shell_exit_probe.cpp",
    "tools/public_hip_bootstrap_console_title.cpp",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_inputs():
    names = {"dependencies.json", "elevation-inputs.json", "controller-inputs.json",
             "title.json", "assets/param.json", "assets/icon0.png",
             "tools/public_hip_weighted_qkv_console_title.cpp", "tools/public_hip_weighted_qkv_assets.py",
             "tools/audit_public_hip_weighted_qkv_native.py", "tools/audit_public_hip_rmsnorm_native.py",
             "tests/public_hip_weighted_qkv_assets_test.py", "tests/public_hip_weighted_qkv_build_test.py",
             "tests/public_hip_weighted_qkv_native_audit_test.py", "tests/public_hip_compile_inputs_test.py", "tools/verify_weighted_qkv_readbacks.py",
             "tests/weighted_qkv_readbacks_test.py", "tools/build_llm_weighted_qkv_reference.py",
             "tests/llm_weighted_qkv_reference.cpp", "tests/llm_weighted_qkv_reference_test.py",
             "tools/build_llm_weighted_rmsnorm_fixture.py", "tests/llm_weighted_rmsnorm_fixture_test.py",
             "tools/build_llm_q4_halfword_fixture.py", "tests/llm_q4_halfword_fixture_test.py",
             "tools/public_hip_q4_console_title.cpp", "tools/public_hip_q4_assets.py",
             "tests/public_hip_q4_assets_test.py", "tests/public_hip_q4_build_test.py",
             "tools/verify_public_hip_q4.py", "tests/q4_dump_test.py",
             "tools/public_hip_rmsnorm_console_title.cpp", "tools/qualify_public_hip_rmsnorm.py",
             "tools/build_llm_rmsnorm_fixture.py", "tools/build_llm_rmsnorm_reference.py",
             "tests/public_hip_rmsnorm_build_test.py", "tests/rmsnorm_dump_test.py",
             "tools/build_public_hip.py", "tools/prepare_clr_gpu_source.py",
             "tools/public_hip_embedded_console_title.cpp", "tools/embedded_module_linker.py",
             "tools/audit_public_hip_embedded_native.py", "tools/build_public_hip_embedded_fixture.py",
             "tools/probe_host_registration.py", "tools/native_multi_query.py",
             "tools/build_native_vector_title.py", "tools/native_vector_title.py",
             "tests/fat_binary_fixture_test.py", "tests/public_hip_embedded.hip",
             "tools/public_hip_aggregate_console_title.cpp", "tools/qualify_public_hip_aggregate.py",
             "tests/hip_aggregate_fixture.h", "tests/hip_aggregate_fixture.hip",
             "tests/hip_aggregate_stub_observer.cpp", "tests/hip_aggregate_compiler_test.py",
             "tests/public_hip_aggregate_qualification_test.py",
             "tests/public_hip_embedded_build_test.py", "tests/public_hip_embedded_native_audit_test.py",
             ".deps/sources/native-tools/tooling/native/ps5-pie.ld",
             "tools/prepare_public_hip_resources.py", "tools/prepare_public_hip_command_stats.py",
             "tools/prepare_public_hip_static_module.py", "tools/prepare_public_hip_embedded.py",
             "tools/prepare_hip_admission.py", "tools/guard_native_hip_gl.py",
             "tools/native_runtime.h", "tools/native_shell_exit_probe.cpp",
             "tools/public_hip_app_crt.cpp", "tools/public_hip_bootstrap_console_title.cpp",
             "tools/ps5-clangxx", "patches/native-import-libraries.patch",
             "tests/native_import_libraries_test.py", "tests/public_hip_imports_test.py",
             "tools/public_hip_memory_console_title.cpp", "tools/prepare_public_hip_memory.py",
             "tools/public_hip_allocation_lifetime_fixture.h",
             "tools/public_hip_copy_gpu_fixture.h", "tools/public_hip_copy_recovery_fixture.h", "tools/public_hip_copy_gpu_console_title.cpp",
             "tools/public_hip_capture_gpu_console_title.cpp", "tools/public_hip_graph_gpu_fixture.h", "tools/public_hip_graph_gpu_console_title.cpp",
             "tools/public_hip_stream_gpu_fixture.h", "tools/public_hip_stream_console_title.cpp",
             "tools/public_hip_graph_fixture.h", "tools/public_hip_graph_console_title.cpp", "tools/public_hip_fill_console_title.cpp", "tools/public_hip_copy_console_title.cpp", "tools/prepare_public_hip_copy.py",
             "tools/prepare_public_hip_module.py", "tools/prepare_public_hip_symbols.py",
             "tools/prepare_public_hip_capture.py", "tools/prepare_public_hip_fill.py", "tools/prepare_public_hip_rect.py", "tools/prepare_public_hip_aot.py", "tools/prepare_public_hip_launch.py", "tools/public_hip_module_console_title.cpp", "tools/public_hip_launch_console_title.cpp",
             "kernels/thread_words.hip",
             "tools/build_canaries.py", "kernels/canary.hip"}
    for folder in ("src", "include", "ports/clr", "ports/hip", "tools/clr_gpu"):
        names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).rglob("*")
                     if path.is_file() and path.suffix in
                     (".cpp", ".h", ".hpp", ".inc", ".patch", ".json", ".ld"))
    return {name: sha(ROOT / name) for name in sorted(names)}


def run(output, label, argv):
    process = subprocess.run([str(arg) for arg in argv], cwd=ROOT, capture_output=True,
                             text=True, timeout=300,
                             env=dict(os.environ, DOTNET_ROLL_FORWARD="Major",
                                      DOTNET_CLI_TELEMETRY_OPTOUT="1"))
    (output / (label + ".stdout")).write_text(process.stdout)
    (output / (label + ".stderr")).write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"{label} failed ({process.returncode}): {process.stderr[-3000:]}")


def source_tree(output, fixture="bootstrap"):
    base = output / "source-base"
    public = output / "source-public"
    if not (base / "receipt.json").exists():
        if base.exists():
            raise RuntimeError(f"incomplete source preparation: {base}")
        run(output, "prepare-clr", [sys.executable, ROOT / "tools/prepare_clr_gpu_source.py",
                                    "--output", base])
    if not (public / "public-hip-preparation.json").exists():
        if public.exists():
            raise RuntimeError(f"incomplete public HIP preparation: {public}")
        run(output, "prepare-public", [sys.executable, ROOT / "tools/prepare_hip_admission.py",
                                       "--base-source", base / "source",
                                       "--hip-source", DEPS / "sources/clr", "--output", public])
    correlation = output / "hip-correlation.sha256"
    if not correlation.exists():
        patch = ROOT / "ports/clr/ps5-hip-correlation.patch"
        run(output, "correlation", ["patch", "--batch", "--forward", "-p1",
                                    "-d", public, "-i", patch])
        correlation.write_text(sha(public / "hipamd/src/hip_prof_api.h") + "\n")
    if correlation.read_text().strip() != sha(public / "hipamd/src/hip_prof_api.h"):
        raise RuntimeError("HIP correlation source changed")
    guard = output / "hip-gl-guard.sha256"
    if not guard.exists():
        run(output, "guard-hip-gl", [sys.executable, ROOT / "tools/guard_native_hip_gl.py", public])
        guard.write_text(sha(public / "hipamd/src/hip_gl.cpp") + "\n")
    if guard.read_text().strip() != sha(public / "hipamd/src/hip_gl.cpp"):
        raise RuntimeError("guarded HIP GL source changed")
    # Restore the exact previous-stage bytes so its existing hash guards also run on reuse.
    if resources_prepared(public):
        run(output, "restore-resource-base", [sys.executable, ROOT / "tools/prepare_public_hip_resources.py", public, "--reverse"])
    has_embedded = embedded_prepared(public)
    if not has_embedded:
        run(output, "prepare-module", [sys.executable, ROOT / "tools/prepare_public_hip_aot.py", public])
    run(output, "prepare-symbols", [sys.executable, ROOT / "tools/prepare_public_hip_symbols.py", public])
    run(output, "prepare-rect", [sys.executable, ROOT / "tools/prepare_public_hip_rect.py", public])
    run(output, "prepare-launch", [sys.executable, ROOT / "tools/prepare_public_hip_launch.py", public])
    run(output, "prepare-capture", [sys.executable, ROOT / "tools/prepare_public_hip_capture.py", public])
    run(output, "prepare-command-stats", [sys.executable, ROOT / "tools/prepare_public_hip_command_stats.py", public])
    if not has_embedded:
        run(output, "prepare-static-module", [sys.executable, ROOT / "tools/prepare_public_hip_static_module.py", public])
    run(output, "prepare-embedded", [sys.executable, ROOT / "tools/prepare_public_hip_embedded.py", public])
    run(output, "prepare-resources", [sys.executable, ROOT / "tools/prepare_public_hip_resources.py", public])
    return public


def compiler_flags(source):
    paths = [
        "opencl/khronos/headers/opencl2.2/CL", "opencl/khronos/headers/opencl2.2",
        "opencl/khronos/headers", "opencl/khronos", "opencl", "opencl/amdocl",
        "rocclr", "rocclr/include", "rocclr/compiler/lib",
        "rocclr/compiler/lib/include", "rocclr/compiler/lib/backends/common",
        "rocclr/device", "rocclr/elf", "rocclr/platform", "hipamd/include",
        "hipamd/src",
    ]
    includes = [source / name for name in paths]
    includes += [DEPS / "sources/hip-headers/include", ROOT / "ports/hip", ROOT / "include",
                 ROOT / "ports/clr", ROOT / "tools", DEPS / "elevation/client"]
    if not all(path.is_dir() for path in includes):
        raise RuntimeError("native compiler include path is missing")
    defines = [
        "LITTLEENDIAN_CPU", "HAVE_CL2_HPP", "OPENCL_MAJOR=2", "OPENCL_MINOR=1",
        "OPENCL_C_MAJOR=2", "OPENCL_C_MINOR=0", "CL_TARGET_OPENCL_VERSION=220",
        "CL_USE_DEPRECATED_OPENCL_1_0_APIS", "CL_USE_DEPRECATED_OPENCL_1_1_APIS",
        "CL_USE_DEPRECATED_OPENCL_1_2_APIS", "CL_USE_DEPRECATED_OPENCL_2_0_APIS",
        "__HIP_PLATFORM_AMD__", "PS5HIP_CLR_PLATFORM", "PS5HIP_CLR_NO_GRAPHICS",
        "WITH_PS5_DEVICE", "PS5HIP_WITH_ELEVATION=1", "PS5HIP_SHELL_EXIT_PROBE=1",
        "PS5HIP_CLR_GPU_TITLE=1", "PS5HIP_PUBLIC_HIP", "DISABLE_DIRECT_DISPATCH",
    ]
    return [ROOT / "tools/ps5-clangxx", *["-I" + str(p) for p in includes],
            "-std=c++20", "-O0", "-fPIC", "-pthread", "-ferror-limit=0",
            "-ffunction-sections", "-fdata-sections", *["-D" + d for d in defines]]


def compile_bound_unit(output, path, obj, flags, label):
    """Bind the compiler's header closure before compiling, then verify it."""
    def dependencies(depfile):
        text = depfile.read_text().replace("\\\n", " ")
        return {str(Path(p).resolve()) for p in shlex.split(text.split(":", 1)[1])}

    before = obj.with_suffix(".headers.d")
    actual = obj.with_suffix(".compiled.d")
    run(output, label + "-headers", [*flags, "-c", "-M", path, "-MF", before])
    headers = {p: sha(p) for p in dependencies(before)}
    if str(path.resolve()) not in headers:
        raise RuntimeError("compiler dependency inventory omits source")
    run(output, label, [*flags, "-MD", "-MF", actual, "-c", path, "-o", obj])
    if dependencies(actual) != set(headers):
        raise RuntimeError("actual compiler dependencies changed")
    if any(sha(p) != digest for p, digest in headers.items()):
        raise RuntimeError("compiler input changed during compilation")
    return {"source": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "source_sha256": headers[str(path.resolve())],
            "object": str(obj.relative_to(output)), "object_sha256": sha(obj),
            "headers": headers}


def compile_units(output, source, fixture="bootstrap", client_phase="vocab"):
    if client_phase not in ("vocab", "model-gate") or (fixture != "llm_client" and client_phase != "vocab"):
        raise ValueError("invalid client compile phase")
    object_dir = output / "objects"
    object_dir.mkdir(exist_ok=True)
    units = [source / name for name in SOURCE_UNITS]
    units += [ROOT / name for name in PROJECT_UNITS]
    if fixture != "bootstrap":
        units[units.index(ROOT / "tools/public_hip_bootstrap_console_title.cpp")] = ROOT / ("tools/public_hip_" + fixture + "_console_title.cpp")
    units += [DEPS / "elevation/client/elevation.cpp"]
    hip = sorted((source / "hipamd/src").glob("*.cpp"))
    hip = [path for path in hip if path.stem not in ("fixme", "hip_runtime")]
    if len(units) != 46 or len(hip) != 30:
        raise RuntimeError(f"expected 46 platform and 30 HIP units; got {len(units)} and {len(hip)}")
    units += hip
    flags = compiler_flags(source)

    def compile_one(pair):
        index, path = pair
        obj = object_dir / (f"{index:02d}-" + path.stem + ".o")
        extra = (["-include", "hip/hip_runtime.h", "-include", "hip/hip_deprecated.h",
                  "-include", "hip/amd_detail/amd_hip_gl_interop.h"]
                 if path.stem in ("hip_api_trace", "hip_table_interface",
                                  "hip_table_interface_c") else [])
        if fixture == "llm_client" and path.name == "public_hip_llm_client_console_title.cpp":
            if client_phase == "model-gate":
                extra.append("-DPS5HIP_LLM_AUTHENTICATED_MODEL_GATE=1")
            client = DEPS / "sources/llama-client"
            extra += ["-I" + str(p) for p in (ROOT / "ports/ggml", client / "include",
                      client / "ggml/include", client / "ggml/src")]
        return compile_bound_unit(output, path, obj, [*flags, *extra], f"compile-{index:02d}")

    # Fail quickly on the selected client before spending time on the upstream closure.
    title_index = units.index(ROOT / ("tools/public_hip_" + fixture + "_console_title.cpp"))
    title_record = compile_one((title_index, units[title_index]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(compile_one, ((index, path) for index, path in enumerate(units)
                                              if index != title_index)))
    records.insert(title_index, title_record)
    if len(records) != 76 or len({record["object"] for record in records}) != 76:
        raise RuntimeError("not a clean 76-object closure")
    return records


def native_stubs(output):
    names = {"libSceAgc": 'extern "C" {int sceAgcInit(unsigned){return -1;} int sceAgcSuspendPoint(){return -1;}}\n',
             "libSceAgcDriver": 'extern "C" {int sceAgcDriverSubmitDcb(void*){return -1;}}\n'}
    for name, text in names.items():
        source = output / (name + ".cpp")
        source.write_text(text)
        obj = output / (name + ".o")
        run(output, "stub-compile-" + name,
            [ROOT / "tools/ps5-clangxx", "-fPIC", "-c", source, "-o", obj])
        run(output, "stub-link-" + name,
            ["ld.lld-18", "-shared", "-soname", name + ".prx", obj,
             "-o", output / (name + ".so")])


def embedded_fixture_inputs(folder):
    folder = Path(folder).resolve()
    receipt_path = folder / "receipt.json"
    report = json.loads(receipt_path.read_text())
    if (report.get("classification") != "public-hip-genuine-embedded-compiler-fixture" or
            any(report.get(name) is not False for name in ("runtime_executed", "native_executed", "gpu_executed"))):
        raise ValueError("expected an offline genuine compiler fixture")
    for field in ("inputs", "artifacts", "tools"):
        if not isinstance(report.get(field), dict) or not report[field]:
            raise ValueError("missing compiler fixture " + field)
    for name, digest in report["inputs"].items():
        if sha(name) != digest:
            raise ValueError("compiler fixture input changed: " + name)
    for name, digest in report["artifacts"].items():
        path = (folder / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder) or sha(path) != digest:
            raise ValueError("compiler fixture artifact changed: " + name)
    for tool in report["tools"].values():
        if sha(tool["path"]) != tool["sha256"]:
            raise ValueError("compiler fixture tool changed: " + tool["path"])
    name = report["profiles"]["ps5"]["a"]["object"]
    if name != "ps5/public-embedded.o" or name not in report["artifacts"]:
        raise ValueError("expected exact native compiler module")
    for relative in ("tests/public_hip_embedded.hip", "tools/build_public_hip_embedded_fixture.py"):
        if report["inputs"].get(str(ROOT / relative)) != sha(ROOT / relative):
            raise ValueError("compiler fixture source identity missing: " + relative)
    return {"receipt": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "object": str(folder / name), "object_sha256": report["artifacts"][name]}


def compiler_fixture_inputs(folder, fixture):
    if fixture == "embedded":
        return embedded_fixture_inputs(folder)
    if fixture != "aggregate":
        raise ValueError("compiler objects require embedded or aggregate fixture")
    folder = Path(folder).resolve()
    verified = aggregate_compiler_inputs(folder)
    receipt = folder / "receipt.json"
    obj = folder / "ps5-fixture.o"
    return {"receipt": str(receipt), "receipt_sha256": verified[str(receipt)],
            "object": str(obj), "object_sha256": verified[str(obj)]}


RMSNORM_CPU_IMPLEMENTATIONS = ("ggml/src/ggml.c", "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu.cpp", "ggml/src/ggml-cpu/ops.cpp", "ggml/src/ggml-cpu/vec.cpp")


def rmsnorm_cpu_provenance(cpu, commands, model_digest):
    """Bind current model/source/tool bytes; not independent historical build proof."""
    identity = cpu.get("identity", {})
    inputs = checked_inputs(identity, "inputs", [ROOT / n for n in
        ("llm-inputs.json", "tools/build_llm_cpu_reference.py", "tools/fetch_locked_deps.py")])
    inputs.update(checked_tools(identity, "tools", ("cmake", "ninja", "clang-18", "clang++-18")))
    runs = [c.get("argv") for c in commands if c.get("label") in ("run1", "run2")]
    if (len(runs) != 2 or any(not isinstance(a,list) or len(a) != 4 for a in runs) or
            runs[0][:3] != runs[1][:3] or not Path(runs[0][1]).is_absolute()):
        raise ValueError("CPU model invocation identity missing")
    model = Path(runs[0][1]).resolve()
    if sha(model) != model_digest:
        raise ValueError("CPU model bytes differ")
    inputs[str(model)] = model_digest
    client = ROOT / ".deps/sources/llama-client"
    git = shutil.which("git")
    if not git:
        raise ValueError("git required to verify CPU implementation")
    inputs[str(Path(git).resolve())] = sha(git)
    def query(*args):
        return subprocess.check_output([git,"-C",str(client),*args],text=True).strip()
    if query("rev-parse","HEAD") != cpu["source_revision"] or query("status","--porcelain","--untracked-files=no"):
        raise ValueError("CPU implementation tree changed")
    tracked = set(query("ls-files").splitlines())
    for name in RMSNORM_CPU_IMPLEMENTATIONS:
        if name not in tracked:
            raise ValueError("CPU implementation source missing")
        path = client / name
        inputs[str(path)] = sha(path)
    return inputs


def rmsnorm_reference_inputs(folder):
    """Authenticate frozen CPU assets and their source/profile closure before packaging."""
    folder = Path(folder).resolve()
    receipt = folder / "receipt.json"
    report = json.loads(receipt.read_text())
    if (report.get("classification") != "host-cpu-rmsnorm-reference" or
            report.get("gpu_executed") is not False or report.get("fresh_repeats_bitwise_equal") is not True or
            report.get("negative_controls") != ["noop rejected", "single-element corruption rejected"]):
        raise ValueError("unqualified CPU RMSNorm reference")
    required = ["tools/build_llm_rmsnorm_reference.py", "tests/llm_rmsnorm_reference.cpp",
                "tests/llm_rmsnorm_reference_test.py", "llm-inputs.json",
                "evidence/offline-llm-model-conversion.json", "tests/fixtures/llm-reference-prompts.json"]
    inputs = checked_inputs(report, "identities", [ROOT / n for n in required])
    inputs[str(receipt)] = sha(receipt)
    lock = json.loads((ROOT / "llm-inputs.json").read_text())
    conversion = json.loads((ROOT / "evidence/offline-llm-model-conversion.json").read_text())
    if (report.get("source_revision") != lock["client"]["revision"] or
            report.get("model_sha256") != conversion["outputs"]["tinyllama-q4_0.gguf"]["sha256"]):
        raise ValueError("reference source/model identity differs")
    cpu_receipts = [(Path(n), json.loads(Path(n).read_text())) for n in inputs if Path(n).name == "receipt.json" and n != str(receipt)]
    cpu_receipts = [(p, j) for p, j in cpu_receipts if j.get("classification") == "host-cpu-reference-tools-built"]
    if len(cpu_receipts) != 1:
        raise ValueError("one bound CPU build receipt required")
    cpu_path, cpu = cpu_receipts[0]
    if (cpu.get("identity", {}).get("source_revision") != report["source_revision"] or
            cpu.get("identity", {}).get("source_url") != lock["client"]["source_url"]):
        raise ValueError("CPU build source identity differs")
    flags = cpu.get("identity", {}).get("flags", [])
    if cpu.get("source_revision") != report["source_revision"] or not all(f in flags for f in (
            "-DGGML_BACKEND_DL=OFF", "-DGGML_CPU=ON", "-DGGML_HIP=OFF", "-DGGML_CUDA=OFF", "-DGGML_BLAS=OFF")):
        raise ValueError("CPU reference backend differs")
    if not cpu.get("artifacts"):
        raise ValueError("CPU build artifacts missing")
    for name, digest in cpu["artifacts"].items():
        path = (cpu_path.parent / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(cpu_path.parent) or inputs.get(str(path)) != digest:
            raise ValueError("CPU artifact identity missing")
    for name in ("libggml.so", "libggml-cpu.so", "libggml-base.so"):
        if str((cpu_path.parent / "cmake/bin" / name).resolve()) not in inputs:
            raise ValueError("CPU runtime library identity missing")
    for path in (Path("/usr/bin/clang++-18").resolve(), Path(sys.executable).resolve()):
        if str(path) not in inputs:
            raise ValueError("CPU reference tool identity missing")
    artifacts = report.get("artifacts")
    required_artifacts = {"fixture", "fixture.d", "declared-profile.json", "tokens.txt", "commands.json",
                          "run1.stdout", "run2.stdout", "noop-rejection.txt", "corrupt-rejection.txt"}
    if not isinstance(artifacts, dict) or not required_artifacts <= artifacts.keys():
        raise ValueError("CPU reference artifacts missing")
    for name, digest in artifacts.items():
        path = (folder / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder) or sha(path) != digest:
            raise ValueError("CPU reference artifact changed")
        inputs[str(path)] = digest
    for name in shlex.split((folder / "fixture.d").read_text().replace(chr(92)+chr(10), " ").split(":", 1)[1]):
        if str(Path(name).resolve()) not in inputs:
            raise ValueError("CPU reference header identity missing")
    profile_tokens = json.loads((ROOT / "tests/fixtures/llm-reference-prompts.json").read_text())
    tokens = list(dict.fromkeys(t for c in profile_tokens["cases"] for t in c["tokens"]))[:17]
    profile = dict(width=2048, epsilon_f32=rms_reference.EPSILON, row_counts=[1,2,17],
                   embedding_tokens=tokens, threads=1, operator="GGML_OP_RMS_NORM unfused F32",
                   tolerance=rms_reference.TOLERANCE)
    if (len(tokens) != 17 or profile_tokens["model_revision"] != conversion["model_revision"] or
            report.get("profile") != profile or json.loads((folder / "declared-profile.json").read_text()) != profile or
            (folder / "tokens.txt").read_text() != " ".join(map(str,tokens))+"\n"):
        raise ValueError("frozen RMSNorm profile differs")
    commands = json.loads((folder / "commands.json").read_text())
    if ([c.get("label") for c in commands] != ["dependencies", "compile", "runtime-libraries", "run1", "run2"] or
            any(c.get("exit_code") != 0 or c.get("timed_out") for c in commands)):
        raise ValueError("CPU reference commands incomplete")
    provenance = rmsnorm_cpu_provenance(cpu, commands, report["model_sha256"])
    for name, digest in provenance.items():
        if name in inputs and inputs[name] != digest:
            raise ValueError("CPU provenance changed during validation")
        inputs[name] = digest
    for run in (1,2):
        if (folder / ("run%d.stdout" % run)).read_text().strip() != "GGML_RMS_NORM_CPU_PASS cases=6 width=2048 threads=1 gpu_execution=false":
            raise ValueError("CPU reference transcript differs")
    cases = report.get("cases", [])
    expected = [(family+"-"+str(rows),rows) for family in ("stress","embedding") for rows in (1,2,17)]
    if [(c.get("name"), c.get("rows")) for c in cases] != expected:
        raise ValueError("CPU reference cases differ")
    assets = {}
    for case in cases:
        name, rows = case["name"], case["rows"]
        for kind in ("input", "cpu"):
            relative = "run1/"+name+"."+kind+".f32le"
            path = folder / relative
            other = folder / ("run2/"+name+"."+kind+".f32le")
            if (artifacts.get(relative) != case[kind+"_sha256"] or
                    artifacts.get(str(other.relative_to(folder))) != case[kind+"_sha256"] or
                    path.stat().st_size != rows*2048*4 or other.read_bytes() != path.read_bytes()):
                raise ValueError("CPU reference tensor identity/size differs")
            rms_reference.read_f32(path, rows*2048)
            assets["rmsnorm/"+path.name] = str(path)
        fp64 = name+".fp64le"
        values = rms_reference.read_f32(folder / ("run1/"+name+".input.f32le"), rows*2048)
        reference = rms_reference.fp64_reference(values)
        import struct
        if artifacts.get(fp64) != case["fp64_sha256"] or (folder/fp64).read_bytes() != struct.pack("<"+"d"*len(reference),*reference):
            raise ValueError("independent FP64 reference differs")
        actual = rms_reference.read_f32(folder / ("run1/"+name+".cpu.f32le"), rows*2048)
        metrics = rms_reference.compare(actual, reference)
        if metrics != case["metrics"] or metrics["maximum_budget_fraction"] > 1/32:
            raise ValueError("CPU reference accuracy differs")
    return dict(receipt=str(receipt), receipt_sha256=inputs[str(receipt)], inputs=inputs, assets=assets,
                provenance_limit="Current locked source tree, tools, model and CPU artifacts are bound; this does not independently prove historical CPU build inputs.")


def rmsnorm_assets(fixture, reference):
    inputs = rmsnorm_fixture_inputs(fixture)
    reference = rmsnorm_reference_inputs(reference)
    module = str(Path(fixture).resolve() / "norm.hsaco")
    return dict(module_inputs=inputs, reference=reference,
                assets={"rmsnorm/norm.hsaco":module, **reference["assets"]})


def link_title(output, units, fixture="bootstrap", embedded=None):
    sdk = DEPS / "sdk/target/lib"
    static = ["libc++.a", "libc++abi.a", "libunwind.a", "libc.a",
              "libps5platform.a", "libdl.a", "libpthread.a"]
    imports = [sdk / (name + ".so") for name in
               ("libkernel", "libSceLibcInternal", "libSceNet", "libSceSystemService")]
    imports += [output / "libSceAgc.so", output / "libSceAgcDriver.so"]
    (output / "symbols.map").write_text("{ local: *; };\n")
    unwind = [argument for section in ("eh_frame", "eh_frame_hdr")
              for argument in ("--defsym", "__" + section + "_start=ADDR(." + section + ")",
                               "--defsym", "__" + section + "_end=ADDR(." + section + ")+SIZEOF(." + section + ")")]
    linker = ROOT / "ports/clr/ps5-pie.ld"
    extra_objects = []
    if fixture in ("embedded", "aggregate"):
        if embedded is None:
            raise ValueError("compiler title requires a verified compiler fixture")
        clean = (DEPS / "sources/native-tools/tooling/native/ps5-pie.ld").read_text()
        optional = linker.read_text().split("/* Audited optional provider absence", 1)
        if len(optional) != 2:
            raise ValueError("optional-provider linker definitions missing")
        linker = output / "embedded-pie.ld"
        linker.write_text(embedded_linker(clean, [Path(embedded["object"]).name]) +
                          "\n/* Audited optional provider absence" + optional[1])
        extra_objects = [Path(embedded["object"])]
    command = ["ld.lld-18", "-pie", "--no-undefined", "--gc-sections", "-T",
               linker, "--eh-frame-hdr", *unwind,
               "-z", "max-page-size=0x4000", "--hash-style=gnu", "--version-script",
               output / "symbols.map", "--exclude-libs=ALL", "-L", sdk, "-e", "_start",
               "-o", output / "title.elf", *[output / item["object"] for item in units], *extra_objects,
               "--start-group", *[sdk / name for name in static], "--end-group",
               "--as-needed", *imports]
    if fixture == "llm_client":
        command += ["--wrap=ggml_backend_sched_graph_compute_async",
                    "--undefined=__wrap_ggml_backend_sched_graph_compute_async"]
    run(output, "link", command)
    symbols = subprocess.check_output(["llvm-readelf-18", "-Ws", str(output / "title.elf")], text=True)
    required = ["hipInit", "hipGetDeviceCount", "ps5hip_clr_admit_device"]
    if fixture == "llm_client":
        required += ["hipMalloc", "hipFree", "hipMemcpy", "hipMemset", "hipDeviceSynchronize",
                     "llama_model_load_from_file", "llama_tokenize",
                     "__wrap_ggml_backend_sched_graph_compute_async"]
    if fixture in ("memory", "copy", "launch", "fill", "stream", "graph_gpu", "capture_gpu", "copy_gpu"):
        required += ["hipMalloc", "hipFree"]
    if fixture in ("copy", "launch", "fill", "stream", "graph_gpu", "capture_gpu", "copy_gpu"):
        required += ["hipMemcpy"]
    if fixture in ("module", "launch", "stream", "graph_gpu", "capture_gpu", "copy_gpu"):
        required += ["hipModuleLoad", "hipModuleGetFunction", "hipModuleUnload"]
    if fixture in ("rmsnorm", "q4", "weighted_qkv"):
        required += ["hipMalloc", "hipFree", "hipMemcpy", "hipModuleLoad", "hipModuleGetFunction",
                     "hipModuleUnload", "hipModuleLaunchKernel", "hipDeviceSynchronize"]
    if fixture == "launch":
        required += ["hipModuleLaunchKernel", "hipDeviceSynchronize"]
    if fixture == "stream":
        required += ["hipModuleLaunchKernel", "hipMemcpyAsync", "hipMemset", "hipStreamCreateWithFlags",
                     "hipStreamWaitEvent", "hipStreamSynchronize", "hipStreamQuery", "hipStreamDestroy",
                     "hipEventCreateWithFlags", "hipEventRecord", "hipEventSynchronize", "hipEventQuery",
                     "hipEventDestroy", "hipLaunchHostFunc"]
    if fixture == "fill":
        required += ["hipMemset", "hipMemsetD16", "hipMemsetD32Async", "hipMemsetAsync", "hipDeviceSynchronize"]
    if fixture in ("graph_gpu", "capture_gpu", "copy_gpu"):
        required += ["hipGraphAddKernelNode", "hipGraphAddHostNode", "hipGraphInstantiate", "hipGraphLaunch",
                     "hipGraphExecKernelNodeSetParams", "hipGraphNodeSetEnabled", "hipGraphDestroy", "hipGraphExecDestroy",
                     "hipStreamCreateWithFlags", "hipStreamSynchronize", "hipStreamDestroy"]
    if fixture in ("capture_gpu", "copy_gpu"):
        required += ["hipModuleLaunchKernel", "hipStreamBeginCapture", "hipStreamEndCapture", "hipLaunchHostFunc"]
    if fixture == "copy_gpu":
        required += ["hipMemcpyAsync", "hipMemset", "hipGraphAddMemcpyNode1D"]
    if fixture == "graph":
        required += ["hipGraphInstantiate", "hipGraphLaunch", "hipGraphExecHostNodeSetParams", "hipStreamBeginCapture", "hipStreamEndCapture", "hipGraphDestroy", "hipGraphExecDestroy"]
    if fixture in ("embedded", "aggregate"):
        required += ["hipMalloc", "hipFree", "hipMemcpy", "hipDeviceSynchronize", "hipLaunchKernel",
                     "launch_public_embedded" if fixture == "embedded" else "launch_aggregate_fixture",
                     "__hipRegisterFatBinary", "__hipRegisterFunction",
                     "__hipUnregisterFatBinary", "__hipPushCallConfiguration", "__hipPopCallConfiguration"]
    for name in required:
        if not any(line.split()[-1:] == [name] and " UND " not in line for line in symbols.splitlines()):
            raise RuntimeError(f"missing public bootstrap symbol {name}")


def native_tool(output):
    upstream = DEPS / "sources/native-tools/tooling/native"
    vendor = output / "native-tool-source"
    if vendor.exists():
        raise RuntimeError("native converter source output already exists; use a fresh build")
    shutil.copytree(upstream, vendor)
    run(output, "native-import-patch", ["patch", "--batch", "--fuzz=0", "-p1", "-d", vendor,
                                       "-i", ROOT / "patches/native-import-libraries.patch"])
    zlib = DEPS / "host-zlib-1.3.2/root"
    tool = output / "ps5-native-tool"
    sources = [vendor / name for name in ("native_app_builder.cpp", "self_container.cpp",
                                         "elf_object.cpp", "sce_module_writer.cpp")]
    run(output, "native-tool", ["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra",
                                "-Werror", "-I", zlib / "include", *sources,
                                zlib / "lib/libz.a", "-o", tool])
    return tool


def clean_room_libc(output, tool):
    vendor = DEPS / "sources/native-tools/tooling/native"
    builder = output / "libc-builder"
    run(output, "libc-builder", ["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra",
                                 "-Werror", vendor / "libc_builder.cpp", "-o", builder])
    raw = output / "libc.raw.elf"
    signed = output / "libc.prx"
    run(output, "libc-raw", [builder, vendor / "runtime/api-surface.txt",
                             vendor / "runtime/imports.txt", raw])
    run(output, "libc-sign", [tool, "self", "--sign", "--in", raw, "--out", signed])
    expected = (DEPS / "sources/native-tools/runtime/libc.prx.sha256").read_text().split()[0]
    if sha(signed) != expected:
        raise RuntimeError("clean-room libc differs from locked native-tools digest")
    return signed


def convert_title(output, tool, fixture="bootstrap"):
    sdk = DEPS / "sdk/target/lib"
    converted = output / "eboot.elf"
    signed = output / "eboot.bin"
    extracted = output / "extracted.elf"
    run(output, "convert", [tool, "link", "--in", output / "title.elf", "--out", converted,
                            "--stub-dir", sdk, "--stub", output / "libSceAgc.so",
                            "--stub", output / "libSceAgcDriver.so", "--file-name", "eboot.elf",
                            "--import-library", "libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil"])
    run(output, "verify-sce-imports", [sys.executable, ROOT / "tests/public_hip_imports_test.py", converted,
                                     *(["--no-gpu-imports"] if fixture in ("module", "fill", "copy", "graph") else [])])
    run(output, "sign", [tool, "self", "--sign", "--in", converted, "--out", signed])
    run(output, "extract", [tool, "self", "--extract", "--file", signed,
                            "--out", extracted])
    relation = audit_roundtrip(converted.read_bytes(), extracted.read_bytes())
    if not relation["matches_except_unmapped_sie_note"]:
        raise RuntimeError("FSELF roundtrip differs outside the unmapped SIE note")
    return signed, relation


def inventory(folder):
    if any(path.is_symlink() for path in folder.rglob("*")):
        raise RuntimeError("package contains a symlink")
    return {path.relative_to(folder).as_posix(): sha(path) for path in folder.rglob("*")
            if path.is_file()}


def validate_results_directory(results_dir):
    parts = results_dir[1:].split("/") if results_dir.startswith("/") else []
    if (not parts or not 2 <= len(results_dir) <= 240 or results_dir.endswith("/") or
            any(part in ("", ".", "..") for part in parts) or
            any(not 32 <= ord(char) <= 126 or char == "\\" for char in results_dir)):
        raise ValueError("results directory must be a bounded absolute console path")


def validate_client_assets(assets):
    phase = Path(assets["client-phase.txt"]).read_bytes() if "client-phase.txt" in assets else b""
    expected = {"vocab.gguf", "client-phase.txt"}
    if phase == b"model-gate\n":
        expected.add("model.gguf")
    elif phase != b"vocab\n":
        raise ValueError("invalid client phase bytes")
    if set(assets) != expected or any(Path(p).is_symlink() or not Path(p).is_file() for p in assets.values()):
        raise ValueError("invalid phase-specific client assets")
    if phase == b"model-gate\n":
        from fetch_llm_model import verify
        evidence = json.loads((ROOT / "evidence/offline-llm-model-conversion.json").read_text())["outputs"]["tinyllama-q4_0.gguf"]
        verify(Path(assets["model.gguf"]), evidence["bytes"], evidence["sha256"])


def copy_checked_package(packed, final):
    with Path(packed).open("rb") as stream:
        stream.seek(0x1055c)
        if stream.read(4) != bytes.fromhex("19015419"):
            raise RuntimeError("package format marker differs")
    if Path(final).exists() or Path(final).is_symlink():
        raise FileExistsError(final)
    shutil.copyfile(packed, final)


def package_title(output, image, libc, results_dir, code_object=None, extra_assets=None, client_assets=None):
    title = json.loads((ROOT / "title.json").read_text())["title_id"]
    validate_results_directory(results_dir)
    stage = output / "package"
    app = stage / "app"
    sources = {
        "eboot.bin": image,
        "lapy.elf": DEPS / "elevation/lapy/build" / ("owned_root_helper-" + title) / "lapy.elf",
        "sce_module/libc.prx": libc,
        "sce_sys/param.json": ROOT / "assets/param.json",
        "sce_sys/icon0.png": ROOT / "assets/icon0.png",
    }
    if client_assets is not None:
        validate_client_assets(client_assets)
        if code_object is not None or extra_assets:
            raise ValueError("client assets cannot mix legacy assets")
        sources.update({name: Path(path) for name, path in client_assets.items()})
    for name, path in (extra_assets or {}).items():
        if (not name.startswith(("rmsnorm/", "q4/", "weighted-qkv/")) or ".." in Path(name).parts or
                "\\" in name or Path(name).is_absolute()):
            raise ValueError("invalid numerical asset destination")
        sources[name] = Path(path)
    if code_object is not None:
        sources[code_object.name] = code_object
    if app.exists():
        expected = {name: sha(source) for name, source in sources.items()}
        expected["results-dir.txt"] = hashlib.sha256(results_dir.encode()).hexdigest()
        if inventory(app) != expected:
            raise RuntimeError("existing package stage differs from locked inputs")
    else:
        for name, source in sources.items():
            destination = app / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (app / "results-dir.txt").write_text(results_dir)
    if json.loads((app / "sce_sys/param.json").read_text())["titleId"] != title:
        raise RuntimeError("package title identity differs from title.json")
    before = inventory(app)
    version = json.loads((ROOT / "dependencies.json").read_text())["ufs2tool"]["revision"]
    ufs = DEPS / ("ufs2tool-" + version) / "bin/UFS2Tool.dll"
    command = ["dotnet", ufs]
    with tempfile.TemporaryDirectory(prefix="pack-", dir=stage) as temporary:
        temporary = Path(temporary)
        packed = temporary / (title + ".ffpkg")
        extracted = temporary / "extracted"
        run(output, "makefs", [*command, "makefs", "-S", "4096", "-b", "20%", "-t", "ffs",
                               "-o", "version=2,bsize=32768,fsize=4096,minfree=0,softupdates=0,optimization=space",
                               packed, app])
        run(output, "fsck", [*command, "fsck_ufs", "-f", "-n", packed])
        run(output, "package-extract", [*command, "extract", packed, extracted])
        if inventory(extracted) != before:
            raise RuntimeError("UFS extraction inventory differs from staged title")
        final = stage / (title + ".ffpkg")
        copy_checked_package(packed, final)
    return final, before


def validate_weighted_qkv_audit(output, expected_sha=None):
    from audit_public_hip_weighted_qkv_native import SOURCES
    path = output / "weighted-qkv-native-audit.json"
    digest = sha(path)
    if expected_sha is not None and digest != expected_sha:
        raise RuntimeError("weighted QKV audit identity changed")
    audit = json.loads(path.read_text())
    inputs = audit.get("inputs")
    required = [output / n for n in ("title.elf", "eboot.elf", "extracted.elf")]
    required += [ROOT / n for n in SOURCES]
    if (audit.get("status") != "PASS" or audit.get("classification") != "offline-public-hip-weighted-qkv-native-image-audit" or
            audit.get("native_executed") is not False or audit.get("audited_function_count") != 21 or
            not isinstance(inputs, dict) or not inputs or
            any(inputs.get(str(p.resolve())) != sha(p) for p in required) or
            any(sha(p) != h for p, h in inputs.items())):
        raise RuntimeError("weighted QKV image audit contract or inputs differ")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/public-hip")
    parser.add_argument("--fixture", choices=("bootstrap", "memory", "copy", "module", "launch", "fill", "graph", "stream", "graph_gpu", "capture_gpu", "copy_gpu", "embedded", "aggregate", "rmsnorm", "q4", "weighted_qkv"), default="bootstrap")
    parser.add_argument("--compiler-fixture", type=Path, help="Verified compiler fixture required for embedded or aggregate title")
    parser.add_argument("--rmsnorm-fixture", type=Path)
    parser.add_argument("--rmsnorm-reference", type=Path)
    parser.add_argument("--q4-fixture", type=Path)
    parser.add_argument("--q4-reference", type=Path)
    parser.add_argument("--weighted-rmsnorm-fixture", type=Path)
    parser.add_argument("--weighted-qkv-reference", type=Path)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "/user/homebrew/ps5-hip"))
    args = parser.parse_args()
    if (args.fixture in ("embedded", "aggregate")) != (args.compiler_fixture is not None):
        parser.error("--compiler-fixture is required only for --fixture embedded or aggregate")
    embedded = compiler_fixture_inputs(args.compiler_fixture, args.fixture) if args.compiler_fixture else None
    if ((args.fixture == "rmsnorm") != (args.rmsnorm_fixture is not None) or
            (args.fixture == "rmsnorm") != (args.rmsnorm_reference is not None)):
        parser.error("--rmsnorm-fixture and --rmsnorm-reference are required only for rmsnorm")
    rmsnorm = rmsnorm_assets(args.rmsnorm_fixture, args.rmsnorm_reference) if args.fixture == "rmsnorm" else None
    if ((args.fixture in ("q4", "weighted_qkv")) != (args.q4_fixture is not None) or
            (args.fixture == "q4") != (args.q4_reference is not None)):
        parser.error("--q4-fixture requires q4 or weighted_qkv; --q4-reference requires q4")
    if ((args.fixture == "weighted_qkv") != (args.weighted_rmsnorm_fixture is not None) or
            (args.fixture == "weighted_qkv") != (args.weighted_qkv_reference is not None)):
        parser.error("weighted_qkv requires --weighted-rmsnorm-fixture, --q4-fixture and --weighted-qkv-reference")
    q4 = q4_assets(args.q4_fixture, args.q4_reference) if args.fixture == "q4" else None
    weighted_qkv = weighted_qkv_assets(args.weighted_rmsnorm_fixture, args.q4_fixture, args.weighted_qkv_reference) if args.fixture == "weighted_qkv" else None
    initial_inputs = build_inputs()
    validate_results_directory(args.results_dir)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "build").resolve()) and not output.is_relative_to((ROOT / ".local").resolve()):
        parser.error("output must stay within build/ or .local/")
    output.mkdir(parents=True, exist_ok=True)
    cached = output / "link-receipt.json"
    if cached.exists() and json.loads(cached.read_text()).get("fixture", "bootstrap") != args.fixture:
        raise RuntimeError("cached fixture differs; use a fresh output")
    source = source_tree(output, args.fixture)
    if cached.exists():
        previous = json.loads(cached.read_text())
        if previous.get("fixture", "bootstrap") != args.fixture:
            raise RuntimeError("cached fixture differs; use a fresh output")
        if previous.get("embedded_compiler") != embedded:
            raise RuntimeError("cached compiler fixture differs; use a fresh output")
        if previous.get("rmsnorm") != rmsnorm:
            raise RuntimeError("cached RMSNorm assets differ; use a fresh output")
        if previous.get("q4") != q4:
            raise RuntimeError("cached Q4 assets differ; use a fresh output")
        if previous.get("weighted_qkv") != weighted_qkv:
            raise RuntimeError("cached weighted QKV assets differ; use a fresh output")
        units = previous["units"]
        if (len(units) != 76 or previous["title_elf_sha256"] != sha(output / "title.elf") or
                any(not unit.get("headers") or
                    any(sha(p) != h for p, h in unit["headers"].items()) or
                    sha(ROOT / unit["source"]) != unit["source_sha256"] or
                    sha(output / unit["object"]) != unit["object_sha256"] for unit in units)):
            raise RuntimeError("cached native link or its source changed; use a fresh output")
        if "package_sha256" in previous:
            if (previous.get("input_sha256") != build_inputs() or
                    previous.get("results_dir") != args.results_dir or
                    sha(output / "package" / (json.loads((ROOT / "title.json").read_text())["title_id"] + ".ffpkg")) != previous["package_sha256"] or
                    inventory(output / "package/app") != previous["package_files"]):
                raise RuntimeError("cached package differs from current inputs; use a fresh output")
            if rmsnorm is not None:
                expected_assets = {name: sha(path) for name, path in rmsnorm["assets"].items()}
                if {n: d for n, d in previous["package_files"].items() if n.startswith("rmsnorm/")} != expected_assets:
                    raise RuntimeError("cached RMSNorm package inventory differs")
                if rmsnorm_assets(args.rmsnorm_fixture, args.rmsnorm_reference) != rmsnorm:
                    raise RuntimeError("cached RMSNorm inputs changed")
            if q4 is not None:
                expected_assets = {name: sha(path) for name, path in q4["assets"].items()}
                if {n: d for n, d in previous["package_files"].items() if n.startswith("q4/")} != expected_assets:
                    raise RuntimeError("cached Q4 package inventory differs")
                if q4_assets(args.q4_fixture, args.q4_reference) != q4:
                    raise RuntimeError("cached Q4 inputs changed")
            if weighted_qkv is not None:
                expected_assets = {name: sha(path) for name, path in weighted_qkv["assets"].items()}
                if {n: d for n, d in previous["package_files"].items() if n.startswith("weighted-qkv/")} != expected_assets:
                    raise RuntimeError("cached weighted QKV package inventory differs")
                if weighted_qkv_assets(args.weighted_rmsnorm_fixture, args.q4_fixture, args.weighted_qkv_reference) != weighted_qkv:
                    raise RuntimeError("cached weighted QKV inputs changed")
                validate_weighted_qkv_audit(output, previous["weighted_qkv_audit_sha256"])
            if embedded is not None:
                audit_path = output / "embedded-native-audit.json"
                if sha(audit_path) != previous["embedded_audit_sha256"]:
                    raise RuntimeError("cached embedded audit changed")
                if sha(output / "embedded-pie.ld") != previous["embedded_linker_sha256"]:
                    raise RuntimeError("cached embedded linker changed")
                audit = json.loads(audit_path.read_text())
                if audit.get("fixture", "embedded") != args.fixture:
                    raise RuntimeError("cached compiler audit fixture differs")
                if any(sha(name) != digest for name, digest in audit["inputs"].items()):
                    raise RuntimeError("cached embedded audit inputs changed")
                if compiler_fixture_inputs(args.compiler_fixture, args.fixture) != embedded:
                    raise RuntimeError("cached compiler fixture changed")
            print(cached)
            return
    else:
        if args.fixture in ("module", "launch", "stream", "graph_gpu", "capture_gpu", "copy_gpu"):
            run(output, "build-code-object", [sys.executable, ROOT / "tools/build_canaries.py",
                "--fixture", "thread_words" if args.fixture in ("launch", "stream", "graph_gpu", "capture_gpu", "copy_gpu") else "canary",
                "--output", output / "canaries"])
        native_stubs(output)
        units = compile_units(output, source, args.fixture)
        link_title(output, units, args.fixture, embedded)
    linked_script_sha = sha(output / "embedded-pie.ld") if embedded is not None else None
    tool = native_tool(output)
    libc = clean_room_libc(output, tool)
    image, roundtrip = convert_title(output, tool, args.fixture)
    if embedded is not None:
        run(output, "audit-embedded-native", [sys.executable, ROOT / "tools/audit_public_hip_embedded_native.py",
            "--fixture", args.fixture, "--raw", output / "title.elf", "--converted", output / "eboot.elf", "--extracted", output / "extracted.elf",
            "--compiler-object", embedded["object"], "--output", output / "embedded-native-audit.json"])
    if weighted_qkv is not None:
        audit_path = output / "weighted-qkv-native-audit.json"
        if not audit_path.exists():
            run(output, "audit-weighted-qkv-native", [sys.executable, ROOT / "tools/audit_public_hip_weighted_qkv_native.py",
                "--raw", output / "title.elf", "--converted", output / "eboot.elf", "--extracted", output / "extracted.elf", "--output", audit_path])
        weighted_audit_sha = validate_weighted_qkv_audit(output)
    package, files = package_title(output, image, libc, args.results_dir,
                                   output / "canaries/gfx1013/canary.hsaco" if args.fixture == "module" else
                                   output / "canaries/gfx1013/thread_words/thread_words.hsaco" if args.fixture in ("launch", "stream", "graph_gpu", "capture_gpu", "copy_gpu") else None,
                                   extra_assets=weighted_qkv["assets"] if weighted_qkv else q4["assets"] if q4 else rmsnorm["assets"] if rmsnorm else None)
    if embedded is not None:
        audit = json.loads((output / "embedded-native-audit.json").read_text())
        if (audit.get("fixture", "embedded") != args.fixture or
                sha(output / "embedded-pie.ld") != linked_script_sha or
                any(sha(name) != digest for name, digest in audit["inputs"].items())):
            raise RuntimeError("native embedded audit or linker inputs changed before publication")
    if rmsnorm is not None and rmsnorm_assets(args.rmsnorm_fixture, args.rmsnorm_reference) != rmsnorm:
        raise RuntimeError("RMSNorm assets changed before publication")
    if q4 is not None and q4_assets(args.q4_fixture, args.q4_reference) != q4:
        raise RuntimeError("Q4 assets changed before publication")
    if weighted_qkv is not None:
        if weighted_qkv_assets(args.weighted_rmsnorm_fixture, args.q4_fixture, args.weighted_qkv_reference) != weighted_qkv:
            raise RuntimeError("weighted QKV assets changed before publication")
        validate_weighted_qkv_audit(output, weighted_audit_sha)
    if initial_inputs != build_inputs() or (embedded is not None and
            compiler_fixture_inputs(args.compiler_fixture, args.fixture) != embedded):
        raise RuntimeError("native build inputs changed before publication")
    if any(not unit.get("headers") or
           any(sha(p) != h for p, h in unit["headers"].items()) or
           sha(ROOT / unit["source"]) != unit["source_sha256"] or
           sha(output / unit["object"]) != unit["object_sha256"] for unit in units):
        raise RuntimeError("native compilation inputs or objects changed before publication")
    receipt = {"classification": "offline-public-hip-native-link", "source_commit":
               subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "fixture": args.fixture, "units": units, "object_count": len(units), "title_elf_sha256": sha(output / "title.elf"),
               "fself_sha256": sha(image), "fself_roundtrip": roundtrip,
               "package_sha256": sha(package), "package_files": files,
               "input_sha256": build_inputs(), "results_dir": args.results_dir,
               "native_executed": False}
    if rmsnorm is not None:
        receipt["rmsnorm"] = rmsnorm
    if q4 is not None:
        receipt["q4"] = q4
    if weighted_qkv is not None:
        receipt.update(weighted_qkv=weighted_qkv, weighted_qkv_audit_sha256=weighted_audit_sha)
    if embedded is not None:
        receipt.update(embedded_compiler=embedded, compiler_object_count=1, total_linked_object_count=len(units)+1,
                       embedded_linker_sha256=sha(output / "embedded-pie.ld"),
                       embedded_audit_sha256=sha(output / "embedded-native-audit.json"))
    (output / "link-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output / "link-receipt.json")


if __name__ == "__main__":
    main()
