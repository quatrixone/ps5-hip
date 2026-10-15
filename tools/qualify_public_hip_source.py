"""Build and run genuine HIP/ROCclr host qualification from pinned sources only.

Platform providers and packet transport are modeled; this proves no GPU execution.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import threading

from build_public_hip import ROOT, DEPS, SOURCE_UNITS, source_tree, compiler_flags, sha, build_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'results'):
        parser.error('output must be under results/')
    output.mkdir(parents=True, exist_ok=False)
    dependency_check = subprocess.run([sys.executable, ROOT / 'tools/fetch_locked_deps.py'],
                                      cwd=ROOT, capture_output=True, text=True, timeout=300)
    (output / 'dependency-check.log').write_text(dependency_check.stdout + dependency_check.stderr)
    dependency_check.check_returncode()
    toolchain = {}
    for name in ('clang++-18', 'ld.lld-18', 'llvm-readobj-18', 'llvm-objdump-18'):
        path = Path(shutil.which(name))
        toolchain[name] = {'path': str(path), 'resolved_path': str(path.resolve()), 'sha256': sha(path),
                          'version': subprocess.check_output([path, '--version'], text=True)}
    inputs = build_inputs()
    for name in ('tools/qualify_public_hip_source.py', 'tests/hip_admission_model.cpp',
                 'tests/clr_adapter_model.cpp', 'tests/public_hip_launch_model.cpp',
                 'tests/public_hip_stream_model.cpp', 'tests/public_hip_stream_gpu_model.cpp', 'tests/public_hip_fill_model.cpp',
                 'tests/public_hip_retirement_model.cpp', 'tests/public_hip_rect_copy_model.cpp',
                 'tests/public_hip_graph_model.cpp', 'tests/public_hip_graph_gpu_model.cpp', 'tests/public_hip_copy_graph_model.cpp', 'tests/public_hip_copy_gpu_model.cpp', 'kernels/thread_words.hip',
                 'tools/build_canaries.py', 'kernels/canary.hip'):
        inputs[name] = sha(ROOT / name)
    source = source_tree(output, 'copy')
    flags = [str(p) for p in compiler_flags(source)[1:]]
    flags = [p for p in flags if p not in ('-DPS5HIP_CLR_GPU_TITLE=1',
                                          '-DPS5HIP_SHELL_EXIT_PROBE=1', '-DPS5HIP_WITH_ELEVATION=1')]
    flags += ['--driver-mode=g++', '-DPS5HIP_CLR_MODEL', '-I' + str(DEPS / 'sources/sdk/platform/include'),
              '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    units = [source / name for name in SOURCE_UNITS]
    units += [ROOT / name for name in ('ports/clr/ps5-clr-tls.cpp', 'ports/clr/ps5-device.cpp',
                                      'tests/hip_admission_model.cpp')]
    units += [ROOT / 'src' / (name + '.cpp') for name in (
        'code_object', 'fat_binary', 'host_launch', 'launch', 'native_memory',
        'native_data', 'native_launch', 'submission', 'hip_launch_profile')]
    units += [p for p in sorted((source / 'hipamd/src').glob('*.cpp'))
              if p.stem not in ('fixme', 'hip_runtime')]
    if len(units) != 69:
        raise RuntimeError('unexpected public HIP source closure')
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in units}
    header_hashes = {}
    header_lock = threading.Lock()

    def run(label, argv, env=None, expected_output_failure=False,
            failure_marker='PUBLIC_LAUNCH_OUTPUT_MISMATCH index=0'):
        command = [str(p) for p in argv]
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True,
                                     text=True, timeout=300, env=env)
        except subprocess.TimeoutExpired as error:
            for suffix, value in (('stdout', error.stdout), ('stderr', error.stderr)):
                (output / (label + '.' + suffix)).write_bytes(
                    value.encode() if isinstance(value, str) else value or b'')
            (output / (label + '.json')).write_text(json.dumps(
                {'argv': command, 'timed_out': True, 'deadline_seconds': 300}, indent=2) + '\n')
            raise
        (output / (label + '.stdout')).write_text(process.stdout)
        (output / (label + '.stderr')).write_text(process.stderr)
        record = {'argv': [str(p) for p in argv], 'exit_code': process.returncode}
        if env:
            record['sanitizer_environment'] = {key: env.get(key) for key in ('ASAN_OPTIONS', 'UBSAN_OPTIONS')}
            record['runtime_environment'] = {'HIP_USE_RUNTIME_UNBUNDLER': env['HIP_USE_RUNTIME_UNBUNDLER']}
        (output / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        if expected_output_failure:
            if process.returncode != -6 or failure_marker not in process.stderr:
                raise RuntimeError('negative control did not fail the output oracle: ' + label)
        elif process.returncode:
            raise RuntimeError(label + ': ' + process.stderr[-2500:])
        return process.stdout

    def compile_one(pair):
        index, path = pair
        extra = (['-include', 'hip/hip_runtime.h', '-include', 'hip/hip_deprecated.h',
                  '-include', 'hip/amd_detail/amd_hip_gl_interop.h']
                 if path.stem in ('hip_api_trace', 'hip_table_interface', 'hip_table_interface_c') else [])
        obj = output / (str(index) + '.o')
        compiler = toolchain['clang++-18']['path']
        dep = output / (str(index) + '.d')
        run('headers-' + str(index), [compiler, *flags, *extra, '-M', path, '-MF', dep])
        paths = [Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])]
        before = {str(p): sha(p) for p in paths}
        with header_lock:
            for name, digest in before.items():
                if name in header_hashes and header_hashes[name] != digest:
                    raise RuntimeError('header changed between translation units: ' + name)
                header_hashes[name] = digest
        run('compile-' + str(index), [compiler, *flags, *extra, '-c', path, '-o', obj])
        if any(sha(p) != expected for p, expected in before.items()):
            raise RuntimeError('header changed during compilation: ' + str(path))
        return obj

    run('code-object', [sys.executable, ROOT / 'tools/build_canaries.py', '--output', output / 'canaries'])
    code_object = output / 'canaries/gfx1013/canary.hsaco'
    code_hash = sha(code_object)
    wrong_target = output / 'canaries/gfx1030/canary.hsaco'
    wrong_target_hash = sha(wrong_target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        objects = list(pool.map(compile_one, enumerate(units)))
    run('link', [toolchain['clang++-18']['path'], *flags, '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
                 '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *objects, '-o', output / 'model'])
    modes = ['normal', 'default', 'tls', 'init-fail', 'create-fail', 'join-fail',
             'self-destroy', 'reset', 'reset-join-fail', 'memory', 'module', 'module-unbundler']
    for mode in modes:
        retained = mode in ('join-fail', 'self-destroy', 'reset-join-fail')
        env = dict(os.environ, ASAN_OPTIONS='protect_shadow_gap=0:halt_on_error=1:detect_leaks=' +
                   ('0' if retained else '1'), UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        env['HIP_USE_RUNTIME_UNBUNDLER'] = '1' if mode == 'module-unbundler' else '0'
        text = run(mode, [output / 'model', mode, *([code_object, wrong_target] if mode.startswith('module') else [])], env)
        lines = text.splitlines()
        passed = 'HIP_PUBLIC_PASS mode=' + mode + ' gpu_execution=false'
        if mode not in ('init-fail', 'memory', 'create-fail', 'module', 'module-unbundler'):
            passed += ' quarantine=' + str(int(retained))
        if lines.count(passed) != 1 or sum(line.startswith('HIP_PUBLIC_PASS ') for line in lines) != 1:
            raise RuntimeError('missing qualification marker: ' + mode)
        if mode != 'init-fail':
            queues = 1 if mode in ('create-fail', 'memory', 'module', 'module-unbundler') else 4 if mode in ('default', 'tls') else 3
            expected = f'HIP_PUBLIC_FINAL queues={queues} retired={2 if mode == "self-destroy" else queues} gate_destroyed={int(not retained)} retained={int(retained)}'
            if lines.count(expected) != 1 or sum(line.startswith('HIP_PUBLIC_FINAL ') for line in lines) != 1:
                raise RuntimeError('missing exact lifetime result: ' + mode)
        elif any(line.startswith('HIP_PUBLIC_FINAL ') for line in lines):
            raise RuntimeError('unexpected finalizer after initialization failure')
        print(mode + ' PASS', flush=True)
    # Reuse the freshly built runtime, replacing only the model's main object.
    launch_source = ROOT / 'tests/public_hip_launch_model.cpp'
    source_hashes[str(launch_source.relative_to(ROOT))] = sha(launch_source)
    launch_object = compile_one((len(units), launch_source))
    launch_objects = [launch_object if unit.name == 'hip_admission_model.cpp' else obj
                      for unit, obj in zip(units, objects)]
    run('launch-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *launch_objects, '-o', output / 'launch-model'])
    run('launch-code-object', [sys.executable, ROOT / 'tools/build_canaries.py',
        '--fixture', 'thread_words', '--output', output / 'launch-canaries'])
    launch_code = output / 'launch-canaries/gfx1013/thread_words/thread_words.hsaco'
    launch_code_hash = sha(launch_code)
    env = dict(os.environ, ASAN_OPTIONS='protect_shadow_gap=0:halt_on_error=1:detect_leaks=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', HIP_USE_RUNTIME_UNBUNDLER='0')
    result = run('launch-normal', [output / 'launch-model', launch_code, 'normal'], env)
    if result.splitlines() != [
        'PUBLIC_LAUNCH_PASS values=73 guards=8 captured_args=true gpu_execution=false',
        'PUBLIC_LAUNCH_FINAL ownership=clean gpu_execution=false']:
        raise RuntimeError('public launch output/lifetime contract differs')
    result = run('launch-events', [output / 'launch-model', launch_code, 'events'], env)
    if result.splitlines() != [
        'PUBLIC_EVENT_PASS pending_then_complete=true values=73 guards=8 captured_args=true gpu_execution=false',
        'PUBLIC_LAUNCH_FINAL ownership=clean gpu_execution=false']:
        raise RuntimeError('public event completion/ownership contract differs')
    for negative in ('noop', 'corrupt'):
        run('launch-' + negative, [output / 'launch-model', launch_code, negative], env,
            expected_output_failure=True)
    if sha(launch_code) != launch_code_hash:
        raise RuntimeError('launch fixture changed during qualification')
    print('public launch and both output negative controls PASS', flush=True)
    stream_gpu_source = ROOT / 'tests/public_hip_stream_gpu_model.cpp'
    source_hashes[str(stream_gpu_source.relative_to(ROOT))] = sha(stream_gpu_source)
    stream_gpu_object = compile_one((len(units) + 10, stream_gpu_source))
    stream_gpu_objects = [stream_gpu_object if unit.name == 'hip_admission_model.cpp' else obj
                          for unit, obj in zip(units, objects)]
    run('stream-gpu-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *stream_gpu_objects, '-o', output / 'stream-gpu-model'])
    result = run('stream-gpu-normal', [output / 'stream-gpu-model', launch_code, 'normal'], env)
    if result.splitlines() != [
        'PUBLIC_STREAM_GPU_PASS rounds=2 launches=2 values=146 guards=16 source_preserved=true event_reused=true gpu_execution=false',
        'PUBLIC_STREAM_GPU_FINAL queues=3 retired=3 allocations=2 frees=2 ownership=clean gpu_execution=false']:
        raise RuntimeError('public stream GPU ordering/output/lifetime contract differs')
    for negative in ('bypass', 'noop', 'corrupt'):
        run('stream-gpu-' + negative, [output / 'stream-gpu-model', launch_code, negative], env,
            expected_output_failure=True, failure_marker='PUBLIC_STREAM_GPU_OUTPUT_MISMATCH round=0 index=0')
    print('public stream GPU ordering and three negative controls PASS', flush=True)
    graph_gpu_source = ROOT / 'tests/public_hip_graph_gpu_model.cpp'
    source_hashes[str(graph_gpu_source.relative_to(ROOT))] = sha(graph_gpu_source)
    graph_gpu_object = compile_one((len(units) + 11, graph_gpu_source))
    graph_gpu_objects = [graph_gpu_object if unit.name == 'hip_admission_model.cpp' else obj
                         for unit, obj in zip(units, objects)]
    run('graph-gpu-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *graph_gpu_objects, '-o', output / 'graph-gpu-model'])
    result = run('graph-gpu-normal', [output / 'graph-gpu-model', launch_code, 'normal'], env)
    if result.splitlines() != [
        'PUBLIC_GRAPH_GPU_PASS rounds=2 values=146 guards=16 owned_args=true updated=true source_destroyed=true gpu_execution=false',
        'PUBLIC_GRAPH_GPU_FINAL queues_retired=true allocations=1 frees=1 ownership=clean gpu_execution=false']:
        raise RuntimeError('public GPU graph argument/replay/lifetime contract differs')
    for negative in ('noop', 'corrupt'):
        run('graph-gpu-' + negative, [output / 'graph-gpu-model', launch_code, negative], env,
            expected_output_failure=True, failure_marker='PUBLIC_GRAPH_GPU_OUTPUT_MISMATCH round=0 index=0')
    result = run('graph-gpu-native-oracle', [output / 'graph-gpu-model', launch_code, 'native'], env)
    if result.splitlines() != [
        'PUBLIC_GRAPH_GPU_NATIVE_ORACLE_PASS rounds=2 values=146 guards=16 disabled_replays=2 callbacks=4 gpu_execution=false',
        'PUBLIC_GRAPH_GPU_FINAL queues_retired=true allocations=1 frees=1 ownership=clean gpu_execution=false']:
        raise RuntimeError('shared native GPU graph oracle differs')
    for mode in ('capture', 'capture-recover'):
        result = run('kernel-' + mode, [output / 'graph-gpu-model', launch_code, mode], env)
        expected = ([
            'PUBLIC_KERNEL_CAPTURE_RECOVERY_PASS invalidations=36 modes=3 caller_graphs=18 callbacks=0 gpu_execution=false']
            if mode == 'capture-recover' else []) + [
            'PUBLIC_KERNEL_CAPTURE_PASS rounds=2 values=146 guards=16 disabled_replays=2 callbacks=4 gpu_execution=false',
            'PUBLIC_GRAPH_GPU_FINAL queues_retired=true allocations=1 frees=1 ownership=clean gpu_execution=false']
        if result.splitlines() != expected:
            raise RuntimeError('public kernel capture/recovery oracle differs')
    print('public kernel capture and36 recovery cases PASS', flush=True)
    print('public GPU graph replay and two negative controls PASS', flush=True)
    stream_source = ROOT / 'tests/public_hip_stream_model.cpp'
    source_hashes[str(stream_source.relative_to(ROOT))] = sha(stream_source)
    stream_object = compile_one((len(units) + 1, stream_source))
    stream_objects = [stream_object if unit.name == 'hip_admission_model.cpp' else obj
                      for unit, obj in zip(units, objects)]
    run('stream-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *stream_objects, '-o', output / 'stream-model'])
    for mode, marker in (
        ('normal', 'PUBLIC_STREAM_PASS event_ordered=true gpu_execution=false'),
        ('bypass', 'PUBLIC_STREAM_BYPASS early_consumer_detected=true gpu_execution=false')):
        result = run('stream-' + mode, [output / 'stream-model', mode], env)
        if result.splitlines() != [marker,
            'PUBLIC_STREAM_FINAL queues=2 retired=2 ownership=clean gpu_execution=false']:
            raise RuntimeError('public stream ordering/ownership contract differs: ' + mode)
        print('public stream ' + mode + ' PASS', flush=True)
    for mode, marker in (
        ('copy', 'PUBLIC_STREAM_COPY_PASS exact_bytes=128 event_ordered=true gpu_execution=false'),
        ('copy-bypass', 'PUBLIC_STREAM_COPY_BYPASS stale_bytes=128 gpu_execution=false')):
        result = run('stream-' + mode, [output / 'stream-model', mode], env)
        if result.splitlines() != [marker,
            'PUBLIC_STREAM_COPY_FINAL queues=3 retired=3 allocations=3 frees=3 ownership=clean gpu_execution=false']:
            raise RuntimeError('public async copy ordering/ownership differs: ' + mode)
        print('public stream ' + mode + ' PASS', flush=True)
    for mode, direction, nonblocking in (
        ('default-blocking', 'explicit-to-default', False),
        ('default-nonblocking', 'explicit-to-default', True),
        ('default-reverse-blocking', 'default-to-explicit', False),
        ('default-reverse-nonblocking', 'default-to-explicit', True)):
        result = run('stream-' + mode, [output / 'stream-model', mode], env)
        marker = ('PUBLIC_DEFAULT_STREAM_PASS direction=' + direction +
                  ' nonblocking=' + str(nonblocking).lower() +
                  ' implicit_order=' + str(not nonblocking).lower() + ' gpu_execution=false')
        if result.splitlines() != [marker,
            'PUBLIC_STREAM_FINAL queues=2 retired=2 ownership=clean gpu_execution=false']:
            raise RuntimeError('public default stream semantics differ: ' + mode)
        print('public stream ' + mode + ' PASS', flush=True)
    copy_gpu_source = ROOT / 'tests/public_hip_copy_gpu_model.cpp'
    source_hashes[str(copy_gpu_source.relative_to(ROOT))] = sha(copy_gpu_source)
    copy_gpu_object = compile_one((len(units) + 13, copy_gpu_source))
    copy_gpu_objects = [copy_gpu_object if unit.name == 'hip_admission_model.cpp' else obj
                       for unit, obj in zip(units, objects)]
    run('copy-gpu-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *copy_gpu_objects, '-o', output / 'copy-gpu-model'])
    for mode in ('explicit', 'capture'):
        result = run('copy-gpu-' + mode, [output / 'copy-gpu-model', launch_code, mode], env)
        if result.splitlines() != [
            'PUBLIC_COPY_GPU_PASS rounds=2 launches=4 values=292 guards=32 eager_equal=true disabled_replays=6 callbacks=8 gpu_execution=false',
            'PUBLIC_COPY_GPU_FINAL queues_retired=true allocations=2 frees=2 ownership=clean gpu_execution=false']:
            raise RuntimeError('shared copy/kernel graph oracle differs')
    for negative in ('noop', 'corrupt'):
        run('copy-gpu-' + negative, [output / 'copy-gpu-model', launch_code, negative], env,
            expected_output_failure=True, failure_marker='PUBLIC_COPY_GPU_OUTPUT_MISMATCH round=0 index=0')
    for negative, index in (('copy-h2d', 73), ('copy-d2d', 0), ('copy-d2h', 0)):
        run('copy-gpu-' + negative, [output / 'copy-gpu-model', launch_code, negative], env,
            expected_output_failure=True, failure_marker='PUBLIC_COPY_GPU_OUTPUT_MISMATCH round=0 index=' + str(index))
    print('public copy/kernel graph and five negative controls PASS', flush=True)
    copy_graph_source = ROOT / 'tests/public_hip_copy_graph_model.cpp'
    source_hashes[str(copy_graph_source.relative_to(ROOT))] = sha(copy_graph_source)
    copy_graph_object = compile_one((len(units) + 12, copy_graph_source))
    copy_graph_objects = [copy_graph_object if unit.name == 'hip_admission_model.cpp' else obj
                          for unit, obj in zip(units, objects)]
    run('copy-graph-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *copy_graph_objects, '-o', output / 'copy-graph-model'])
    for mode in ('explicit', 'capture', 'parallel'):
        result = run('copy-graph-' + mode, [output / 'copy-graph-model', mode], env)
        expected = ['PUBLIC_COPY_CAPTURE_ERRORS_PASS cases=27 modes=3 aliases=4 partial_graph_rejected=true stream_reused=true gpu_execution=false', 'PUBLIC_COPY_GRAPH_UPDATE_PASS source_isolated=true offsets=true guards=true rejected_update_preserved=true gpu_execution=false'] + (['PUBLIC_COPY_GRAPH_JOIN_PASS rounds=2 dependency_preserved=true gpu_execution=false'] if mode == 'parallel' else []) + [
            'PUBLIC_COPY_GRAPH_PASS rounds=4 bytes=512 disabled_replays=6 source_destroyed=true gpu_execution=false',
            'PUBLIC_COPY_GRAPH_FINAL queues_retired=true ownership=clean gpu_execution=false']
        if result.splitlines() != expected:
            raise RuntimeError('public copy graph replay/retirement differs')
    print('public explicit/captured copy graph PASS', flush=True)
    fill_source = ROOT / 'tests/public_hip_fill_model.cpp'
    source_hashes[str(fill_source.relative_to(ROOT))] = sha(fill_source)
    fill_object = compile_one((len(units) + 2, fill_source))
    fill_objects = [fill_object if unit.name == 'hip_admission_model.cpp' else obj
                    for unit, obj in zip(units, objects)]
    run('fill-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *fill_objects, '-o', output / 'fill-model'])
    result = run('fill-normal', [output / 'fill-model'], env)
    if result.splitlines() != [
        'PUBLIC_FILL_PASS bytes=128 patterns=1,2,4 guards=true gpu_execution=false',
        'PUBLIC_FILL_FINAL queues=2 retired=2 ownership=clean gpu_execution=false']:
        raise RuntimeError('public fill byte/ownership contract differs')
    print('public fill PASS', flush=True)
    result = run('fill-pitched', [output / 'fill-model', 'pitched'], env)
    if result.splitlines() != [
        'PUBLIC_PITCHED_FILL_PASS compared_bytes=2048 rounds=2 padding=true gpu_execution=false',
        'PUBLIC_FILL_FINAL queues=2 retired=2 ownership=clean gpu_execution=false']:
        raise RuntimeError('public pitched fill padding/ownership contract differs')
    print('public pitched fill PASS', flush=True)

    # Inject allocation failures into independent source copies, not production hooks.
    fill_failure_models = {}
    memory_source = source / 'hipamd/src/hip_memory.cpp'
    for index, (stage, anchor) in enumerate((
        ('reserve', '      commands.reserve(commands.size() + 1);'),
        ('command', '      command = new amd::FillMemoryCommand(*stream, CL_COMMAND_FILL_BUFFER,'))):
        text = memory_source.read_text()
        if text.count(anchor) != 1:
            raise RuntimeError('fill allocation injection anchor differs: ' + stage)
        injected = output / ('hip_memory_fill_' + stage + '.cpp')
        injected.write_text(text.replace(anchor, '      throw std::bad_alloc();\n' + anchor))
        source_hashes[str(injected.relative_to(ROOT))] = sha(injected)
        injected_object = compile_one((len(units) + 3 + index, injected))
        fault_objects = [injected_object if unit == memory_source else obj
                         for unit, obj in zip(units, fill_objects)]
        binary = output / ('fill-oom-' + stage)
        run('fill-oom-' + stage + '-link', [toolchain['clang++-18']['path'], *flags,
            '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
            '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *fault_objects, '-o', binary])
        result = run('fill-oom-' + stage, [binary, 'pitched-oom'], env)
        if result.splitlines() != [
            'PUBLIC_PITCHED_FILL_OOM unchanged=true ownership=clean gpu_execution=false',
            'PUBLIC_FILL_FINAL queues=2 retired=2 ownership=clean gpu_execution=false']:
            raise RuntimeError('public fill allocation failure contract differs: ' + stage)
        fill_failure_models[stage] = sha(binary)
        print('public pitched fill ' + stage + ' OOM PASS', flush=True)
    # Hold a completed fill before worker retirement. Only generated copies contain hooks.
    retirement_source = ROOT / 'tests/public_hip_retirement_model.cpp'
    source_hashes[str(retirement_source.relative_to(ROOT))] = sha(retirement_source)
    retirement_object = compile_one((len(units) + 5, retirement_source))
    adapter = ROOT / 'ports/clr/ps5-device.cpp'
    adapter_text = adapter.read_text()
    completion = '        c.setStatus(status==CL_SUCCESS?CL_COMPLETE:status);'
    barrier = 'bool isHandlerPending()const override{return true;}'
    if adapter_text.count(completion) != 1 or adapter_text.count(barrier) != 1:
        raise RuntimeError('retirement injection anchors differ')
    hooked = 'extern "C" void qualification_hold_after_fill();\n' + adapter_text.replace(
        completion, completion + '\n        if(c.type()==CL_COMMAND_FILL_BUFFER)qualification_hold_after_fill();')
    retirement_models = {}
    for index, mode in enumerate(('new', 'old')):
        injected = output / ('retirement-' + mode + '.cpp')
        injected.write_text(hooked if mode == 'new' else hooked.replace(
            barrier, 'bool isHandlerPending()const override{return false;}'))
        source_hashes[str(injected.relative_to(ROOT))] = sha(injected)
        adapter_object = compile_one((len(units) + 6 + index, injected))
        selected = [adapter_object if unit == adapter else
                    retirement_object if unit.name == 'hip_admission_model.cpp' else obj
                    for unit, obj in zip(units, objects)]
        binary = output / ('retirement-' + mode)
        run('retirement-' + mode + '-link', [toolchain['clang++-18']['path'], *flags,
            '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
            '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *selected, '-o', binary])
        result = run('retirement-' + mode, [binary, mode], env)
        marker = ('RETIREMENT_NEW held_until_retirement=true free_once=true' if mode == 'new'
                  else 'RETIREMENT_OLD early_sync_observed=true')
        if result.splitlines() != [marker, 'RETIREMENT_COUNTS queues=1 retired=1',
                'RETIREMENT_FINAL queues=1 retired=1 ownership=clean gpu_execution=false']:
            raise RuntimeError('retirement ordering/ownership differs: ' + mode)
        retirement_models[mode] = sha(binary)
        print('public retirement ' + mode + ' PASS', flush=True)
    rect_source = ROOT / 'tests/public_hip_rect_copy_model.cpp'
    source_hashes[str(rect_source.relative_to(ROOT))] = sha(rect_source)
    rect_object = compile_one((len(units) + 8, rect_source))
    rect_objects = [rect_object if unit.name == 'hip_admission_model.cpp' else obj
                    for unit, obj in zip(units, objects)]
    run('rect-copy-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *rect_objects, '-o', output / 'rect-copy-model'])
    result = run('rect-copy', [output / 'rect-copy-model'], env)
    if result.splitlines() != [
        'PUBLIC_RECT_COPY_PASS compared_bytes=4096 rounds=2 padding=true rejected=31 synchronous_failure=true gpu_execution=false',
        'PUBLIC_STREAM_COPY_FINAL queues=3 retired=3 allocations=3 frees=3 ownership=clean gpu_execution=false']:
        raise RuntimeError('rectangular copy data/ownership contract differs')
    print('public rectangular copy PASS', flush=True)
    graph_source = ROOT / 'tests/public_hip_graph_model.cpp'
    source_hashes[str(graph_source.relative_to(ROOT))] = sha(graph_source)
    graph_object = compile_one((len(units) + 9, graph_source))
    graph_objects = [graph_object if unit.name == 'hip_admission_model.cpp' else obj
                     for unit, obj in zip(units, objects)]
    run('graph-link', [toolchain['clang++-18']['path'], *flags,
        '-fuse-ld=' + toolchain['ld.lld-18']['path'], '-Wl,--gc-sections', '-Wl,-z,defs',
        '-Wl,--wrap=pthread_create', '-Wl,--wrap=pthread_join', *graph_objects, '-o', output / 'graph-model'])
    for mode in ('explicit', 'capture'):
        result = run('graph-' + mode, [output / 'graph-model', mode], env)
        if result.splitlines() != [
            'PUBLIC_GRAPH_PASS mode=' + mode + ' replays=16 callbacks=32 parameter_update=true source_destroyed=true gpu_execution=false',
            'PUBLIC_GRAPH_FINAL queues=3 retired=3 allocations=0 frees=0 ownership=clean gpu_execution=false']:
            raise RuntimeError('host graph replay/update/ownership differs: ' + mode)
        print('public host graph ' + mode + ' PASS', flush=True)
    result = run('graph-recover', [output / 'graph-model', 'recover'], env)
    if result.splitlines() != [
        'PUBLIC_CAPTURE_RECOVERY_PASS invalidations=6 modes=3 external_graphs=3 callbacks=0 stream_reusable=true',
        'PUBLIC_GRAPH_PASS mode=capture replays=16 callbacks=32 parameter_update=true source_destroyed=true gpu_execution=false',
        'PUBLIC_GRAPH_FINAL queues=3 retired=3 allocations=0 frees=0 ownership=clean gpu_execution=false']:
        raise RuntimeError('capture invalidation cleanup/reuse contract differs')
    print('public capture invalidation recovery PASS', flush=True)
    result = run('graph-session', [output / 'graph-model', 'session'], env)
    if result.splitlines() != [
        'PUBLIC_CAPTURE_SESSION_PASS invalidations=24 streams=4 modes=3 external_graphs=12 callbacks=1 events_reusable=true',
        'PUBLIC_GRAPH_FINAL queues=4 retired=4 allocations=0 frees=0 ownership=clean gpu_execution=false']:
        raise RuntimeError('cross-stream capture invalidation/recovery differs')
    print('public capture session recovery PASS', flush=True)
    result = run('graph-rejoin', [output / 'graph-model', 'rejoin'], env)
    if result.splitlines() != [
        'PUBLIC_CAPTURE_REJOIN_PASS invalidations=24 streams=4 modes=3 external_graphs=12 merges_rejected=3 replays=8 replay_nodes=5 ordinary_callbacks=1 events_reused=4',
        'PUBLIC_GRAPH_FINAL queues=7 retired=7 allocations=0 frees=0 ownership=clean gpu_execution=false']:
        raise RuntimeError('capture rejoin invalidation/recovery differs')
    print('public capture rejoin recovery PASS', flush=True)
    result = run('graph-combined', [output / 'graph-model', 'combined'], env)
    if result.splitlines() != [
        'PUBLIC_GRAPH_COMBINED_PASS graphs=2 replays=32 callbacks=64 invalidations=6 gpu_execution=false',
        'PUBLIC_GRAPH_FINAL queues=5 retired=5 allocations=0 frees=0 ownership=clean gpu_execution=false']:
        raise RuntimeError('combined native graph oracle/ownership differs')
    print('public combined graph fixture PASS', flush=True)
    run('graph-explicit-noop', [output / 'graph-model', 'explicit-noop'], env,
        expected_output_failure=True, failure_marker='PUBLIC_GRAPH_OUTPUT_MISMATCH round=0')
    print('public host graph no-op output control PASS', flush=True)
    for collection in (inputs, source_hashes):
        for name, expected in collection.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('input changed during qualification: ' + name)
    for name, expected in header_hashes.items():
        if sha(name) != expected:
            raise RuntimeError('header changed during qualification: ' + name)
    for record in toolchain.values():
        if sha(record['path']) != record['sha256']:
            raise RuntimeError('toolchain changed during qualification')
    if sha(code_object) != code_hash or sha(wrong_target) != wrong_target_hash:
        raise RuntimeError('module fixture changed during qualification')
    receipt = {'classification': 'source-only-genuine-public-hip-model', 'source_units': len(units),
               'inputs': inputs, 'prepared_sources': source_hashes, 'modes': modes,
               'included_files': header_hashes, 'toolchain': toolchain,
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'module_fixture_sha256': code_hash,
               'launch_model_sha256': sha(output / 'launch-model'),
               'stream_model_sha256': sha(output / 'stream-model'),
               'stream_cases': ['event-ordered', 'bypass-early-consumer-detected',
                                'async-d2d-event-ordered', 'async-d2d-bypass-stale-data',
                                'default-blocking', 'default-nonblocking',
                                'default-reverse-blocking', 'default-reverse-nonblocking'],
               'total_compilations': len(units) + 14,
               'copy_gpu_model_sha256': sha(output / 'copy-gpu-model'),
               'copy_gpu_cases': ['explicit', 'capture', 'eager-equality', 'six-disabled-directions', 'update-source-destroy', 'noop-rejected', 'corrupt-rejected', 'h2d-omitted-rejected', 'd2d-omitted-rejected', 'd2h-omitted-rejected'],
               'copy_graph_model_sha256': sha(output / 'copy-graph-model'),
               'copy_graph_cases': ['explicit', 'capture', 'changing-input', 'disabled-reenabled', 'source-destroyed', 'allocation-retirement', 'copy-capture-invalidation-27-cases', 'source-update-isolation', 'exec-update-offsets-guards', 'rejected-update-preserves-parameters'],
               'graph_gpu_model_sha256': sha(output / 'graph-gpu-model'),
               'graph_gpu_cases': ['module-kernel-capture', '36-capture-failure-recoveries', 'shared-native-oracle', 'exec-update-handle-type-rejections', 'owned-arguments-update-source-destroy', 'argument-allocation-failures', 'single-parallel-child-command-failures', 'disabled-reenabled', 'noop-output-rejected', 'corrupt-output-rejected'],
               'stream_gpu_model_sha256': sha(output / 'stream-gpu-model'),
               'stream_gpu_cases': ['event-ordered-two-rounds', 'bypass-stale-output-rejected', 'noop-output-rejected', 'corrupt-output-rejected'],
               'graph_model_sha256': sha(output / 'graph-model'),
               'graph_cases': ['explicit-host-replay-update-source-destroy',
                               'captured-host-replay-update-source-destroy', 'noop-output-rejected',
                               'capture-invalidation-cleanup-and-reuse', 'cross-stream-capture-session-recovery', 'capture-rejoin-recovery', 'combined-native-fixture-oracle'],
               'rect_copy_model_sha256': sha(output / 'rect-copy-model'),
               'retirement_models': retirement_models,
               'retirement_observation_ms': 500,
               'fill_allocation_failure_models': fill_failure_models,
               'fill_model_sha256': sha(output / 'fill-model'),
               'fill_cases': ['patterns-offsets-guards-admission', 'pitched-2d-3d-padding-admission'],
               'launch_fixture_sha256': launch_code_hash,
               'launch_cases': ['normal', 'events-pending-then-complete', 'noop-output-rejected', 'corrupt-output-rejected'],
               'wrong_target_fixture_sha256': wrong_target_hash,
               'leak_detection_disabled_modes': ['join-fail', 'self-destroy', 'reset-join-fail'],
               'gpu_executed': False, 'native_executed': False,
               'sanitizer': 'address,undefined', 'model_sha256': sha(output / 'model')}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(output / 'receipt.json')


if __name__ == '__main__':
    main()
