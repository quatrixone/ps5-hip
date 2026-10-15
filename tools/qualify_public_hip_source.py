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
                 'tests/public_hip_stream_model.cpp', 'tests/public_hip_fill_model.cpp',
                 'tests/public_hip_retirement_model.cpp', 'kernels/thread_words.hip',
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

    def run(label, argv, env=None, expected_output_failure=False):
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
            if process.returncode != -6 or 'PUBLIC_LAUNCH_OUTPUT_MISMATCH index=0' not in process.stderr:
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
    for name, expected in {**inputs, **source_hashes}.items():
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
                                'async-d2d-event-ordered', 'async-d2d-bypass-stale-data'],
               'total_compilations': len(units) + 8,
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
