"""Optional genuine CLR + fresh llama host gate. No CPU tensor/GPU inference claim."""
import concurrent.futures
import threading
import os
from pathlib import Path
import re

from build_jobs import build_jobs
from build_public_hip import ROOT, sha
from build_llm_executed_graph import cpu_inputs, dependencies, merge, recheck
from build_llm_native_client import selection
from source_llm_model_log import MARKERS as NATIVE_MARKERS
MARKERS = tuple(line for line in NATIVE_MARKERS if not line.startswith("M0_APP MEMORY "))

BLOCKED_ENV = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'C_INCLUDE_PATH',
               'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH',
               'GCC_EXEC_PREFIX', 'LLAMA_GRAPH_REUSE_DISABLE')
SOURCES = ('tools/qualify_public_hip_llm.py', 'tests/public_hip_llm_qualification_test.py',
           'tests/llm_ps5_backend_test.cpp', 'ports/ggml/ps5_backend.cpp',
           'ports/ggml/ps5_backend.h', 'ports/ggml/model_gate.h',
           'tools/build_llm_executed_graph.py', 'tools/build_llm_native_client.py',
           'tools/source_llm_model_log.py', 'tools/source_public_hip_log.py',
           'tools/prepare_llm_native_scheduler.py', 'tools/build_jobs.py',
           'llm-inputs.json', 'evidence/offline-llm-model-conversion.json')


def environment():
    env = {k: v for k, v in os.environ.items() if k not in BLOCKED_ENV}
    env.update(ASAN_OPTIONS='protect_shadow_gap=0:halt_on_error=1:detect_leaks=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', HIP_USE_RUNTIME_UNBUNDLER='0')
    return env


def replace_admission(units, objects, replacement):
    if len(units) != len(objects) or sum(p.name == 'hip_admission_model.cpp' for p in units) != 1:
        raise ValueError('exact genuine admission object required')
    return [replacement if p.name == 'hip_admission_model.cpp' else obj for p, obj in zip(units, objects)]


def host_sources(client, folder, revision):
    ordinary, models = selection(client)
    core = [p for p in ordinary if p.parent == client/'src']
    if len(core) != 33:
        raise ValueError('llama core inventory differs')
    units = list(core)
    for i in range(0, len(models), 16):
        p = folder/('llama-unity-'+str(i//16)+'.cpp')
        p.write_text('\n'.join('#include "'+str(s)+'"' for s in models[i:i+16])+'\n')
        units.append(p)
    # This locked client's development version header is generated, never borrowed.
    cmake = (client/'CMakeLists.txt').read_text()
    for token in ('set(LLAMA_VERSION_MAJOR 0)', 'set(LLAMA_VERSION_MINOR 7)', 'set(LLAMA_VERSION_PATCH 0)'):
        if token not in cmake:
            raise ValueError('locked llama version changed')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('invalid locked llama revision')
    template = (client/'src/llama-version.h.in').read_text()
    version = template.replace('@LLAMA_VERSION@', '0.7.0-dev').replace('@LLAMA_BUILD_COMMIT@', revision[:7])
    if '@' in version:
        raise ValueError('unresolved version field')
    (folder/'llama-version.h').write_text(version)
    return units, models


def runtime_inputs(text, lib, authenticated):
    inputs, names = {}, set()
    for line in text.splitlines():
        match = re.search(r'=> (/.+) \(', line) or re.match(r'\s*(/.+) \(', line)
        if not match:
            if 'linux-vdso.so' in line or not line.strip():
                continue
            raise ValueError('unresolved or malformed runtime library')
        path = Path(match.group(1)).resolve()
        if path.name.startswith('libllama'):
            raise ValueError('shared llama bypasses actual caller wrapping')
        h = sha(path)
        if path.name.startswith('libggml'):
            names.add(path.name.split('.so')[0])
            if not path.is_relative_to(lib.resolve()) or not any(Path(k).resolve() == path and v == h for k, v in authenticated.items()):
                raise ValueError('CPU library not authenticated')
        merge(inputs, path, h)
    if names != {'libggml', 'libggml-base', 'libggml-cpu'}:
        raise ValueError('exact ggml libraries required')
    return inputs


def check_transcript(text):
    expected = [
        'LOG_POLICY_PASS controls=7 unexpected_loader_error_rejected=true',
        'M0_APP STORAGE copied=16 cleared=16 synthetic_add_rejected=true cpu_compute_calls=0 hip_launches=0 client_graph_executed=false',
        *MARKERS,
        'PRODUCTION_GATE_HOST_PASS actual_llama_process=true cpu_bypass=false storage_bounds=3 allocation_limit=67108864 model_compute=0 gpu_submits=0']
    if text.splitlines() != expected:
        raise ValueError('actual client gate transcript differs')


def check_call_chain(symbols, disassembly):
    found = {line.split()[-1] for line in symbols.splitlines() if line.split() and
             line.split()[-1].startswith('_ZN13llama_context13graph_compute')}
    if len(found) != 1:
        raise ValueError('unique real llama context caller required')
    name = next(iter(found))
    if '<'+name+'>:' not in disassembly or not re.search(
            r'\bcallq?\s+[^\n]*<__wrap_ggml_backend_sched_graph_compute_async>', disassembly):
        raise ValueError('actual context does not call mandatory wrapper')
    if re.search(r'\bcallq?\s+[^\n]*<ggml_backend_sched_graph_compute_async(?:@plt)?>', disassembly):
        raise ValueError('unwrapped scheduler call in actual caller')
    return name


def qualify(output, cpu, model, units, objects, flags, tools, run):
    import json
    folder = output/'llm'
    folder.mkdir()
    inputs = {str(ROOT/n): sha(ROOT/n) for n in SOURCES}
    for p, h in cpu_inputs(cpu, model).items():
        merge(inputs, p, h)
    for record in tools.values():
        merge(inputs, record['path'], record['sha256'])
    merge(inputs, '/usr/bin/ldd', sha('/usr/bin/ldd'))
    client = ROOT/'.deps/sources/llama-client'
    lock = json.loads((ROOT/'llm-inputs.json').read_text())['client']['revision']
    llama_units, models = host_sources(client, folder, lock)
    for p in [*llama_units, *models, folder/'llama-version.h', client/'CMakeLists.txt',
              client/'src/CMakeLists.txt', client/'src/llama-version.h.in',
              client/'ggml/src/ggml-cpu/CMakeLists.txt']:
        merge(inputs, p, sha(p))
    env = environment()
    compiler = tools['clang++-18']['path']
    def command(label, argv):
        return run('llm-'+label, argv, env)
    compiled = []
    compile_lock = threading.Lock()
    def compile_unit(label, source, compile_flags):
        dep, actual, obj = [folder/(label+suffix) for suffix in ('.headers.d', '.compiled.d', '.o')]
        command(label+'-headers', [compiler, *compile_flags, '-M', '-MT', label, '-MF', dep, source])
        headers = dependencies(dep)
        before = {p: sha(p) for p in headers}
        with compile_lock:
            for p, digest in before.items():
                merge(inputs, p, digest)
        command(label+'-compile', [compiler, *compile_flags, '-MD', '-MT', label, '-MF', actual, '-c', source, '-o', obj])
        if dependencies(actual) != headers or any(sha(p) != before[p] for p in headers):
            raise ValueError('compile dependency closure changed')
        with compile_lock:
            compiled.append({'source': str(source), 'source_sha256': sha(source), 'object': str(obj),
                             'object_sha256': sha(obj), 'headers': dict(sorted(before.items()))})
        return obj
    includes = ['-I'+str(p) for p in (ROOT/'tests', ROOT/'ports/ggml', client/'include',
                                     client/'ggml/include', client/'ggml/src')]
    test = compile_unit('backend-test', ROOT/'tests/llm_ps5_backend_test.cpp', [*flags, *includes])
    backend = compile_unit('backend', ROOT/'ports/ggml/ps5_backend.cpp', [*flags, *includes])
    llama_flags = ['-std=c++17', '-O1', '-fPIC', '-pthread', '-ffunction-sections', '-fdata-sections',
                   '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                   *['-D'+n for n in ('GGML_BACKEND_SHARED', 'GGML_SHARED', 'GGML_USE_CPU',
                                     'LLAMA_BUILD', 'LLAMA_SHARED', 'llama_EXPORTS')],
                   *includes, '-I'+str(client/'src'), '-I'+str(folder)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=build_jobs()) as pool:
        llama_objects = list(pool.map(lambda pair: compile_unit('llama-'+str(pair[0]), pair[1], llama_flags),
                                      enumerate(llama_units)))
    compiled.sort(key=lambda record: record['object'])
    selected = replace_admission(units, objects, test)+[backend, *llama_objects]
    object_hashes = {str(p): sha(p) for p in selected}
    lib, binary = cpu/'cmake/bin', folder/'model'
    command('link', [compiler, *flags, '-fuse-ld='+tools['ld.lld-18']['path'],
                    '-Wl,--gc-sections', '-Wl,-z,defs', '-Wl,--wrap=pthread_create',
                    '-Wl,--wrap=pthread_join', '-Wl,--wrap=ggml_backend_sched_graph_compute_async',
                    *selected, '-L'+str(lib), '-Wl,-rpath,'+str(lib),
                    '-lggml', '-lggml-cpu', '-lggml-base', '-o', binary])
    symbols = command('symbols', [tools['llvm-objdump-18']['path'], '--syms', binary])
    names = {line.split()[-1] for line in symbols.splitlines() if line.split() and line.split()[-1].startswith('_ZN13llama_context13graph_compute')}
    if len(names) != 1:
        raise ValueError('unique real llama context caller required')
    disassembly = command('caller', [tools['llvm-objdump-18']['path'], '-d', '--disassemble-symbols='+next(iter(names)), binary])
    caller = check_call_chain(symbols, disassembly)
    resolved = command('runtime-libraries', ['/usr/bin/ldd', binary])
    for p, h in runtime_inputs(resolved, lib, inputs).items():
        merge(inputs, p, h)
    binary_hash = sha(binary)
    for eager in (False, True):
        recheck(inputs)
        env['HIP_ENABLE_DEFERRED_LOADING'] = '0' if eager else '1'
        check_transcript(command('eager' if eager else 'deferred', [binary, model]))
    if sha(binary) != binary_hash:
        raise ValueError('executed binary changed')
    recheck(object_hashes)
    for p, h in cpu_inputs(cpu, model).items():
        merge(inputs, p, h)
    recheck(inputs)
    return dict(status='PASS', classification='source-only-genuine-clr-llama-host-gate',
                inputs=inputs, compiled=compiled, objects=object_hashes, executable_sha256=binary_hash, model_asset_sha256=inputs[str(model)],
                artifacts={str(p.relative_to(output)): sha(p) for p in [*folder.rglob('*'), *output.glob('llm-*')] if p.is_file()},
                actual_caller=caller, modes=['deferred','eager'], fresh_llama_units=len(llama_units),
                actual_client_calls_per_mode=2, second_graph_reused=True, forwarded=0, post_compute=0,
                gpu_submits=0, native_executed=False, gpu_executed=False,
                limits=['Host modeled transport; no native client or inference claim.',
                        'Only the actual audited external context scheduler call is intercepted; no universal same-TU wrapping claim.',
                        'ggml CPU libraries inherit authenticated CPU producer provenance.'])
