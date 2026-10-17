"""Focused trace-derived full-dispatch state/alias/control tests; mocked HIP only."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def build(cpu, graph, output):
    cpu, graph, output = (p.resolve() for p in (cpu, graph, output))
    if output.exists() or not output.is_relative_to(ROOT / 'results'):
        raise ValueError('fresh workspace results directory required')
    client = ROOT / '.deps/sources/llama-client'
    lib = cpu / 'cmake/bin'
    sources = ['tools/build_llm_full_dispatch_test.py', 'tests/llm_full_dispatch_test.cpp',
               'tests/llm_layer_hip_mock.h', 'ports/ggml/full_dispatch.h',
               'ports/ggml/full_graph.h', 'ports/ggml/layer_dispatch.h',
               'ports/ggml/model_layer.h', 'ports/ggml/model_prefix.h',
               'ports/ggml/prefix_dispatch.h']
    paths = [ROOT / p for p in sources] + [graph, Path('/usr/bin/clang++-18')]
    paths += [p for p in lib.glob('libggml-base.so*') if p.is_file()]
    inputs = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True)
    flags = ['-std=c++17', '-O1', '-gline-tables-only', '-Wall', '-Wextra', '-Werror',
             '-Wno-misleading-indentation', '-fsanitize=address,undefined', '-D__HIP_PLATFORM_AMD__']
    flags += ['-I' + str(p) for p in (ROOT/'ports/ggml', ROOT/'ports/hip', ROOT/'tests',
              ROOT/'.deps/sources/hip-headers/include', ROOT/'.deps/sources/clr/hipamd/include',
              client/'include', client/'ggml/include', client/'vendor')]
    flags += ['-isystem', str(client/'ggml/src')]
    scrub = ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'CPATH', 'C_INCLUDE_PATH',
             'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LSAN_OPTIONS', 'ASAN_SYMBOLIZER_PATH')
    env = {k: v for k, v in os.environ.items() if k not in scrub}
    env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    commands = []
    def run(label, argv):
        argv = list(map(str, argv))
        p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=300)
        for suffix, data in [('stdout', p.stdout), ('stderr', p.stderr)]:
            (output/(label+'.'+suffix)).write_bytes(data)
        commands.append(dict(label=label, argv=argv, exit_code=p.returncode))
        (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if p.returncode or any(s in p.stderr for s in (b'ERROR: AddressSanitizer', b'LeakSanitizer:', b'runtime error:')):
            raise RuntimeError(label+' failed; inspect retained diagnostics')
    run('compile', ['/usr/bin/clang++-18', *flags, ROOT/'tests/llm_full_dispatch_test.cpp',
                    '-L'+str(lib), '-Wl,-rpath,'+str(lib), '-lggml-base', '-o', output/'full-dispatch'])
    run('test', [output/'full-dispatch', graph])
    for p, digest in inputs.items():
        if sha(Path(p)) != digest:
            raise RuntimeError('input changed during test: '+p)
    receipt = dict(status='PASS', classification='focused-trace-derived-dispatch-mocked-HIP',
                   inputs=inputs, commands=commands, gpu_executed=False, numerical_verified=False,
                   limits=['Mocked transport performs no tensor arithmetic',
                           'Trace bytes bound directly, no recursive trace-producer authentication',
                           'Existing CPU library is not sanitizer-instrumented'])
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PASS focused full dispatcher; mocked HIP only')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-build', 'graph', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.cpu_build, args.graph, args.output)
