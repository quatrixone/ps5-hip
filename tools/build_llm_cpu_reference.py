#!/usr/bin/env python3
"""Build the locked host-only LLM reference tools; never claim PS5 execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from fetch_locked_deps import ROOT, git, read, source
from build_jobs import build_jobs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    jobs = build_jobs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'build'):
        parser.error('output must be below build/')
    spec = read('llm-inputs.json')['client']
    client = source('llama-client', spec, False)
    if not (client / 'ggml/src/ggml-cpu/CMakeLists.txt').is_file():
        raise RuntimeError('incomplete client; run make llm-deps-fetch')
    tools = {}
    for name in ('cmake', 'ninja', 'clang-18', 'clang++-18'):
        found = shutil.which(name)
        if not found:
            raise RuntimeError('required host tool missing: ' + name)
        # clang++ invocation name selects the C++ driver even when it is a symlink.
        path = Path(found).absolute()
        tools[name] = dict(path=str(path), resolved_path=str(path.resolve()), sha256=sha(path),
                           version=subprocess.check_output([path, '--version'], text=True).splitlines()[0])
    if any('18.1.8' not in tools[name]['version'] for name in ('clang-18', 'clang++-18')):
        raise RuntimeError('LLVM 18.1.8 required for the reference profile')
    flags = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER=' + tools['clang-18']['path'],
             '-DCMAKE_CXX_COMPILER=' + tools['clang++-18']['path'],
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=OFF', '-DGGML_NATIVE=OFF',
             '-DGGML_CPU=ON', '-DGGML_HIP=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
             '-DGGML_METAL=OFF', '-DGGML_SYCL=OFF', '-DGGML_RPC=OFF', '-DGGML_BLAS=OFF',
             '-DGGML_OPENMP=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=ON',
             '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_EXAMPLES=OFF',
             '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF']
    targets = ['llama-completion', 'llama-tokenize', 'llama-quantize', 'llama-perplexity',
               'test-quantize-fns', 'test-rope']
    inputs = {name: sha(ROOT / name) for name in ('llm-inputs.json',
              'tools/build_llm_cpu_reference.py', 'tools/build_jobs.py', 'tools/fetch_locked_deps.py')}
    identity = dict(source_revision=spec['revision'], source_url=spec['source_url'],
                    tools=tools, flags=flags, targets=targets, inputs=inputs)
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / 'configuration.json'
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise RuntimeError('reference inputs changed; choose a fresh LLM_CPU_BUILD_DIR')
    elif any(output.iterdir()):
        raise RuntimeError('unidentified output directory; choose a fresh LLM_CPU_BUILD_DIR')
    else:
        identity_file.write_text(json.dumps(identity, indent=2) + '\n')
    (output / 'receipt.json').unlink(missing_ok=True)
    commands = []

    def run(label, argv, timeout=300):
        with (output / (label + '.stdout')).open('w') as stdout, (output / (label + '.stderr')).open('w') as stderr:
            try:
                result = subprocess.run([str(arg) for arg in argv], cwd=ROOT,
                                        stdout=stdout, stderr=stderr, timeout=timeout)
            except subprocess.TimeoutExpired:
                commands.append(dict(label=label, argv=[str(arg) for arg in argv],
                                     timed_out=True, deadline_seconds=timeout))
                (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
                raise
        commands.append(dict(label=label, argv=[str(arg) for arg in argv], exit_code=result.returncode))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        result.check_returncode()
        print(label + ' PASS', flush=True)

    run('configure', [tools['cmake']['path'], '-S', client, '-B', output / 'cmake', *flags])
    run('build', [tools['cmake']['path'], '--build', output / 'cmake', '--parallel', str(jobs),
                  '--target', *targets], timeout=1800)
    bins = output / 'cmake/bin'
    run('completion-version', [bins / 'llama-completion', '--version'])
    run('quantize-fns', [bins / 'test-quantize-fns', '-v'])
    if '0 tests failed' not in (output / 'quantize-fns.stdout').read_text():
        raise RuntimeError('missing upstream quantization test result')
    run('rope', [bins / 'test-rope'])
    source('llama-client', spec, False)
    if any(sha(ROOT / name) != digest for name, digest in inputs.items()):
        raise RuntimeError('build inputs changed during reference qualification')
    for tool in tools.values():
        if sha(tool['path']) != tool['sha256']:
            raise RuntimeError('host tool changed during reference qualification')
    receipt = dict(classification='host-cpu-reference-tools-built', source_revision=git('-C', client, 'rev-parse', 'HEAD'),
                   identity=identity, commands=commands,
                   artifacts={str(path.relative_to(output)): sha(path) for path in sorted(bins.iterdir()) if path.is_file()},
                   model_loaded=False, native_executed=False, gpu_executed=False,
                   limits=['Host tools and upstream CPU quantization/RoPE tests only.',
                           'No model conversion, inference logits, PS5 client linkage or GPU operation qualification.'])
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(output / 'receipt.json')


if __name__ == '__main__':
    main()
