#!/usr/bin/env python3
"""Build the locked host-only LLM reference tools; never claim PS5 execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from fetch_locked_deps import ROOT, git, read, source
from build_jobs import build_jobs
import prepare_llm_native_loader as loader
from native_ccache import shared_libraries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compiler_cache(env):
    mode = env.get('PS5_HIP_CCACHE', 'auto')
    if mode not in ('auto', '0', '1'):
        raise ValueError('PS5_HIP_CCACHE must be auto, 0 or 1')
    tool = shutil.which('ccache', path='/usr/bin:/bin') if mode != '0' else None
    if tool is None:
        if mode == '1':
            raise ValueError('PS5_HIP_CCACHE=1 requires ccache')
        return {'enabled': False, 'requested': mode}, []
    tool = str(Path(tool).resolve())
    directory = str(Path(env.get('PS5_HIP_CCACHE_DIR', str(Path.home() / '.cache/ps5-hip/ccache'))).expanduser().absolute())
    options = ['compiler_type=clang', 'compiler_check=content', 'direct_mode=false',
               'sloppiness=', 'remote_storage=', 'namespace=ps5-hip-cpu-v1',
               'max_size=1GiB', 'hash_dir=true', 'hard_link=false', 'cache_dir=' + directory]
    prefix = [tool, *options]
    if any(';' in arg or '\n' in arg or '\r' in arg for arg in prefix):
        raise ValueError('compiler cache path cannot contain CMake list delimiters or newlines')
    record = dict(enabled=True, requested=mode, tool=tool, tool_sha256=sha(tool),
                  directory=directory, options=options,
                  inputs={str(p): sha(p) for p in shared_libraries(tool)})
    return record, prefix


def cache_flags(prefix):
    # Explicit empty launchers also prevent inherited CMake environment defaults.
    return ['-DCMAKE_' + language + '_COMPILER_LAUNCHER=' + ';'.join(prefix)
            for language in ('C', 'CXX')]


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
    cache_record, cache_prefix = compiler_cache(os.environ)
    build_env = {k: v for k, v in os.environ.items() if not k.startswith('CCACHE_')}
    build_env['CCACHE_CONFIGPATH'] = '/dev/null'
    generated_loader = output / 'llama-model-loader-user.cpp'
    cmake_hook = ROOT / 'tools/llm_user_loader.cmake'
    flags = ['-DCMAKE_PROJECT_INCLUDE=' + str(cmake_hook),
             '-DPS5HIP_USER_LOADER=' + str(generated_loader), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER=' + tools['clang-18']['path'],
             '-DCMAKE_CXX_COMPILER=' + tools['clang++-18']['path'],
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=OFF', '-DGGML_NATIVE=OFF',
             '-DGGML_CPU=ON', '-DGGML_HIP=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
             '-DGGML_METAL=OFF', '-DGGML_SYCL=OFF', '-DGGML_RPC=OFF', '-DGGML_BLAS=OFF',
             '-DGGML_OPENMP=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=ON',
             '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_EXAMPLES=OFF',
             '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF']
    flags.extend(cache_flags(cache_prefix))
    targets = ['llama-completion', 'llama-tokenize', 'llama-quantize', 'llama-perplexity',
               'test-quantize-fns', 'test-rope']
    inputs = {name: sha(ROOT / name) for name in ('llm-inputs.json',
              'tools/build_llm_cpu_reference.py', 'tools/build_jobs.py', 'tools/fetch_locked_deps.py',
              'tools/native_ccache.py', 'tests/llm_cpu_ccache_test.py',
              'tools/prepare_llm_native_loader.py', 'tools/llm_user_loader.cmake', 'tests/llm_user_loader_test.py')}
    identity = dict(source_revision=spec['revision'], source_url=spec['source_url'],
                    tools=tools, flags=flags, targets=targets, inputs=inputs, compiler_cache=cache_record)
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
                                        stdout=stdout, stderr=stderr, timeout=timeout, env=build_env)
            except subprocess.TimeoutExpired:
                commands.append(dict(label=label, argv=[str(arg) for arg in argv],
                                     timed_out=True, deadline_seconds=timeout))
                (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
                raise
        commands.append(dict(label=label, argv=[str(arg) for arg in argv], exit_code=result.returncode))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        result.check_returncode()
        print(label + ' PASS', flush=True)

    loader_text = loader.render_user((client / 'src/llama-model-loader.cpp').read_text())
    if generated_loader.is_symlink() or (generated_loader.exists() and generated_loader.read_text() != loader_text):
        raise RuntimeError('generated CPU user loader differs')
    if not generated_loader.exists():
        generated_loader.write_text(loader_text)
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
    if generated_loader.is_symlink() or generated_loader.read_text() != loader.render_user((client / 'src/llama-model-loader.cpp').read_text()):
        raise RuntimeError('generated CPU user loader changed')
    if any(sha(ROOT / name) != digest for name, digest in inputs.items()):
        raise RuntimeError('build inputs changed during reference qualification')
    for tool in tools.values():
        if sha(tool['path']) != tool['sha256']:
            raise RuntimeError('host tool changed during reference qualification')
    if compiler_cache(os.environ)[0] != cache_record:
        raise RuntimeError('CPU compiler cache identity changed during build')
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
