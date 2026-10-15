"""Qualify real HIP aggregate compiler stubs through source-qualified public CLR.

Only memory providers and packet transport are modeled; this is not GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from embedded_module_linker import linux_linker_fragment
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


COMPILER_SOURCE_INPUTS = (
    'tests/hip_aggregate_fixture.h', 'tests/hip_aggregate_fixture.hip',
    'tests/hip_aggregate_stub_observer.cpp', 'tests/hip_aggregate_compiler_test.py',
    'tools/ps5-clangxx', '.deps/sdk/bin/prospero-clang++', '.deps/sdk/bin/clang++',
    'ports/hip/hip/hip_version.h', '.deps/sources/hip-headers/include/hip/hip_runtime.h',
    '.deps/sources/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h')
COMPILER_TOOLS = ('clang++-18','ld.lld-18','llvm-readobj-18','llvm-nm-18','clang-offload-bundler-18')
COMPILER_ARTIFACTS = {'fixture.o','fixture.hsaco','ps5-fixture.o','commands.json','inputs.json',
                      'linux-host.d','linux-device.d','ps5-host.d','ps5-device.d'}

def compiler_inputs(compiler):
    compiler=Path(compiler).resolve()
    receipt=json.loads((compiler/'receipt.json').read_text())
    if receipt.get('classification')!='public-hip-genuine-aggregate-compiler-fixture' or receipt.get('tests_passed')!=2 or receipt.get('gpu_executed') is not False:
        raise ValueError('unexpected compiler fixture receipt')
    if not isinstance(receipt.get('inputs'),dict) or not receipt['inputs'] or not isinstance(receipt.get('artifacts'),dict) or not receipt['artifacts']:
        raise ValueError('compiler receipt requires nonempty inputs and artifacts')
    identities={(Path(name) if Path(name).is_absolute() else ROOT/name).absolute() for name in receipt['inputs']}
    required={ (ROOT/name).absolute() for name in COMPILER_SOURCE_INPUTS }
    required.add(Path(sys.executable).absolute())
    for name in COMPILER_TOOLS:
        tool=shutil.which(name)
        if not tool: raise ValueError('missing compiler tool: '+name)
        required.add(Path(tool).absolute())
    if not required.issubset(identities):
        raise ValueError('compiler receipt omits required source, header or tool identity')
    if not COMPILER_ARTIFACTS.issubset(receipt['artifacts']):
        raise ValueError('compiler receipt omits required objects or dependency manifests')
    inputs={str(compiler/'receipt.json'):sha(compiler/'receipt.json')}
    for name,digest in receipt['inputs'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(p)!=digest: raise ValueError('compiler input changed: '+name)
        inputs[str(p)]=digest
    for name,digest in receipt['artifacts'].items():
        if Path(name).is_absolute(): raise ValueError('absolute compiler artifact path')
        p=(compiler/name).resolve()
        if not p.is_relative_to(compiler): raise ValueError('compiler artifact outside fixture')
        if sha(p)!=digest: raise ValueError('compiler artifact changed: '+name)
        inputs[str(p)]=digest
    if 'fixture.o' not in receipt['artifacts'] or 'fixture.hsaco' not in receipt['artifacts']:
        raise ValueError('compiler receipt omits required objects')
    return inputs


def qualify(base, compiler, output):
    base, compiler, output = (Path(p).resolve() for p in (base,compiler,output))
    if not output.is_relative_to(ROOT/'results'):
        raise ValueError('output must be under results/')
    receipt = json.loads((base/'receipt.json').read_text())
    if receipt['classification'] != 'source-only-genuine-public-hip-model':
        raise ValueError('unexpected base qualification')
    inputs = compiler_inputs(compiler)
    for group in ('inputs', 'prepared_sources', 'included_files'):
        for name, digest in receipt[group].items():
            p = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(p) != digest:
                raise ValueError('stale base input: '+name)
            inputs[str(p)] = digest
    for tool in receipt.get('toolchain', {}).values():
        if sha(tool['path']) != tool['sha256']:
            raise ValueError('stale base tool')
        inputs[tool['path']] = tool['sha256']
    for name in ('tools/qualify_public_hip_aggregate.py',
                 'tools/embedded_module_linker.py', 'tests/hip_aggregate_compiler_test.py',
                 'tests/hip_aggregate_fixture.h', 'tests/hip_aggregate_fixture.hip',
                 'tests/hip_aggregate_stub_observer.cpp', 'tests/public_hip_aggregate_model.cpp',
                 'tests/clr_adapter_model.cpp', 'tools/ps5-clangxx', '.deps/sdk/bin/prospero-clang++', '.deps/sdk/bin/clang++'):
        inputs[str(ROOT/name)] = sha(ROOT/name)
    inputs[str(base/'receipt.json')] = sha(base/'receipt.json')
    for name in ('clang++-18', 'ld.lld-18', 'llvm-readobj-18', 'llvm-nm-18', 'clang-offload-bundler-18'):
        found = shutil.which(name)
        if not found:
            raise ValueError('missing tool: '+name)
        inputs[found] = sha(found)
    inputs[sys.executable] = sha(sys.executable)
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, argv, negative=False, eager=False, extra_env=None):
        argv = list(map(str, argv))
        env = dict(os.environ, ASAN_OPTIONS='protect_shadow_gap=0:halt_on_error=1:detect_leaks=1',
                   UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', HIP_USE_RUNTIME_UNBUNDLER='0',
                   HIP_ENABLE_DEFERRED_LOADING='0' if eager else '1')
        env.update(extra_env or {})
        try:
            result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=180, env=env)
        except subprocess.TimeoutExpired as error:
            for suffix, value in (('stdout', error.stdout), ('stderr', error.stderr)):
                (output/(label+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
            commands.append(dict(label=label, argv=argv, timed_out=True))
            (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            raise
        for suffix, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            (output/(label+'.'+suffix)).write_text(text)
        commands.append(dict(label=label, argv=argv, exit_code=result.returncode, eager=eager))
        (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if negative:
            if result.returncode != -6 or 'PUBLIC_AGGREGATE_OUTPUT_MISMATCH index=0' not in result.stderr:
                raise RuntimeError('negative output oracle did not fire: '+label)
        elif result.returncode:
            raise RuntimeError(label+': '+result.stderr[-4000:])
        return result.stdout

    compiler_object = compiler/'fixture.o';inputs[str(compiler_object)] = sha(compiler_object)
    matches = []
    for path in base.glob('compile-*.json'):
        record = json.loads(path.read_text())
        if str(ROOT/'tests/public_hip_launch_model.cpp') in record['argv']:
            matches.append((path,record))
    if len(matches) != 1 or matches[0][1]['exit_code'] != 0:
        raise ValueError('expected one successful base launch model compile')
    path, record = matches[0];inputs[str(path)] = sha(path)
    argv = record['argv'][:];original = argv[argv.index('-o')+1]
    argv[argv.index('-c')+1] = str(ROOT/'tests/public_hip_aggregate_model.cpp')
    model = output/'aggregate-model.o';argv[argv.index('-o')+1] = str(model)
    run('model-compile', argv)
    path = base/'launch-link.json';inputs[str(path)] = sha(path)
    link_record = json.loads(path.read_text())
    if link_record['exit_code'] != 0:
        raise ValueError('base launch link failed')
    argv = link_record['argv'][:]
    if argv.count(original) != 1:
        raise ValueError('expected exactly one base launch model object')
    for name in argv:
        if name.endswith('.o'):
            inputs[name] = sha(name)
    argv[argv.index(original)] = str(model)
    binary = output/'model';argv[argv.index('-o')+1] = str(binary)
    linker = output/'aggregate.ld';linker.write_text(linux_linker_fragment([compiler_object.name]))
    run('model-link', argv+['-pie',compiler_object,'-Wl,-T,'+str(linker)])
    for eager in (False, True):
        text = run('eager' if eager else 'deferred', [binary,'normal'], eager=eager)
        expected = ['PUBLIC_AGGREGATE_PASS compiler_stubs=true captures=2 invalid_calls=2 gap_zero=true caller_mutation=true guards=8 gpu_execution=false',
                    'PUBLIC_AGGREGATE_FINAL ownership=clean gpu_execution=false']
        if text.splitlines() != expected:
            raise RuntimeError('aggregate output or teardown differs')
    for mode in ('noop','corrupt'):
        run(mode,[binary,mode],negative=True)
    for name,digest in inputs.items():
        if sha(name) != digest:
            raise RuntimeError('input changed during qualification: '+name)
    result = dict(classification='genuine-compiler-public-clr-aggregate-host-model', inputs=inputs,
                  artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                  runtime_executed=True, native_executed=False, gpu_executed=False,
                  cases=['12/16/128-byte-aggregate','real-compiler-stubs','metadata-layout-and-gap','deferred-and-eager',
                         'caller-mutation-after-enqueue','unsupported-geometry/shared-memory-then-valid','two-captures',
                         'eight-output-guard-bytes','clean-modeled-ownership','noop-rejected','corruption-rejected'],
                  limits=['Packet transport copies bytes in the host model; it does not execute GPU ISA.',
                          'Aggregate internal padding is observed but not a portable C++ member value.',
                          'No nested-pointer lifetime or pointer-free metadata guarantee.',
                          'Hashes current source-qualified base objects; base receipt does not independently bind every object byte.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args.base,args.compiler,args.output)
