"""Qualify dynamic upstream RMSNorm argument/resource admission through genuine CLR.

Only memory providers and packet transport are modeled; this is not GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
from build_llm_rmsnorm_fixture import selected_metadata, SYMBOL, CONSTANTS
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

FIXTURE_SOURCES = ('tools/build_llm_rmsnorm_fixture.py','tools/fetch_locked_deps.py',
    'tools/probe_ocml_math.py','tools/probe_hip_routes.py','tools/build_canaries.py',
    'tests/fat_binary_fixture_test.py','tools/ocml_math_inputs.json',
    'evidence/offline-llm-hip-compile-probe.json','dependencies.json','llm-inputs.json',
    'ports/clr/manifest.json','.deps/sources/llama-client/ggml/src/ggml-cuda/norm.cu',
    '.deps/sources/llama-client/ggml/src/ggml-cuda/norm.cuh','ports/hip/hip/hip_version.h',
    '.deps/sources/hip-headers/include/hip/hip_runtime.h',
    '.deps/sources/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h')
FIXTURE_TOOLS = ('clang-18','clang++-18','llvm-config-18','llvm-dis-18','llvm-nm-18',
    'llvm-readobj-18','llvm-objdump-18','llvm-link-18','ld.lld-18','opt-18','cmake','ninja')
FIXTURE_ARTIFACTS = ('norm.hsaco','norm.d','norm-notes.stdout','norm-isa.stdout','commands.json','device-tree.json')
BASE_SOURCES = ('ports/clr/ps5-device.cpp','ports/clr/ps5-device.hpp',
    'tools/qualify_public_hip_source.py','tools/prepare_public_hip_resources.py',
    'ports/clr/ps5-public-resources.patch','ports/clr/ps5-public-resources.json',
    'src/code_object.cpp','src/launch.cpp','src/native_launch.cpp','src/host_launch.cpp','src/hip_launch_profile.cpp')
BASE_TOOLS = ('clang++-18','ld.lld-18','llvm-readobj-18','llvm-objdump-18')

def mapping(report, name):
    value=report.get(name)
    if not isinstance(value,dict) or not value: raise ValueError('nonempty '+name+' required')
    return value

def checked_inputs(report, name, required=()):
    values=mapping(report,name)
    paths={str((Path(n) if Path(n).is_absolute() else ROOT/n).resolve()):d for n,d in values.items()}
    if not {str(Path(p).resolve()) for p in required} <= paths.keys(): raise ValueError('required '+name+' identities missing')
    for path,digest in paths.items():
        if sha(path)!=digest: raise ValueError('stale '+name+': '+path)
    return paths

def checked_tools(report, group, required):
    tools=mapping(report,group)
    if not set(required)<=tools.keys(): raise ValueError('required '+group+' missing')
    inputs={}
    for name,row in tools.items():
        if not isinstance(row,dict) or not row.get('path') or not row.get('sha256'): raise ValueError('invalid tool identity')
        path=Path(row['path']).resolve()
        if name in required and (not shutil.which(name) or path!=Path(shutil.which(name)).resolve()): raise ValueError('tool selection changed: '+name)
        if sha(path)!=row['sha256']: raise ValueError('stale tool: '+name)
        inputs[str(path)]=row['sha256']
    return inputs

def fixture_inputs(folder):
    folder=Path(folder).resolve();receipt=folder/'receipt.json';j=json.loads(receipt.read_text())
    if j.get('status')!='PASS' or j.get('native_executed') is not False or j.get('gpu_executed') is not False: raise ValueError('unqualified RMSNorm fixture')
    inputs=checked_inputs(j,'inputs',[ROOT/n for n in FIXTURE_SOURCES])
    inputs.update(checked_tools(j,'tools',FIXTURE_TOOLS));inputs[str(receipt)]=sha(receipt)
    artifacts=mapping(j,'artifacts')
    if not set(FIXTURE_ARTIFACTS)<=artifacts.keys(): raise ValueError('required fixture artifacts missing')
    for name,row in artifacts.items():
        path=(folder/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder): raise ValueError('fixture artifact outside folder')
        if not isinstance(row,dict) or sha(path)!=row.get('sha256') or path.stat().st_size!=row.get('bytes'): raise ValueError('stale fixture artifact')
        inputs[str(path)]=row['sha256']
    # The generated dependency closure must be included, not merely its manifest file.
    for name in shlex.split((folder/'norm.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):
        if str(Path(name).resolve()) not in inputs: raise ValueError('fixture dependency identity missing')
    order=['ockl-workitem-only','mul_hi','ocml',*(n for n in CONSTANTS if n!='oclc_isa_version_1030')]
    if j.get('library_order')!=order: raise ValueError('fixture library order changed')
    for name in order:
        if str(folder/'libs'/(name+'.bc')) not in inputs: raise ValueError('fixture device library identity missing')
    selected=selected_metadata((folder/'norm-notes.stdout').read_text())
    expected=dict(group_bytes=0,private_bytes=0,kernarg_bytes=488,rsrc1=0x60af00c3,rsrc2=0x38c,rsrc3=0,properties=0x409,preload=0)
    if j.get('selected')!=selected or j.get('descriptor')!=expected or j.get('kernel_count')!=16 or j.get('undefined_symbols')!=[]: raise ValueError('fixture metadata contract changed')
    if not j.get('commands') or any(c.get('exit_code')!=0 for c in j['commands']): raise ValueError('fixture commands incomplete')
    return inputs

def base_inputs(base):
    base=Path(base).resolve();p=base/'receipt.json';j=json.loads(p.read_text())
    if j.get('classification')!='source-only-genuine-public-hip-model' or j.get('gpu_executed') is not False or j.get('native_executed') is not False: raise ValueError('unexpected base qualification')
    inputs=checked_inputs(j,'inputs',[ROOT/n for n in BASE_SOURCES])
    prepared=checked_inputs(j,'prepared_sources',[base/'source-public/hipamd/src/hip_module.cpp',base/'source-public/rocclr/platform/kernel.cpp'])
    inputs.update(prepared);inputs.update(checked_inputs(j,'included_files'))
    inputs.update(checked_tools(j,'toolchain',BASE_TOOLS));inputs[str(p)]=sha(p)
    spec=json.loads((ROOT/'ports/clr/ps5-public-resources.json').read_text())
    for name,row in spec['files'].items():
        if prepared.get(str(base/'source-public'/name))!=row['after']: raise ValueError('base resource patch not prepared')
    return inputs

def qualify(base, fixture, output):
    base, fixture, output = (Path(p).resolve() for p in (base,fixture,output))
    norm = fixture/'norm.hsaco'
    if not output.is_relative_to(ROOT/'results'):
        raise ValueError('output must be under results/')
    inputs = fixture_inputs(fixture)
    original_fixture=dict(inputs)
    original_base=base_inputs(base)
    for name,digest in original_base.items():
        if name in inputs and inputs[name]!=digest: raise ValueError('input changed between receipts: '+name)
        inputs[name]=digest
    for name in ('tools/qualify_public_hip_rmsnorm.py', 'tests/public_hip_rmsnorm_model.cpp', 'tests/clr_adapter_model.cpp', 'tests/public_hip_rmsnorm_qualification_test.py'):
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
            if result.returncode != -6 or 'PUBLIC_RMSNORM_PACKET_MISMATCH' not in result.stderr:
                raise RuntimeError('negative output oracle did not fire: '+label)
        elif result.returncode:
            raise RuntimeError(label+': '+result.stderr[-4000:])
        return result.stdout

    matches = []
    for path in base.glob('compile-*.json'):
        record = json.loads(path.read_text())
        if str(ROOT/'tests/public_hip_launch_model.cpp') in record['argv']:
            matches.append((path,record))
    if len(matches) != 1 or matches[0][1]['exit_code'] != 0:
        raise ValueError('expected one successful base launch model compile')
    path, record = matches[0];inputs[str(path)] = sha(path)
    argv = record['argv'][:];original = argv[argv.index('-o')+1]
    argv[argv.index('-c')+1] = str(ROOT/'tests/public_hip_rmsnorm_model.cpp')
    model = output/'rmsnorm-model.o';argv[argv.index('-o')+1] = str(model)
    if '-MF' in argv: argv[argv.index('-MF')+1] = str(output/'model.d')
    dependency=argv[:];dependency.remove('-c')
    at=dependency.index('-o');del dependency[at:at+2]
    dependency += ['-M','-MF',str(output/'model.d')]
    run('model-dependencies', dependency)
    inputs[str(output/'model.d')]=sha(output/'model.d')
    for name in shlex.split((output/'model.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):
        path=str(Path(name).resolve());digest=sha(path)
        if path in inputs and inputs[path]!=digest: raise RuntimeError('dependency changed: '+path)
        inputs[path]=digest
    run('model-compile', argv)
    inputs[str(model)]=sha(model)
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
    run('model-link', argv)
    inputs[str(binary)]=sha(binary)
    expected = ['PUBLIC_RMSNORM_PASS rows=1,2,17 arguments=26 kernargs=488 dynamic_lds=128 packed_nulls=2 caller_mutation=true unsupported_absent=true modules_rejected=true numerical_execution=false gpu_execution=false',
                'PUBLIC_RMSNORM_FINAL ownership=clean gpu_execution=false']
    for eager in (False, True):
        label = 'eager' if eager else 'deferred'
        case = output/label;case.mkdir()
        text = run(label, [binary,norm,'normal',case], eager=eager)
        if text.splitlines() != expected:
            raise RuntimeError('RMSNorm packet/teardown transcript differs')
    case = output/'mismatch';case.mkdir()
    run('mismatch',[binary,norm,'mismatch',case],negative=True)
    if fixture_inputs(fixture)!=original_fixture or base_inputs(base)!=original_base:
        raise RuntimeError('fixture/base identities changed during qualification')
    for name,digest in inputs.items():
        if sha(name) != digest:
            raise RuntimeError('input changed during qualification: '+name)
    result = dict(classification='public-clr-rmsnorm-packet-host-model', inputs=inputs,
                  artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                  runtime_executed=True, native_executed=False, gpu_executed=False, numerical_executed=False,
                  cases=['unchanged-dynamic-module','26-arguments-488-kernargs','rows-1-2-17','1024-threads-128-dynamic-LDS',
                         'optional-null-pointers','caller-mutation-after-enqueue','supported-function-attributes',
                         'unsupported-sibling-lookup-repeated','absent-handle-attribute-rejected',
                         'all-unsupported-and-malformed-modules-rejected','deferred-and-eager','packet-mismatch-rejected'],
                  mutations={'all-unsupported.hsaco':'Set bit20 in every kernel descriptor rsrc1; structurally reinspected before public load.',
                             'malformed.hsaco':'XOR bit0 of ELF magic byte0; all other bytes unchanged.'},
                  limits=['Packet transport validates selected code backing/entry, arguments and resources only; no numerical output, ISA execution or GPU claim.',
                          'Dynamic module API path, not compiler registration/launch-stub proof.',
                          'Null handle attribute rejection does not query a nonexistent sibling handle.',
                          'Hashes current source-qualified base objects; base receipt does not independently bind every object byte.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args.base,args.fixture,args.output)
