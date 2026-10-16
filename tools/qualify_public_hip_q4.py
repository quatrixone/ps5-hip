"""Qualify selected upstream Q8_1/Q4_0 public CLR packet and ownership paths offline."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from qualify_public_hip_rmsnorm import ROOT, sha, checked_inputs, checked_tools, mapping, base_inputs, fixture_inputs as prerequisite_inputs
from build_llm_q4_fixture import PROFILES, GET_ROWS, metadata_contract, descriptor_contract, preserved_definitions, preserved_data


def fixture_inputs(folder):
    folder=Path(folder).resolve();receipt=folder/'receipt.json';report=json.loads(receipt.read_text())
    if report.get('classification')=='diagnostic-q4-halfword-fixture':
        from build_llm_q4_halfword_fixture import fixture_inputs as halfword_inputs
        return halfword_inputs(folder)
    getrows=report.get('classification')=='source-only-selected-upstream-getrows-fixture'
    if ((not getrows and report.get('classification')!='source-only-selected-upstream-q4-q8-fixture') or report.get('status')!='PASS' or
            report.get('native_executed') is not False or report.get('gpu_executed') is not False):
        raise ValueError('unqualified selected operator fixture')
    profiles={'getrows':GET_ROWS} if getrows else PROFILES
    if set(report.get('profiles',{}))!=set(profiles):raise ValueError('wrong selected module inventory')
    required=['tools/build_llm_q4_fixture.py','tests/llm_q4_fixture_test.py','tools/qualify_public_hip_rmsnorm.py']
    required += ['.deps/sources/llama-client/ggml/src/ggml-cuda/'+p['source'] for p in profiles.values()]
    if getrows:required.append('tests/llm_getrows_fixture_test.py')
    inputs=checked_inputs(report,'inputs',[ROOT/n for n in required])
    prerequisite=Path(report['prerequisite']).resolve()
    verified=prerequisite_inputs(prerequisite.parent)
    if any(inputs.get(n)!=d for n,d in verified.items()):raise ValueError('prerequisite closure missing')
    inputs.update(checked_tools(report,'tools',('clang++-18','llvm-config-18','llvm-dis-18','llvm-extract-18','llc-18','ld.lld-18','llvm-nm-18','llvm-readobj-18','llvm-objdump-18')))
    artifacts=mapping(report,'artifacts')
    for name,row in artifacts.items():
        path=(folder/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder) or sha(path)!=row.get('sha256') or path.stat().st_size!=row.get('bytes'):raise ValueError('fixture artifact changed')
        inputs[str(path)]=row['sha256']
    for name,profile in profiles.items():
        for suffix in ('.hsaco','.ll','-full.ll','.d','-notes.stdout','-isa.stdout','-undefined.stdout'):
            if name+suffix not in artifacts:raise ValueError('required fixture artifact missing')
        for path in shlex.split((folder/(name+'.d')).read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):
            if str(Path(path).resolve()) not in inputs:raise ValueError('fixture dependency missing')
        selected=report['profiles'][name]
        if selected.get('symbol')!=profile['symbol'] or selected.get('body_preserved') is not True or selected.get('undefined_symbols')!=[] or (folder/(name+'-undefined.stdout')).read_text().strip():raise ValueError('selected module qualification differs')
        if metadata_contract((folder/(name+'-notes.stdout')).read_text(),profile)!=selected['metadata']:raise ValueError('metadata contract differs')
        descriptor_contract(selected['descriptor'],profile)
        original=(folder/(name+'-full.ll')).read_text();extracted=(folder/(name+'.ll')).read_text()
        digest,definitions=preserved_definitions(original,extracted,profile['symbol'])
        if digest!=selected['selected_ir_sha256'] or definitions!=selected['retained_definitions'] or preserved_data(original,extracted,profile['symbol'])!=selected['data_preservation']:raise ValueError('IR preservation receipt differs')
    if not report.get('commands') or any(c.get('exit_code')!=0 for c in report['commands']):raise ValueError('fixture commands incomplete')
    inputs[str(receipt)]=sha(receipt)
    return inputs

def qualify(base, fixture, output):
    base, fixture, output = (Path(p).resolve() for p in (base,fixture,output))
    q8, q4 = fixture/'q8.hsaco', fixture/'q4.hsaco'
    if not output.is_relative_to(ROOT/'results'):
        raise ValueError('output must be under results/')
    inputs = fixture_inputs(fixture)
    original_fixture=dict(inputs)
    original_base=base_inputs(base)
    for name,digest in original_base.items():
        if name in inputs and inputs[name]!=digest: raise ValueError('input changed between receipts: '+name)
        inputs[name]=digest
    for name in ('tools/qualify_public_hip_q4.py', 'tools/qualify_public_hip_rmsnorm.py', 'tests/public_hip_q4_model.cpp', 'tests/clr_adapter_model.cpp'):
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
            if result.returncode != -6 or 'PUBLIC_Q4_PACKET_MISMATCH' not in result.stderr:
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
    argv[argv.index('-c')+1] = str(ROOT/'tests/public_hip_q4_model.cpp')
    model = output/'q4-model.o';argv[argv.index('-o')+1] = str(model)
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
    expected = ['PUBLIC_Q4_PASS modules=2 launches=4 arguments=9,19 kernargs=328,192 threads=256x1,32x4 static_lds=0,384 caller_mutation=true rejects=30 numerical_execution=false gpu_execution=false',
                'PUBLIC_Q4_FINAL ownership=clean gpu_execution=false']
    for eager in (False, True):
        label = 'eager' if eager else 'deferred'
        case = output/label;case.mkdir()
        text = run(label, [binary,q8,q4,'normal'], eager=eager)
        if text.splitlines() != expected:
            raise RuntimeError('Q8/Q4 packet/teardown transcript differs')
    case = output/'mismatch';case.mkdir()
    run('mismatch',[binary,q8,q4,'mismatch'],negative=True)
    if fixture_inputs(fixture)!=original_fixture or base_inputs(base)!=original_base:
        raise RuntimeError('fixture/base identities changed during qualification')
    for name,digest in inputs.items():
        if sha(name) != digest:
            raise RuntimeError('input changed during qualification: '+name)
    result = dict(classification='public-clr-q4-packet-host-model', inputs=inputs,
                  artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                  runtime_executed=True, native_executed=False, gpu_executed=False, numerical_executed=False,
                  cases=['two-selected-source-built-modules','9-and19-arguments','328-and192-kernargs',
                         '256x1-and32x4-threads','static384-dynamic0-LDS','null-ids-and-fusion-pointers',
                         'caller-mutation-after-enqueue','per-function-attributes','30-preflight-rejections',
                         'temporary-retirement-after-sync','deferred-and-eager','packet-mismatch-rejected'],
                  limits=['Packet transport validates selected code backing/entry, arguments and resources only; no numerical output, ISA execution or GPU claim.',
                          'Dynamic module API path, not compiler registration/launch-stub proof.',
                          'Hashes current source-qualified base objects; base receipt does not independently bind every object byte.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args.base,args.fixture,args.output)
