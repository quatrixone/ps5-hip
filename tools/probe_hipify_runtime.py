#!/usr/bin/env python3
"""Unedited HIPIFY client through the real runtime and an explicit host transport model."""
import argparse, hashlib, json, os, pathlib, re, shutil, subprocess
import probe_hipify_client as translation
import probe_hip_routes as shared
from probe_hip_generic import CORE
from build_canaries import require_device_linker, require_tool_versions
from fat_binary_fixture_test import section

ROOT = pathlib.Path(__file__).resolve().parents[1]
sha = translation.sha
require = translation.require
MODES = ('normal', 'data-allocate', 'init', 'preflush', 'missing', 'corrupt', 'input',
         'guard', 'stale-second', 'partial', 'late-partial', 'no-finish', 'submit',
         'kick', 'timeout', 'throw', 'unknown', 'postflush', 'cleanup')
TERMINAL = frozenset(('submit', 'kick', 'timeout', 'throw', 'unknown', 'postflush', 'cleanup'))
# backends, submissions, releases, completed, finalized: safe cleanup and oracle success differ.
COUNTS = {'normal': (1, 6, 33, 6, 1), 'data-allocate': (0, 0, 1, 0, 1),
          'init': (1, 0, 5, 0, 0), 'preflush': (1, 0, 7, 0, 1),
          **{m: (1, 1, 7, 1, 1) for m in ('missing', 'corrupt', 'input', 'guard')},
          'stale-second': (1, 2, 11, 2, 1), 'partial': (0, 0, 0, 0, 0),
          'late-partial': (1, 6, 33, 6, 0), 'no-finish': (1, 6, 33, 6, 0)}
CLIENT_PASS = 'TRANSLATED_CLIENT_PASS launches=6 exact_u32=true inputs_unchanged=true full_guards=true'

def validate_model(text, mode, exit_code):
    require(mode in MODES, 'unknown model mode')
    require(text.splitlines().count(f'HIPIFY_MODEL_BEGIN mode={mode} model_only=true') == 1, 'exact model mode')
    require(not re.search(r'AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:|HIPIFY_MODEL_FAILURE', text), 'model/sanitizer diagnostic')
    require(text.count('VECTOR_REGISTER owned=true gpu_calls=0\n') == 1 and text.count('VECTOR_FUNCTION bound=true\n') == 1, 'genuine registration inventory')
    if mode in TERMINAL:
        require(exit_code == 30, 'quarantine exit')
        live, freed, unresolved = (4, 3, 0) if mode == 'cleanup' else (7, 0, 1)
        line = f'HIPIFY_QUARANTINE live={live} releases={freed} unresolved={unresolved} submits=1 no_retry=true'
        require(text.splitlines().count(line) == 1, 'terminal backing retention')
        require(re.search(r'^TRANSLATED_CLIENT_FAIL clean=0 error=.+$', text, re.M), 'terminal client cleanup rejection')
        require(not any(n in text for n in ('TRANSLATED_CLIENT_PASS', 'HIPIFY_RUNTIME_PASS', 'HIPIFY_MODEL_FINAL', 'VECTOR_FINAL', 'VECTOR_UNREGISTER', 'HIPIFY_EXPECTED_REJECT', 'RUNTIME_COMPLETE')), 'terminal false closure')
        return
    require(exit_code == (0 if mode == 'normal' else 20), 'model exit')
    finals = re.findall(r'^HIPIFY_MODEL_FINAL live=(\d+) backends=(\d+) submits=(\d+) releases=(\d+) completed=(\d+) modules=(\d+) functions=(\d+) finalized=(\d+) model_only=true$', text, re.M)
    require(len(finals) == 1, 'single final model record')
    live, backend, submitted, released, completed, modules, functions, finalized = map(int, finals[0])
    unfinished = mode == 'no-finish'
    require(live == 0 and modules == functions == int(unfinished) and (backend, submitted, released, completed, finalized) == COUNTS[mode], 'exact failure point/ownership/finalization counts')
    unregister = 'VECTOR_UNREGISTER rejected=true' if unfinished else 'VECTOR_UNREGISTER after_main=true active=0'
    require(text.splitlines().count(unregister) == 1 and text.count('VECTOR_UNREGISTER ') == 1, 'compiler unregister outcome')
    require(text.splitlines().count(f'VECTOR_FINAL registered=1 removed={int(not unfinished)} completed={completed} active={int(unfinished)} terminal=0 main_finished={int(not unfinished)} hardware_validated=false') == 1, 'exact runtime finalization record')
    require(re.findall(r'^RUNTIME_COMPLETE sequence=(\d+) kernel=translated_add blocking=true$', text, re.M) == [str(i) for i in range(1,completed+1)], 'ordered runtime completion records')
    success = mode in ('normal', 'late-partial', 'no-finish')
    require(text.splitlines().count(CLIENT_PASS) == int(success), 'original client oracle result')
    require(text.count('TRANSLATED_CLIENT_FAIL ') == int(not success), 'original client failure inventory')
    if not success: require(re.search(r'^TRANSLATED_CLIENT_FAIL clean=1 error=.+$', text, re.M), 'clean failure unwind')
    if mode == 'normal':
        require(text.splitlines().count('HIPIFY_RUNTIME_PASS launches=6 allocations=closed stale_rejected=true') == 1 and 'HIPIFY_EXPECTED_REJECT' not in text, 'runtime closure result')
    else:
        result = 0 if success else 1
        require(text.splitlines().count(f'HIPIFY_EXPECTED_REJECT mode={mode} result={result} completed={completed}') == 1 and 'HIPIFY_RUNTIME_PASS' not in text, 'exact rejection record')
    require('HIPIFY_QUARANTINE' not in text, 'unexpected quarantine')
    ordered = ('VECTOR_REGISTER ', 'VECTOR_FUNCTION ', 'HIPIFY_MODEL_BEGIN ',
               'HIPIFY_RUNTIME_PASS ' if mode=='normal' else 'HIPIFY_EXPECTED_REJECT ',
               'VECTOR_UNREGISTER ', 'VECTOR_FINAL ', 'HIPIFY_MODEL_FINAL ')
    require([text.index(t) for t in ordered] == sorted(text.index(t) for t in ordered), 'compiler/main/destructor order')

def check(output):
    receipt = json.loads((output/'receipt.json').read_text())
    require(receipt['classification'] == 'hipify-owned-runtime-host-model-pass' and receipt['hardware_executed'] is False, 'receipt scope')
    for name, digest in receipt['sources'].items(): require(sha(ROOT/name) == digest, 'source changed: '+name)
    for name, digest in receipt['inputs'].items(): require(sha(name) == digest, 'input changed: '+name)
    for name, item in receipt['tools'].items(): require(sha(item['path']) == item['sha256'], 'tool changed: '+name)
    for name, digest in receipt['artifacts'].items(): require(sha(output/name) == digest, 'artifact changed: '+name)
    require([m['mode'] for m in receipt['models']] == list(MODES), 'model inventory')
    require([c['host'] for c in receipt['cases']] == ['linux', 'ps5', 'linux-model'], 'compiler inventory')
    for model in receipt['models']:
        stem = 'run-'+model['mode']
        validate_model((output/(stem+'.stdout')).read_text()+(output/(stem+'.stderr')).read_text(), model['mode'], model['exit_code'])
    print('HIPIFY saved runtime proof: three genuine objects, nineteen ownership/oracle models; no GPU execution')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args(); ref = args.reference_root.resolve(); out = args.output.resolve()
    require(not out.exists(), 'fresh output required'); out.mkdir(parents=True)
    commands = []; inputs = {}; models = []; cases = []
    sources = [ROOT/'src'/n for n in CORE]+[ROOT/'src/hip_launch_profile.cpp', ROOT/'tests/cuda_vector_add.cu', ROOT/'tests/hipify_runtime_model.cpp', ROOT/'tests/hipify_runtime_recipe_test.py', ROOT/'docs/HIPIFY_RUNTIME.md', pathlib.Path(__file__), ROOT/'tools/probe_hipify_client.py', ROOT/'tools/probe_hip_generic.py', ROOT/'tools/probe_hip_routes.py', ROOT/'tools/build_canaries.py', ROOT/'tests/fat_binary_fixture_test.py', ROOT/'tests/hip_launch_profile_test.cpp', *sorted((ROOT/'include/ps5hip').glob('*.h'))]
    sources.append(ROOT/'CMakeLists.txt')
    before = {str(f.relative_to(ROOT)): sha(f) for f in sources}
    def run(label, argv, expected=0):
        argv = list(map(str, argv)); env = dict(os.environ, LLVM_CONFIG=shutil.which('llvm-config-18'), ASAN_OPTIONS='protect_shadow_gap=0')
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
        (out/(label+'.stdout')).write_text(result.stdout); (out/(label+'.stderr')).write_text(result.stderr)
        commands.append({'label': label, 'argv': argv, 'exit_code': result.returncode, 'expected': expected, 'environment': {'ASAN_OPTIONS': env['ASAN_OPTIONS']}})
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        require(result.returncode == expected, label+' failed: '+result.stderr[-3000:])
        return result.stdout, result.stderr
    try:
        checkpoint = ref/'.local/g4-hipify/results/compiler-v1'; authority = checkpoint/'receipt.json'
        require(sha(authority) == '5268dc54b57e1c6fc6584c78fd2f6879826fb983d990e0d84bf55656f404f2d5', 'compiler checkpoint changed')
        old = json.loads(authority.read_text()); inputs[str(authority)] = sha(authority)
        review = ref/'results/hipify-independent-review-v1/review.json'; inputs[str(review)] = sha(review)
        require(json.loads(review.read_text())['classification'] == 'clear-offline-hipify-translation-compiler-proof', 'checkpoint independent review missing')
        inputs.update(old['inputs']); inputs.update({str(checkpoint/n): v for n,v in old['artifacts'].items()})
        require(all(sha(n) == v for n,v in inputs.items()), 'checkpoint input/artifact changed')
        source = ROOT/'tests/cuda_vector_add.cu'
        require(sha(source) == 'ecd50a2a6c8cfc4f58702b8725dfca2ae25dbf4049a358cc988562e8f1c2922a', 'original CUDA client changed')
        translator = ref/'.local/g4-hipify/.local/hipify-6.2.4/bin/hipify-perl'
        require(sha(translator) == translation.TRANSLATOR_SHA, 'translator changed'); inputs[str(translator)] = sha(translator)
        translator_root = translator.parent.parent
        require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=translator_root,text=True).strip() == translation.REVISION, 'HIPIFY revision changed')
        require(not subprocess.check_output(['git','status','--porcelain'],cwd=translator_root,text=True), 'HIPIFY checkout dirty')
        perl = pathlib.Path(shutil.which('perl')); inputs[str(perl)] = sha(perl)
        translated, _ = run('translation', [perl, translator, source]); translation.check_translation(source.read_text(), translated)
        require(translated.encode() == (checkpoint/'translated.hip').read_bytes(), 'generated HIP bytes differ from checkpoint')
        hip = out/'translated.hip'; hip.write_text(translated)
        _, stage, libs, prepared = shared.prepared_inputs(ref); inputs.update(prepared)
        lock = json.loads((ROOT/'dependencies.json').read_text())
        tools = {n: shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')}
        require(all(tools.values()), 'LLVM18 tools missing')
        versions = {n: run(n+'-version', [v, '--version'])[0] for n,v in tools.items()}; require_tool_versions(versions, lock['llvm']['version'])
        tool_hashes = {n: sha(v) for n,v in tools.items()}
        sdk = (ref/'.deps/sdk').resolve(); wrapper = sdk/'bin/prospero-clang++'; inputs[str(wrapper)] = sha(wrapper)
        require((sdk/'.ps5-sdk-revision').read_text().strip() == lock['sdk']['revision'], 'SDK revision changed')
        for host in ('linux','ps5','linux-model'):
            d = out/host; d.mkdir(); obj = d/'translated.o'
            command = [wrapper if host=='ps5' else tools['clang++'], '-x','hip', '-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent), '--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-Dmain=hipify_client_main','-std=c++17','-O2','-fno-fast-math','-c',hip,'--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
            if host=='ps5': command += ['-Xarch_device','-fno-short-wchar']
            if host=='linux-model': command += ['-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
            _, trace = run(host+'-driver', command+['-###']); require_device_linker(trace, tools['ld.lld']); run(host+'-compile', command)
            bundle = section(obj.read_bytes(), b'.hip_fatbin'); (d/'translated.bundle').write_bytes(bundle)
            code = d/'translated.hsaco'; code.write_bytes(translation.device_object(bundle))
            require(sha(code) == old['cases'][len(cases)]['device_sha256'], 'renaming main changed selected device object')
            metadata,_ = run(host+'-metadata', [tools['llvm-readobj'],'--notes',code]); isa,_ = run(host+'-isa', [tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code])
            audit = shared.audit_object(code.read_bytes(), metadata, isa, (translation.NAME,))
            require(audit == old['cases'][len(cases)]['audit'], 'device profile/ISA changed')
            hostread,_ = run(host+'-host-readobj', [tools['llvm-readobj'],'--sections','--symbols','--relocations',obj]); assembly,_ = run(host+'-host-isa', [tools['llvm-objdump'],'-dr',obj])
            require(re.search(r'^    Name: _Z18hipify_client_mainv \(', hostread, re.M) and not re.search(r'^    Name: main \(', hostread, re.M), 'genuine renamed no-argument entry missing')
            lifecycle = shared.audit_registration(assembly, (translation.NAME,)); wrapper_audit = shared.audit_wrapper(section(obj.read_bytes(),b'.hipFatBinSegment'),hostread,host)
            cases.append({'host':host,'device_sha256':sha(code),'audit':audit,'lifecycle':lifecycle,'wrapper':wrapper_audit})
            print(host+' unchanged shader '+sha(code), flush=True)
        platform = ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
        inputs.update({str(f):sha(f) for f in platform.rglob('*.h') if f.is_file()})
        common = ['-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-I',ROOT/'include','-I',stage/'include','-I',platform,'-fsanitize=address,undefined','-fno-omit-frame-pointer']
        policy = ROOT/'src/hip_launch_profile.cpp'
        profile = out/'profile-test'; run('profile-build',[tools['clang++'],*common,ROOT/'tests/hip_launch_profile_test.cpp',policy,ROOT/'src/code_object.cpp',ROOT/'src/host_launch.cpp',ROOT/'src/launch.cpp','-o',profile])
        run('profile-test',[profile,*[out/c['host']/'translated.hsaco' for c in cases]])
        script = out/'linux.ld'; script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
        binary = out/'runtime-model'; run('runtime-build',[tools['clang++'],*common,'-DPS5HIP_RUNTIME_MODEL=1','-fPIE','-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE],policy,ROOT/'tests/hipify_runtime_model.cpp',out/'linux-model/translated.o','-Wl,-T,'+str(script),'-o',binary])
        for mode in MODES:
            expected = 30 if mode in TERMINAL else 0 if mode=='normal' else 20
            stdout,stderr = run('run-'+mode,[binary,mode],expected); validate_model(stdout+stderr,mode,expected)
            models.append({'mode':mode,'exit_code':expected}); print('model '+mode+' passed',flush=True)
        require(before == {str(f.relative_to(ROOT)):sha(f) for f in sources}, 'source drift')
        require(all(sha(n) == v for n,v in inputs.items()), 'input drift')
        require(all(sha(tools[n]) == v for n,v in tool_hashes.items()), 'tool drift')
        receipt = {'schema':1,'classification':'hipify-owned-runtime-host-model-pass','sources':before,'inputs':inputs,'tools':{n:{'path':tools[n],'sha256':v,'version':versions[n]} for n,v in tool_hashes.items()},'cases':cases,'models':models,'commands':commands,'generated_source_hand_edits':0,'original_main_renamed_by_flag':True,'cuda_executed':False,'native_linked':False,'hardware_executed':False,'limits':['Exact translated client; host callbacks do not execute or emulate AMD ISA.','Serialized reusable single-module runtime; no general CUDA compatibility claim.','ASan model uses fixed shader-window mappings with protect_shadow_gap=0; this does not qualify native mapping.','Installed SDK/resource/static host support trees not fully inventoried; no hermetic build claim.']}
        receipt['artifacts'] = {str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print('PASS '+sha(out/'receipt.json'))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'sources_before':before,'inputs_before':inputs,'commands':commands},indent=2)+'\n'); raise

if __name__ == '__main__': main()
