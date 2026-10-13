#!/usr/bin/env python3
"""Unchanged upstream HIP main against real runtime and synthetic CPU transport."""
if not __debug__:raise RuntimeError('optimized Python is forbidden')
import argparse,pathlib,sys,json,hashlib,subprocess,shutil,shlex,os,struct
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from probe_hip_routes import prepared_inputs,sha
CORE=('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp','hip_runtime.cpp','hip_launch_profile.cpp')
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--compiler-root',type=pathlib.Path,required=True);p.add_argument('--companion',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();ref=a.reference_root.resolve();C=a.compiler_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sources=[ROOT/'src'/n for n in CORE]+list((ROOT/'include/ps5hip').glob('*.h'))+[ROOT/'tests/reduction_companion_model.cpp',pathlib.Path(__file__),ROOT/'tools/probe_hip_routes.py',ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py'];before={str(p.relative_to(ROOT)):sha(p) for p in sources}
proof=C/'results/reduction-compiler-v3';compiler=json.loads((proof/'receipt.json').read_text());assert sha(proof/'receipt.json')=='8cfc7008a5d93abbd09f0d88bfe754dabf06d3e79415860199dee9416696267b'
_,stage,libs,inputs=prepared_inputs(ref);inputs[str(proof/'receipt.json')]=sha(proof/'receipt.json');cp=a.companion.resolve();cr=json.loads((cp/'receipt.json').read_text());assert cr['classification']=='bound-declaration-only-reduction-companion-compiler';assert cr['original_authority']['sha256']==sha(proof/'receipt.json');inputs[str(cp/'receipt.json')]=sha(cp/'receipt.json');inputs.update(cr['sources']);inputs.update(cr['inputs']);inputs.update({v['path']:v['sha256'] for v in cr['tools'].values()});inputs.update({str(cp/p):h for p,h in cr['artifacts'].items()});assert all(sha(pathlib.Path(p))==h for p,h in inputs.items());obj=cp/'linux/combined.o';inputs[str(obj)]=sha(obj);inputs[str(proof/'linux/reduction.hsaco')]=sha(proof/'linux/reduction.hsaco');inputs[str(C/'tests/reduction_compiler_test.py')]=sha(C/'tests/reduction_compiler_test.py')
lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';cxx=pathlib.Path(shutil.which('clang++-18'));tool_hash=sha(cxx);linker=pathlib.Path(shutil.which('ld.lld-18'));linker_hash=sha(linker);commands=[]
def run(label,argv,expected=0):
 q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0',LLVM_CONFIG=shutil.which('llvm-config-18')));(out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':list(map(str,argv)),'exit':q.returncode,'expected':expected});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert q.returncode==expected,(label,q.stderr[-3000:]);return q.stdout+q.stderr
run('compiler-authority',['python3',C/'tests/reduction_compiler_test.py','--compiler',proof])
script=out/'linux.ld';script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
common=['-pthread','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1','-fPIE','-I'+str(ROOT/'include'),'-I'+str(stage/'include'),'-I'+str(platform)]
compile_sources=[ROOT/'src'/n for n in CORE]+[ROOT/'tests/reduction_companion_model.cpp']
for i,src in enumerate(compile_sources):
 dep=out/('headers-'+str(i)+'.d');run('dependencies-'+str(i),[cxx,*common,'-M','-MF',dep,src]);words=shlex.split(dep.read_text().replace('\\\n',' '));assert words[0].endswith(':')
 for w in words[1:]:
  hp=pathlib.Path(w);hp=hp if hp.is_absolute() else ROOT/hp;inputs[str(hp.resolve())]=sha(hp)
binary=out/'upstream-model';run('build',[cxx,*common,'-pie','-fuse-ld=lld-18',*compile_sources,obj,'-Wl,-T,'+str(script),'-Wl,--wrap=main','-o',binary])
fixture=ref/'results/reduction-independent-oracle-v1/fixture.json';assert sha(fixture)=='1c7220b801f435ccff3bee195b050041bb175b2285ee97cf90a70265afa77793';inputs[str(fixture)]=sha(fixture);fv=json.loads(fixture.read_text());payload=b''
for case in fv['cases']:
 payload+=struct.pack('<16384i',*case['input'])+struct.pack('<512Q',*case['masks'])
fixture_binary=out/'fixture.bin';fixture_binary.write_bytes(payload)
for mode,code in [('normal',0),('corrupt',20),('missing',20),('stale',20),('highbits',20),('tail',20),('input-guard',20),('mask-guard',20),('output-guard',20),('submit',30),('code-image',30),('entry',30)]:
 text=run('run-'+mode,[binary,mode,proof/'linux/reduction.hsaco',fixture_binary],code);assert not any(x in text for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer'))
 if code==30:assert 'REDUCTION_COMPANION_TERMINAL retained=7 releases=0 submits=1' in text and 'REDUCTION_COMPANION_MODEL_FINAL' not in text
 else:
  count=3 if mode in ('normal','tail') else 2 if mode=='stale' else 1
  assert f'REDUCTION_COMPANION_MODEL_FINAL live=0 backends=1 submits={count} releases={7*count} finalized=1' in text
  assert ('REDUCTION_CLIENT_PASS dispatches=3 exact_values=48 guards=72 allocations=9 frees=9' if mode=='normal' else 'REDUCTION_CLIENT_FAILURE terminal=false clean=1') in text
assert before=={str(p.relative_to(ROOT)):sha(p) for p in sources};assert all(sha(pathlib.Path(p))==h for p,h in inputs.items());assert sha(cxx)==tool_hash;assert sha(linker)==linker_hash
r={'classification':'deterministic-reduction-companion-runtime-model-pass','compiler_proof':{'path':str(proof/'receipt.json'),'sha256':sha(proof/'receipt.json')},'sources':before,'inputs':inputs,'compiler':{'path':str(cxx),'sha256':tool_hash},'linker':{'path':str(linker),'sha256':linker_hash},'commands':commands,'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'hardware_executed':False,'limits':['CPU synthetic transport computes direct sums, not GPU ISA emulation.','Original kernel object and declaration-only companion object are uninstrumented; actual runtime/transport are ASan/UBSan instrumented.','All actual client inputs and masks match parent independent fixture bytes; full original code image and entry are checked.','Companion directly rejects math/API/guard failures; synthetic transport is not GPU execution.']};(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(sha(out/'receipt.json'))
