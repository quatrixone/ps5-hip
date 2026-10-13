#!/usr/bin/env python3
"""Actual reduction compiler/runtime/title lifecycle on CPU transport only."""
if not __debug__:raise RuntimeError('optimized Python forbidden')
import pathlib,sys,argparse,json,hashlib,subprocess,os,shutil,shlex
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from check_canary_log import check_log
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
proof=ROOT/'results/companion-runtime-v2';assert sha(proof/'receipt.json')=='7c5f0b58baafb43f3e9e414eac8f7a89d3b07044a24dba13487a61f7acfb06fd';r=json.loads((proof/'receipt.json').read_text());inputs=dict(r['inputs']);inputs.update({str(ROOT/n):h for n,h in r['sources'].items()});inputs.update({str(proof/n):h for n,h in r['artifacts'].items()});inputs[str(proof/'receipt.json')]=sha(proof/'receipt.json');assert all(sha(pathlib.Path(n))==h for n,h in inputs.items())
core=[ROOT/'src'/n for n in ('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp','hip_runtime.cpp','hip_launch_profile.cpp')]
compile_sources=core+[ROOT/'tests/reduction_title_model.cpp',ROOT/'tools/hip_vector_title.cpp',ROOT/'tools/native_shell_exit_probe.cpp'];sources=compile_sources+[pathlib.Path(__file__),ROOT/'tools/check_canary_log.py',ROOT/'tools/math_native_fixture.py',ROOT/'tools/native_runtime.h',ROOT/'tests/reduction_companion_model.cpp'];before={str(p.relative_to(ROOT)):sha(p) for p in sources};tools={n:{'path':shutil.which(n+'-18'),'sha256':sha(pathlib.Path(shutil.which(n+'-18')))} for n in ('clang++','ld.lld')};commands=[]
def run(label,args):
 q=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'));(out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':list(map(str,args)),'exit':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert q.returncode==0,(label,q.stderr[-3000:]);assert not any(x in q.stdout+q.stderr for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer'));return q.stdout+q.stderr
lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';stage=ref/'results/hip-host-abi-spike/rocm-6.2.4/package-stage/include';elevation=ref/'.deps/elevation/client'
common=['-pthread','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_REDUCTION_TITLE=1','-DPS5HIP_RUNTIME_MODEL=1',*[x for d in (ROOT/'include',stage,platform,elevation) for x in ('-I',str(d))]]
objects=[]
for i,src in enumerate(compile_sources):
 extra=[]
 if src.name=='reduction_title_model.cpp':extra=['-std=c++20']
 if src.name=='hip_vector_title.cpp':extra=['-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync','-std=c++20']
 if src.name=='native_shell_exit_probe.cpp':extra=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
 dep=out/f'headers-{i}.d';run(f'dependencies-{i}',[tools['clang++']['path'],*common,*extra,'-M','-MF',dep,src]);words=shlex.split(dep.read_text().replace('\\\n',' '));assert words[0].endswith(':')
 for w in words[1:]:
  hp=pathlib.Path(w);hp=hp if hp.is_absolute() else ROOT/hp;inputs[str(hp.resolve())]=sha(hp)
 obj=out/(src.stem+'.o');run('compile-'+src.stem,[tools['clang++']['path'],*common,*extra,'-c',src,'-o',obj]);objects.append(obj)
combined=ROOT/'results/companion-compiler-v2/linux/combined.o';binary=out/'title';run('link',[tools['clang++']['path'],*common,'-pie','-fuse-ld=lld-18',*objects,combined,'-Wl,-T,'+str(proof/'linux.ld'),'-Wl,--wrap=main','-Wl,--wrap=__hipUnregisterFatBinary','-o',binary])
faults=('corrupt','missing','stale','highbits','tail','input-guard','mask-guard','output-guard','submit','code-image','entry');cases={}
for mode in ('normal','shell-fail','skip-unregister','elevation','config','log-exists','startup-sync','final-sync',*faults):
 folder=out/'cases'/mode;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('invalid' if mode=='config' else str(folder));log=folder/'ps5-hip-gfx1013-hip_reduction.log'
 if mode=='log-exists':log.write_text('prior-result\n')
 captured=run('run-'+mode,[binary,mode,config,ref/'.local/g4-reduction/results/reduction-compiler-v3/linux/reduction.hsaco',proof/'fixture.bin']);text=log.read_text() if log.exists() else captured
 assert not any(x in text for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer'))
 if mode in ('normal','shell-fail'):
  assert 'REDUCTION_TITLE_MODEL STOP shell_requests=1 live=0 releases=21 submits=3 model_only=true' in text
  production='\n'.join(l for l in text.splitlines() if not l.startswith(('REDUCTION_COMPANION_MODEL_FINAL ','REDUCTION_TITLE_MODEL STOP ')))+'\n'
  if mode=='shell-fail':production=production.replace('PS5HIP FAIL: shell-exit request-return=0xfffffff9','PS5HIP SHELL_EXIT request-return=0')
  assert check_log(production.encode(),'gfx1013','hip_reduction',shell_exit_probe=True)['application_result']=='application-pass'
  if mode=='normal':(out/'production.log').write_text(production)
 else:
  assert 'PS5HIP SHELL_EXIT request=begin' not in text
  if mode=='log-exists':assert text=='prior-result\n'
  elif mode in ('config','elevation'):assert 'REDUCTION_TITLE_MODEL PRECLIENT rejected=true' in captured
  else:assert 'HIP_TITLE QUARANTINED' in text and 'REDUCTION_TITLE_MODEL STOP shell_requests=0' in text
 cases[mode]={'log':str(log.relative_to(out)) if log.exists() else 'run-'+mode+'.stdout','sha256':sha(log) if log.exists() else sha(out/('run-'+mode+'.stdout'))}
production=(out/'production.log').read_text();lines=production.splitlines();mutations=[]
for i in range(len(lines)):mutations += ['\n'.join(lines[:i]+lines[i+1:])+'\n','\n'.join(lines[:i]+[lines[i]]+lines[i:])+'\n']
for old,new in [('dispatches=3','dispatches=2'),('logical=16377','logical=16384'),('exact_values=48','exact_values=47'),('guards=72','guards=71'),('functions=1','functions=2'),('input_unchanged=true','input_unchanged=false'),('masks_unchanged=true','masks_unchanged=false')]:mutations.append(production.replace(old,new))
for bad in mutations:assert check_log(bad.encode(),'gfx1013','hip_reduction',shell_exit_probe=True)['application_result']!='application-pass'
assert before=={str(p.relative_to(ROOT)):sha(p) for p in sources};assert all(sha(pathlib.Path(n))==h for n,h in inputs.items());assert all(sha(pathlib.Path(t['path']))==t['sha256'] for t in tools.values())
v={'classification':'reduction-title-host-lifecycle-pass','sources':before,'inputs':inputs,'tools':tools,'cases':cases,'transcript_mutations':len(mutations),'commands':commands,'hardware_executed':False,'native_linked':False,'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}};(out/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(sha(out/'receipt.json'))
