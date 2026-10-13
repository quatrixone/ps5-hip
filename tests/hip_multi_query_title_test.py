"""Actual two-module title lifecycle with finite modeled host services."""
import argparse,hashlib,json,os,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'tools'),str(R/'tests')]
from hip_multi_query_client_test import MODES
from check_canary_log import check_log
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
CASES=(*MODES,'missing-function','bad-wrapper','duplicate-token','skip-unregister','reject-unregister','no-finish','late-live','post-finish','startup-sync','final-sync','shell-fail','config','elevation','log-exists')
def verify(out):
 cases=json.loads((out/'cases.json').read_text());assert set(cases)==set(CASES)
 for mode,record in cases.items():
  log=(out/record['log']).read_text() if record['log'] else ''
  if mode in ('config','elevation','log-exists'):
   assert ('MULTI_TITLE EARLY_REJECT untouched=true' in (out/record['stdout']).read_text())
   assert log==('previous-result\n' if mode=='log-exists' else '')
   continue
  success=mode in ('normal','post-finish','shell-fail')
  assert log.count('MULTI_TITLE STOP ')==1 and f'MULTI_TITLE STOP shell={int(success)} ' in log
  if not success:assert 'PS5HIP SHELL_EXIT request=begin' not in log
  if mode=='late-live':assert 'MULTI_TITLE PRE_FINISH live=1' in log and 'reason=client-or-ownership' in log
  if mode=='post-finish':assert 'MULTI_TITLE POST_FINISH rejected=true' in log
  if mode=='startup-sync':assert 'reason=startup-log' in log
  if mode=='final-sync':assert 'reason=final-log' in log
  if mode in ('skip-unregister','reject-unregister'):assert 'reason=finalization' in log and 'RUNTIME_RETAINED' in log
  production='\n'.join(x for x in log.splitlines() if not x.startswith(('MULTI_TITLE STOP ','MULTI_TITLE POST_FINISH ')))+'\n'
  if mode=='shell-fail':production=production.replace('PS5HIP FAIL: shell-exit request-return=0xfffffff9','PS5HIP SHELL_EXIT request-return=0')
  assert (check_log(production.encode(),'gfx1013','hip_multi_query',shell_exit_probe=True)['application_result']=='application-pass')==success,mode
 normal=(out/cases['normal']['log']).read_text();normal='\n'.join(x for x in normal.splitlines() if not x.startswith('MULTI_TITLE STOP '))+'\n'
 mutations=[]
 for line in normal.splitlines():mutations += [normal.replace(line+'\n','',1),normal.replace(line+'\n',line+'\n'+line+'\n',1)]
 for old,new in (('modules=2','modules=1'),('removed=2','removed=1'),('success=32','success=31'),('rejected=8','rejected=7'),('module=B','module=A'),('comparisons=522','comparisons=521'),('active=1','active=0')):mutations.append(normal.replace(old,new))
 for bad in mutations:assert check_log(bad.encode(),'gfx1013','hip_multi_query',shell_exit_probe=True)['application_result']!='application-pass'
 return normal,len(mutations)
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',type=pathlib.Path);p.add_argument('--proof',type=pathlib.Path);p.add_argument('--reference-root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
 if a.check:
  out=a.check.resolve();r=json.loads((out/'receipt.json').read_text());assert r['classification']=='offline-two-module-title-lifecycle-pass' and r['cases']==len(CASES)
  assert all(r.get(k) for k in ('sources','inputs','artifacts'))
  required={'tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h','tools/hip_vector_title.cpp','tools/native_runtime.h','tools/native_shell_exit_probe.cpp','tests/hip_multi_query_title_model.cpp','tests/hip_multi_query_title_test.py','tools/check_canary_log.py','tools/math_native_fixture.py','tests/hip_multi_query_client_test.py','src/hip_runtime.cpp','tests/hip_multi_a.hip','tests/hip_multi_b.hip'}
  assert required<=set(r['sources'])
  assert {'title','commands.json','cases.json','production.log',*[f'run-{m}{s}' for m in CASES for s in ('.stdout','.stderr')]}<=set(r['artifacts'])
  assert all(H(R/n)==h for n,h in r['sources'].items()) and all(H(pathlib.Path(n))==h for n,h in r['inputs'].items()) and all(H(out/n)==h for n,h in r['artifacts'].items())
  fixture,mutations=verify(out);assert mutations==r['mutations'] and fixture==(out/'production.log').read_text();print('PASS',r['cases'],mutations);return
 assert a.proof and a.reference_root and a.output
 proof=a.proof.resolve();ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 u=json.loads((proof/'receipt.json').read_text());assert H(proof/'receipt.json')=='1e39ead5c5388cff501366e0d0fa287db7938f8ac6ca3317302f56a6ade2beb9'
 sources=dict(u['sources'])
 extra=['tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h','tools/hip_vector_title.cpp','tools/native_runtime.h','tools/native_shell_exit_probe.cpp','tests/hip_multi_query_title_model.cpp','tests/hip_multi_query_title_test.py']
 extra += [str(pathlib.Path(m.__file__).resolve().relative_to(R)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(R)]
 for n in extra:sources[n]=H(R/n)
 assert all(H(R/n)==v for n,v in sources.items())
 inputs=u['inputs']|{str(proof/n):h for n,h in u['artifacts'].items()}|{str(proof/'receipt.json'):H(proof/'receipt.json')}
 elevation=ref/'.deps/elevation/client'
 for name in ('elevation.hpp','protocol.hpp'):inputs[str(elevation/name)]=H(elevation/name)
 assert all(H(pathlib.Path(n))==h for n,h in inputs.items())
 commands=[];env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=0')
 def run(label,argv,mode=None):
  e=dict(env)
  if mode:e['MULTI_MODE']=mode
  q=subprocess.run(list(map(str,argv)),capture_output=True,text=True,env=e,timeout=180)
  for suffix,text in (('.stdout',q.stdout),('.stderr',q.stderr)):(out/(label+suffix)).write_text(text)
  commands.append(dict(label=label,argv=list(map(str,argv)),mode=mode,exit_code=q.returncode));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  assert q.returncode==0,(label,q.stdout[-1000:],q.stderr[-1800:])
  assert not any(t in q.stdout+q.stderr for t in ('AddressSanitizer','runtime error:','MULTI_INVARIANT'))
 original=next(c['argv'] for c in json.loads((proof/'commands.json').read_text()) if c['label']=='ab-build')
 common=original[:next(i for i,x in enumerate(original) if x.endswith('/src/code_object.cpp'))]
 common=[x for x in common if x not in ('-pie','-fuse-ld=lld-18')]+['-DPS5HIP_MULTI_QUERY_TITLE=1','-I'+str(R/'tools'),'-I'+str(elevation)]
 core=[pathlib.Path(x) for x in original if x.endswith('.cpp') and '/src/' in x]
 files=core+[R/n for n in ('tools/hip_multi_query_client.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp','tests/hip_multi_query_title_model.cpp')];objects=[]
 for source in files:
  flags=[]
  if source.name=='hip_vector_title.cpp':flags=['-std=c++20','-Dmain=ps5hip_multi_title_main','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
  if source.name=='hip_multi_query_title_model.cpp':flags=['-std=c++20']
  if source.name=='native_shell_exit_probe.cpp':flags=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
  obj=out/(source.stem+'.o');run('compile-'+source.stem,[*common,*flags,'-c',source,'-o',obj]);objects.append(obj)
 binary=out/'title';run('title-link',[*common,'-pie','-fuse-ld=lld-18',*objects,proof/'linux/a.o',proof/'linux/b.o','-Wl,-T,'+str(proof/'multi.ld'),'-Wl,--wrap=__hipRegisterFatBinary,--wrap=__hipUnregisterFatBinary,--wrap=__hipRegisterFunction,--wrap=launch_a,--wrap=launch_b,--wrap=hipMemPtrGetInfo,--wrap=hipMemGetAddressRange,--wrap=ps5hip_runtime_finish','-o',binary])
 cases={}
 for mode in CASES:
  folder=out/'cases'/mode;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('invalid-relative' if mode=='config' else str(folder));log=folder/'ps5-hip-gfx1013-hip_multi_query.log'
  if mode=='log-exists':log.write_text('previous-result\n')
  run('run-'+mode,[binary,mode,config],mode)
  if log.exists():assert not any(t in log.read_text() for t in ('AddressSanitizer','runtime error:','MULTI_INVARIANT'))
  cases[mode]={'log':str(log.relative_to(out)) if log.exists() else None,'stdout':'run-'+mode+'.stdout'}
 (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');fixture,mutations=verify(out);(out/'production.log').write_text(fixture)
 assert all(H(R/n)==v for n,v in sources.items()) and all(H(pathlib.Path(n))==v for n,v in inputs.items())
 result=dict(classification='offline-two-module-title-lifecycle-pass',sources=sources,inputs=inputs,cases=len(cases),mutations=mutations,hardware_executed=False,artifacts={str(p.relative_to(out)):H(p) for p in out.rglob('*') if p.is_file()})
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(cases),mutations,H(out/'receipt.json'))
if __name__=='__main__':main()
