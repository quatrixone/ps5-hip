#!/usr/bin/env python3
"""Real worker client/runtime with shared packet transport and client-only faults."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from probe_hip_generic import CORE
from probe_hip_routes import prepared_inputs
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
MODES=('normal','stack','create-first','create-second','join','unlock','wait','wait-unknown','broadcast','free','destroy','guard','template','submit','kick','timeout','throw','unknown','postflush','cleanup')
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def validate(text,mode):
 require('WORKERS_MODEL_PASS mode='+mode+' ' in text,'missing case')
 require(not any(s in text for s in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','assertion failed')),'diagnostic')
 if mode=='normal':require('WORKERS_CLIENT_PASS created=2 joined=2 dispatches=5 exact_values=661 word_comparisons=1355 template_comparisons=1355 phase_checks=17\n' in text and 'THREAD_MODEL_FINAL finalized=1 live=0' in text,'normal closure')
 else:require('WORKERS_CLIENT_PASS' not in text and 'WORKERS_CLIENT_FAIL ' in text and 'THREAD_MODEL_FINAL finalized=0 ' in text and 'VECTOR_FINAL ' not in text,'unsafe failure closure')
def verify(out):
 r=json.loads((out/'receipt.json').read_text());require(r['classification']=='joined-worker-client-host-model-pass','scope')
 for n,h in r['sources'].items():require(sha(ROOT/n)==h,'source drift '+n)
 for n,h in r['inputs'].items():require(sha(n)==h,'input drift '+n)
 for n,h in r['artifacts'].items():require(sha(out/n)==h,'artifact drift '+n)
 for m in MODES:validate((out/(m+'.stdout')).read_text()+(out/(m+'.stderr')).read_text(),m)
 print('Checked',len(MODES),'actual worker client models')
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path);p.add_argument('--compiler-proof',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);p.add_argument('--check',type=pathlib.Path);a=p.parse_args()
 if a.check:verify(a.check.resolve());return
 require(a.reference_root and a.compiler_proof and a.output,'paths required');ref=a.reference_root.resolve();proof=a.compiler_proof.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 prior=json.loads((proof/'receipt.json').read_text());require(prior['classification']=='checked-threaded-runtime-host-model-pass' and prior['sanitizer']=='address,undefined','compiler authority scope')
 inputs={str(proof/'receipt.json'):sha(proof/'receipt.json')}
 for n in ('threads.o','linux.ld'):
  require(sha(proof/n)==prior['artifacts'][n],'compiler artifact');inputs[str(proof/n)]=sha(proof/n)
 require(sha(ROOT/'tests/hip_threads.hip')==prior['sources']['tests/hip_threads.hip'],'kernel changed')
 names=[*['src/'+n for n in CORE],'src/hip_launch_profile.cpp','tests/hip_threads.hip','tests/hip_threads_model.cpp','tests/hip_workers_client_model.cpp','tests/hip_workers_client_test.py','tools/hip_workers_client.cpp','tools/hip_workers_client.h','dependencies.json','tools/probe_hip_generic.py','tools/probe_hip_routes.py','tools/probe_host_registration.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py','tests/native_import_libraries_test.py','tests/native_agc_imports_test.py',*[str(f.relative_to(ROOT)) for f in (ROOT/'include/ps5hip').glob('*.h')]]
 before={n:sha(ROOT/n) for n in names};commands=[]
 def run(label,argv):
  q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,timeout=35,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=0'))
  (out/(label+'.stdout')).write_bytes(q.stdout);(out/(label+'.stderr')).write_bytes(q.stderr);commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(q.returncode==0,label+': '+q.stderr.decode(errors='replace')[-3000:]);return q.stdout.decode()+q.stderr.decode()
 lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
 _,stage,_,prepared=prepared_inputs(ref);inputs.update(prepared);headers=stage/'include'
 for directory in (platform,headers):inputs.update({str(f):sha(f) for f in directory.rglob('*') if f.is_file()})
 for tool in ('clang++-18','ld.lld-18'):
  path=shutil.which(tool);require(path,'missing '+tool);inputs[path]=sha(path)
 common=['clang++-18','-std=c++17','-pthread','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_RUNTIME_MODEL=1','-I',ROOT/'include','-I',headers,'-I',platform]
 try:
  run('runtime-compile',[*common,'-DPS5HIP_THREAD_MODEL=1','-Dpthread_mutex_lock=thread_mutex_lock','-Dpthread_mutex_unlock=thread_mutex_unlock','-Dpthread_key_create=thread_key_create','-Dpthread_setspecific=thread_setspecific','-c',ROOT/'src/hip_runtime.cpp','-o',out/'runtime.o'])
  wrappers=('pthread_create','pthread_join','pthread_mutex_unlock','pthread_cond_timedwait','pthread_cond_broadcast','pthread_mutex_destroy','pthread_cond_destroy','pthread_attr_setstacksize','hipMemcpy','hipFree','threads_launch')
  run('client-compile',[*common,*['-D'+n+'=client_'+n for n in wrappers],'-c',ROOT/'tools/hip_workers_client.cpp','-o',out/'client.o'])
  run('model-link',[*common,'-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE if n!='hip_runtime.cpp'],ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/hip_workers_client_model.cpp',out/'runtime.o',out/'client.o',proof/'threads.o','-Wl,-T,'+str(proof/'linux.ld'),'-Wl,--wrap=__hipUnregisterFatBinary','-o',out/'model'])
  for mode in MODES:validate(run(mode,[out/'model',mode]),mode);print(mode+' PASS',flush=True)
  require(all(sha(ROOT/n)==h for n,h in before.items()),'source drift');require(all(sha(n)==h for n,h in inputs.items()),'input drift')
  r={'schema':1,'classification':'joined-worker-client-host-model-pass','sources':before,'inputs':inputs,'modes':list(MODES),'commands':commands,'hardware_executed':False,'limits':['Synthetic transport, actual runtime/compiler/client/pthreads; no native GPU affinity proof.','Join-failure model joins the real worker before reporting error; no stuck-thread termination claim.','Fixed shader mappings require ASan protect_shadow_gap=0; intentional retention disables leak checking.','Installed compiler resource/system C++ support trees are not fully inventoried; no hermetic build claim.'],'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
  (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))
 except BaseException as e:
  (out/'failure.json').write_text(json.dumps({'error':repr(e),'sources_before':before,'commands':commands},indent=2)+'\n');raise
if __name__=='__main__':main()
