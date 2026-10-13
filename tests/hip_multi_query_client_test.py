"""Genuine two-object client qualification against the merged runtime transport model."""
import argparse,hashlib,json,os,pathlib,re,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MODES=('normal','skip-b','skip-final-a','wrong-owner','source-corrupt','output-corrupt','guard-corrupt','query-size','query-base','info-sentinel','range-sentinel','alloc-first','alloc-second','submit','cleanup','free-first')
def validate(text,mode):
 assert not any(x in text for x in ('AddressSanitizer','runtime error:','MULTI_INVARIANT'))
 assert text.count('MULTI_QUERY_MODEL_PASS\n')==1
 count=3 if mode in ('normal','free-first') else 2 if mode=='skip-final-a' else 1 if mode in ('skip-b','wrong-owner','source-corrupt','output-corrupt','guard-corrupt','submit','cleanup') else 0
 live=6 if mode in ('submit','wrong-owner') else 3 if mode=='cleanup' else 2 if mode=='free-first' else 0
 releases=0 if mode in ('submit','wrong-owner','alloc-first') else 1 if mode=='alloc-second' else 3 if mode=='cleanup' else 12 if mode=='free-first' else 2+4*count
 assert re.findall(r'^MULTI_QUERY_FINAL .*$',text,re.M)==[f'MULTI_QUERY_FINAL callbacks=2 submits={count} live={live} releases={releases} finalized={int(mode=="normal")}']
 if mode=='normal':
  assert text.count('MULTI_QUERY QUERIES success=32 rejected=8 logical_bytes=348\n')==1
  assert text.count('MULTI_QUERY CLIENT_PASS dispatches=3 exact_values=219 comparisons=522 guards=84 allocated=2 freed=2\n')==1
  assert re.findall(r'^MULTI_QUERY STEP .*$',text,re.M)==[f'MULTI_QUERY STEP index={i} module={m} count=73 exact=true source=true guards=true' for i,m in enumerate('ABA')]
  assert re.findall(r'^VECTOR_FINAL .*$',text,re.M)==['VECTOR_FINAL registered=2 removed=2 completed=3 active=0 terminal=0 main_finished=1 hardware_validated=false']
  assert 'RUNTIME_RETAINED' not in text
 else:
  assert 'CLIENT_PASS' not in text and text.count('MULTI_QUERY CLIENT_FAIL ')==1
  assert 'VECTOR_FINAL' not in text and text.count('RUNTIME_RETAINED ')==1

def check(out):
 r=json.loads((out/'receipt.json').read_text());assert r['classification']=='offline-multi-query-client-pass'
 expected={*r['upstream_sources'],'tests/hip_multi_recipe_test.py','tests/hip_multi_query_client_test.py','tests/hip_multi_query_model.cpp','tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h'}
 assert set(r['sources'])==expected and r['upstream_sources']
 assert all(H(R/n)==v for n,v in r['sources'].items())
 upstream=pathlib.Path(r['upstream']['path']);assert H(upstream/'receipt.json')==r['upstream']['sha256']
 u=json.loads((upstream/'receipt.json').read_text());assert r['upstream_sources']==u['sources']
 assert r['inputs']==u['inputs']|{str(upstream/n):h for n,h in u['artifacts'].items()}|{str(upstream/'receipt.json'):H(upstream/'receipt.json')}
 assert all(H(pathlib.Path(n))==v for n,v in r['inputs'].items())
 labels=[o+'-build' for o in ('ab','ba')]+[o+'-'+m for o in ('ab','ba') for m in MODES]
 expected_artifacts={'model-ab','model-ba','commands.json',*[n+s for n in labels for s in ('.stdout','.stderr')]}
 assert set(r['artifacts'])==expected_artifacts and all(H(out/n)==v for n,v in r['artifacts'].items())
 commands=json.loads((out/'commands.json').read_text());assert {c['label'] for c in commands}==set(labels) and all(c['returncode']==0 for c in commands)
 for order in ('ab','ba'):
  for mode in MODES:
   text=(out/(order+'-'+mode+'.stdout')).read_text()+(out/(order+'-'+mode+'.stderr')).read_text();validate(text,mode)
   final=re.search(r'^MULTI_QUERY_FINAL .*$',text,re.M)[0]
   for bad in (text.replace(final,''),text+final+'\n',text.replace('releases=','releases=99')):
    try:validate(bad,mode)
    except AssertionError:pass
    else:raise AssertionError('ownership transcript mutation admitted')
 print('PASS 32 client models and 96 ownership transcript mutations')

def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);p.add_argument('--check',type=pathlib.Path);a=p.parse_args()
 if a.check:return check(a.check.resolve())
 proof=a.proof.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 u=json.loads((proof/'receipt.json').read_text())
 names=set(u['sources'])|{'tests/hip_multi_recipe_test.py','tests/hip_multi_query_client_test.py','tests/hip_multi_query_model.cpp','tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h'}
 before={n:H(R/n) for n in names};assert all(before[n]==h for n,h in u['sources'].items())
 inputs=u['inputs']|{str(proof/n):h for n,h in u['artifacts'].items()}|{str(proof/'receipt.json'):H(proof/'receipt.json')}
 assert all(H(pathlib.Path(n))==v for n,v in inputs.items())
 env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=0',PYTHONPATH=str(R/'tools')+':'+str(R/'tests'))
 subprocess.run([sys.executable,str(R/'tests/hip_multi_recipe_test.py'),'--check',str(proof)],env=env,check=True)
 commands=[]
 def run(label,argv,mode=None):
  e=dict(env)
  if mode:e['MULTI_MODE']=mode
  q=subprocess.run(list(map(str,argv)),env=e,capture_output=True,text=True,timeout=180)
  for suffix,data in (('.stdout',q.stdout),('.stderr',q.stderr)):(out/(label+suffix)).write_text(data)
  commands.append(dict(label=label,argv=list(map(str,argv)),returncode=q.returncode,mode=mode));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  assert q.returncode==0,(label,q.stdout[-1500:],q.stderr[-1500:])
  if mode:validate(q.stdout+q.stderr,mode)
 for order in ('ab','ba'):
  argv=next(c['argv'] for c in json.loads((proof/'commands.json').read_text()) if c['label']==order+'-build')[:]
  argv=[str(R/'tests/hip_multi_query_model.cpp') if x==str(R/'tests/hip_multi_model.cpp') else x for x in argv]
  argv[argv.index('-o')+1]=str(out/('model-'+order))
  argv += ['-I'+str(R/'tools'),str(R/'tools/hip_multi_query_client.cpp'),'-Wl,--wrap=launch_a,--wrap=launch_b,--wrap=hipMemPtrGetInfo,--wrap=hipMemGetAddressRange']
  run(order+'-build',argv)
  for mode in MODES:run(order+'-'+mode,[out/('model-'+order)],mode)
 assert before=={n:H(R/n) for n in names} and all(H(pathlib.Path(n))==v for n,v in inputs.items())
 r=dict(classification='offline-multi-query-client-pass',sources=before,upstream_sources=u['sources'],upstream={'path':str(proof),'sha256':H(proof/'receipt.json')},inputs=inputs,artifacts={str(f.relative_to(out)):H(f) for f in out.iterdir() if f.is_file()},hardware_executed=False)
 (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');check(out)
if __name__=='__main__':main()
