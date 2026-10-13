"""Strict saved-title admission, independently bound without rebuilding its model."""
import argparse,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MERGED='1e39ead5c5388cff501366e0d0fa287db7938f8ac6ca3317302f56a6ade2beb9'
EXTRA=('tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h','tools/hip_vector_title.cpp','tools/native_runtime.h','tools/native_shell_exit_probe.cpp','tests/hip_multi_query_title_model.cpp','tests/hip_multi_query_title_test.py','tests/hip_multi_query_client_test.py','tools/probe_ocml_math.py','tools/math_native_fixture.py','tools/check_canary_log.py')
def check(out):
 r=json.loads((out/'receipt.json').read_text());assert r['classification']=='offline-two-module-title-lifecycle-pass' and r['hardware_executed'] is False
 upstream=[pathlib.Path(n) for n,h in r['inputs'].items() if h==MERGED and n.endswith('/receipt.json')];assert len(upstream)==1 and H(upstream[0])==MERGED
 proof=upstream[0].parent;u=json.loads(upstream[0].read_text());expected_sources=set(u['sources'])|set(EXTRA)
 assert set(r['sources'])==expected_sources and all(H(R/n)==h for n,h in r['sources'].items())
 # Only import the producer's transcript verifier after binding its full closure.
 sys.path[:0]=[str(R/'tests'),str(R/'tools')]
 from hip_multi_query_title_test import CASES,verify
 wrappers=[pathlib.Path(n) for n in u['inputs'] if n.endswith('/.deps/sdk/bin/prospero-clang++')];assert len(wrappers)==1
 ref=wrappers[0].parents[3];elevation=ref/'.deps/elevation/client'
 expected_inputs=u['inputs']|{str(proof/n):h for n,h in u['artifacts'].items()}|{str(proof/'receipt.json'):MERGED}|{str(elevation/n):H(elevation/n) for n in ('elevation.hpp','protocol.hpp')}
 assert r['inputs']==expected_inputs and all(H(pathlib.Path(n))==h for n,h in r['inputs'].items())
 original=next(c['argv'] for c in json.loads((proof/'commands.json').read_text()) if c['label']=='ab-build')
 files=[pathlib.Path(n) for n in original if n.endswith('.cpp') and '/src/' in n]+[R/n for n in ('tools/hip_multi_query_client.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp','tests/hip_multi_query_title_model.cpp')]
 labels=['compile-'+p.stem for p in files]+['title-link']+['run-'+m for m in CASES]
 artifacts={'title','cases.json','commands.json','production.log',*[p.stem+'.o' for p in files],*[label+suffix for label in labels for suffix in ('.stdout','.stderr')],*['cases/'+m+'/results-dir.txt' for m in CASES],*['cases/'+m+'/ps5-hip-gfx1013-hip_multi_query.log' for m in CASES if m not in ('config','elevation')]}
 def inventories(value):
  assert set(value['sources'])==expected_sources and set(value['inputs'])==set(expected_inputs) and set(value['artifacts'])==artifacts
 inventories(r);assert all(H(out/n)==h for n,h in r['artifacts'].items())
 commands=json.loads((out/'commands.json').read_text());assert [c['label'] for c in commands]==labels and all(c['exit_code']==0 for c in commands)
 for c,p in zip(commands,files):
  argv=c['argv'];assert argv[argv.index('-c')+1]==str(p) and argv[argv.index('-o')+1]==str(out/(p.stem+'.o')) and '-fsanitize=address,undefined' in argv
 link=commands[len(files)]['argv'];assert all(str(out/(p.stem+'.o')) in link for p in files) and all(str(proof/'linux'/(m+'.o')) in link for m in 'ab')
 for c,m in zip(commands[len(files)+1:],CASES):assert c['mode']==m and c['argv']==[str(out/'title'),m,str(out/'cases'/m/'results-dir.txt')]
 cases=json.loads((out/'cases.json').read_text());assert set(cases)==set(CASES)
 for m,c in cases.items():
  assert c=={'log':None if m in ('config','elevation') else 'cases/'+m+'/ps5-hip-gfx1013-hip_multi_query.log','stdout':'run-'+m+'.stdout'}
 fixture,mutations=verify(out);assert r['cases']==len(CASES) and mutations==r['mutations'] and fixture==(out/'production.log').read_text()
 omissions=0
 for field in ('sources','inputs','artifacts'):
  for key in (None,*r[field]):
   bad=dict(r);bad[field]={} if key is None else {k:v for k,v in r[field].items() if k!=key}
   try:inventories(bad)
   except AssertionError:omissions+=1
   else:raise AssertionError('incomplete title inventory admitted')
 print('PASS',len(CASES),'title cases',mutations,'transcript mutations',omissions,'inventory omissions')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',type=pathlib.Path,required=True);a=p.parse_args();check(a.check.resolve())
