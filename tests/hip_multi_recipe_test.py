"""Replay exact source-bound two-module model logs and fail-closed mutations."""
import argparse,pathlib,json,hashlib,re,sys
R=pathlib.Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MODES=('normal','wrong-unregister','duplicate-unregister','skip-unregister','no-finish','live-finish','bad-wrapper','duplicate-token','missing-function','late-after-error','submit','cleanup')
def validate(text,mode):
 assert not any(x in text for x in ('AddressSanitizer','runtime error:','MULTI_INVARIANT'))
 early=mode in ('bad-wrapper','duplicate-token','missing-function','late-after-error')
 terminal=mode in ('submit','cleanup');clean=mode in ('normal','wrong-unregister','duplicate-unregister')
 cb=1 if mode=='bad-wrapper' else 2;sub=0 if early else 1 if terminal else 3
 live=6 if mode=='submit' else 3 if mode=='cleanup' else 2 if mode=='live-finish' else 0
 releases=0 if early or mode=='submit' else 3 if mode=='cleanup' else 12 if mode=='live-finish' else 14
 expected=f'MULTI_FINAL modules=2 callbacks={cb} submits={sub} live={live} releases={releases} finalized={int(clean)}'
 assert re.findall(r'^MULTI_FINAL .*$',text,re.M)==[expected]
 complete=re.findall(r'^RUNTIME_COMPLETE sequence=(\d+) kernel=(\w+) blocking=true$',text,re.M)
 assert complete==([] if early or terminal else [('1','multi_a'),('2','multi_b'),('3','multi_a')])
 if clean:
  assert re.findall(r'^VECTOR_FINAL .*$',text,re.M)==['VECTOR_FINAL registered=2 removed=2 completed=3 active=0 terminal=0 main_finished=1 hardware_validated=false']
  assert 'RUNTIME_RETAINED' not in text
 else:
  assert 'VECTOR_FINAL' not in text and len(re.findall(r'^RUNTIME_RETAINED ',text,re.M))==1
 if not early and not terminal:assert text.count('MULTI_PASS launches=3 sequence=A/B/A guarded=true source_preserved=true owning_code=true\n')==1
 else:assert 'MULTI_PASS' not in text
def check(proof):
 r=json.loads((proof/'receipt.json').read_text());assert r['classification']=='offline-two-module-runtime-model-pass' and r['hardware_validated'] is False
 assert r['models']==[dict(order=o,mode=m) for o in ('ab','ba') for m in MODES]
 core=('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp','hip_runtime.cpp','hip_launch_profile.cpp')
 expected_sources={'dependencies.json',*[('src/'+n) for n in core],*[str(p.relative_to(R)) for p in (R/'include/ps5hip').glob('*.h')],*[('tests/'+n) for n in ('hip_multi_a.hip','hip_multi_b.hip','hip_multi_model.cpp','host_multi_registration_test.cpp','fat_binary_fixture_test.py')],*[('tools/'+n) for n in ('probe_hip_multi.py','probe_hip_generic.py','probe_hip_routes.py','build_canaries.py')]}
 assert set(r['sources'])==expected_sources
 assert all(H(R/k)==v for k,v in r['sources'].items())
 # Imported dependency checks are themselves bound by the verified source map.
 sys.path.insert(0,str(R/'tools'))
 from probe_hip_routes import prepared_inputs
 wrappers=[pathlib.Path(k) for k in r['inputs'] if k.endswith('/.deps/sdk/bin/prospero-clang++')];assert len(wrappers)==1
 ref=wrappers[0].parents[3];_,_,_,expected_inputs=prepared_inputs(ref)
 for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-config'):expected_inputs['/usr/bin/'+n+'-18']=H(pathlib.Path('/usr/bin/'+n+'-18'))
 for n in ('prospero-clang++','clang++','prospero-llvm-config','prospero-clang','clang'):
  p=ref/'.deps/sdk/bin'/n
  if p.exists():expected_inputs[str(p)]=H(p)
 lock=json.loads((R/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
 for p in platform.rglob('*'):
  if p.is_file():expected_inputs[str(p)]=H(p)
 labels=[n+'-version' for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-config')]
 labels += [h+a+'-'+action for h in ('linux','ps5') for a in ('a','b') for action in ('driver','compile','readobj','disasm')]
 labels += [o+'-build' for o in ('ab','ba')]+[o+'-'+m for o in ('ab','ba') for m in MODES]+['registry-build','registry-test']
 expected_artifacts={*[label+suffix for label in labels for suffix in ('.stdout','.stderr')],*[h+'/'+a+suffix for h in ('linux','ps5') for a in ('a','b') for suffix in ('.o','.bundle')],'commands.json','multi.ld','model-ab','model-ba','registry'}
 def inventories(value):
  assert set(value['sources'])==expected_sources and set(value['inputs'])==set(expected_inputs) and set(value['artifacts'])==expected_artifacts
 inventories(r);assert r['inputs']==expected_inputs
 assert all(H(R/k)==v for k,v in r['sources'].items());assert all(H(pathlib.Path(k))==v for k,v in r['inputs'].items());assert all(H(proof/k)==v for k,v in r['artifacts'].items())
 omissions=0
 for field in ('sources','inputs','artifacts'):
  for key in (None,*r[field]):
   bad=dict(r);bad[field]={} if key is None else {k:v for k,v in r[field].items() if k!=key}
   try:inventories(bad)
   except AssertionError:omissions+=1
   else:raise AssertionError('incomplete inventory accepted')
 count=0
 for o in ('ab','ba'):
  for m in MODES:
   text=(proof/(o+'-'+m+'.stdout')).read_text()+(proof/(o+'-'+m+'.stderr')).read_text();validate(text,m)
   for bad in (re.sub(r'^MULTI_FINAL .*\n','',text,flags=re.M),text+re.search(r'^MULTI_FINAL .*$',text,re.M)[0]+'\n',text.replace('callbacks=','callbacks=9')):
    try:validate(bad,m)
    except AssertionError:count+=1
    else:raise AssertionError('mutated transcript accepted')
 assert 'ALLOCATION_FAILURES transactional=145 function_rollback=true' in (proof/'registry-test.stdout').read_text()
 print(f'PASS 24 exact model logs, {count} transcript mutations and {omissions} inventory omissions; all hashes match')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',type=pathlib.Path,required=True);a=p.parse_args();check(a.check.resolve())
