#!/usr/bin/env python3
"""Recheck a saved reduction title proof without rebuilding or console access."""
import pathlib,argparse,json,hashlib,sys,shlex
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from check_canary_log import check_log
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--check',type=pathlib.Path,required=True);a=p.parse_args();out=a.check.resolve();r=json.loads((out/'receipt.json').read_text());assert r['classification']=='reduction-title-host-lifecycle-pass' and r['hardware_executed'] is False and r['native_linked'] is False
core=('code_object','fat_binary','host_registration','host_launch','launch','native_memory','native_data','native_launch','submission','hip_runtime','hip_launch_profile')
compiled=[*('src/'+n+'.cpp' for n in core),'tests/reduction_title_model.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp']
required_sources=set(compiled)|{'tests/reduction_title_test.py','tools/check_canary_log.py','tools/math_native_fixture.py','tools/native_runtime.h','tests/reduction_companion_model.cpp'}
modes=('normal','shell-fail','skip-unregister','elevation','config','log-exists','startup-sync','final-sync','corrupt','missing','stale','highbits','tail','input-guard','mask-guard','output-guard','submit','code-image','entry')
labels=[label for i,n in enumerate(compiled) for label in ('dependencies-'+str(i),'compile-'+pathlib.Path(n).stem)]+['link']+['run-'+m for m in modes]
artifacts={'commands.json','production.log','title'}|{'headers-'+str(i)+'.d' for i in range(len(compiled))}|{pathlib.Path(n).stem+'.o' for n in compiled}|{label+suffix for label in labels for suffix in ('.stdout','.stderr')}
artifacts|={'cases/'+m+'/results-dir.txt' for m in modes}|{'cases/'+m+'/ps5-hip-gfx1013-hip_reduction.log' for m in modes if m not in ('config','elevation')}
assert set(r['sources'])==required_sources and set(r['artifacts'])==artifacts and set(r['tools'])=={'clang++','ld.lld'}
assert [c['label'] for c in r['commands']]==labels and all(c['exit']==0 for c in r['commands'])
assert json.loads((out/'commands.json').read_text())==r['commands']
upstream=ROOT/'results/companion-runtime-v2';upstream_receipt=upstream/'receipt.json'
assert sha(upstream_receipt)=='7c5f0b58baafb43f3e9e414eac8f7a89d3b07044a24dba13487a61f7acfb06fd'
u=json.loads(upstream_receipt.read_text());required_inputs=dict(u['inputs'])
required_inputs.update({str(ROOT/n):h for n,h in u['sources'].items()});required_inputs.update({str(upstream/n):h for n,h in u['artifacts'].items()});required_inputs[str(upstream_receipt)]=sha(upstream_receipt)
for i in range(len(compiled)):
 words=shlex.split((out/('headers-'+str(i)+'.d')).read_text().replace('\\\n',' '));assert words[0].endswith(':')
 for word in words[1:]:
  hp=pathlib.Path(word);hp=hp if hp.is_absolute() else ROOT/hp;required_inputs[str(hp.resolve())]=sha(hp)
assert r['inputs']==required_inputs
for n,h in r['sources'].items():assert sha(ROOT/n)==h,n
for n,h in r['inputs'].items():assert sha(pathlib.Path(n))==h,n
for n,h in r['artifacts'].items():assert sha(out/n)==h,n
for t in r['tools'].values():assert sha(pathlib.Path(t['path']))==t['sha256']
expected={'normal','shell-fail','skip-unregister','elevation','config','log-exists','startup-sync','final-sync','corrupt','missing','stale','highbits','tail','input-guard','mask-guard','output-guard','submit','code-image','entry'};assert set(r['cases'])==expected;assert {c['label'][4:] for c in r['commands'] if c['label'].startswith('run-')}==expected;assert all(c['exit']==0 for c in r['commands'])
for mode,record in r['cases'].items():
 assert record['log']==('run-'+mode+'.stdout' if mode in ('config','elevation') else 'cases/'+mode+'/ps5-hip-gfx1013-hip_reduction.log')
 text=(out/record['log']).read_text();assert sha(out/record['log'])==record['sha256'];assert not any(x in text for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer'))
 if mode in ('normal','shell-fail'):assert 'REDUCTION_TITLE_MODEL STOP shell_requests=1 live=0 releases=21 submits=3 model_only=true' in text
 else:
  assert 'PS5HIP SHELL_EXIT request=begin' not in text
  if mode not in ('config','elevation','log-exists'):assert 'HIP_TITLE QUARANTINED' in text
 if mode in ('submit','code-image','entry'):assert 'REDUCTION_TITLE_MODEL STOP shell_requests=0 live=7 releases=0 submits=1 model_only=true' in text
 if mode=='skip-unregister':assert 'reason=finalization' in text and 'HIP_TITLE FINAL ' not in text
production=(out/'production.log').read_text();assert check_log(production.encode(),'gfx1013','hip_reduction',shell_exit_probe=True)['application_result']=='application-pass';lines=production.splitlines();mutations=[]
for i in range(len(lines)):mutations+=['\n'.join(lines[:i]+lines[i+1:])+'\n','\n'.join(lines[:i]+[lines[i]]+lines[i:])+'\n']
for old,new in [('dispatches=3','dispatches=2'),('logical=16377','logical=16384'),('exact_values=48','exact_values=47'),('guards=72','guards=71'),('functions=1','functions=2'),('input_unchanged=true','input_unchanged=false'),('masks_unchanged=true','masks_unchanged=false')]:mutations.append(production.replace(old,new))
assert len(mutations)==r['transcript_mutations']
for bad in mutations:assert check_log(bad.encode(),'gfx1013','hip_reduction',shell_exit_probe=True)['application_result']!='application-pass'
print(f'PASS {len(expected)} title cases and {len(mutations)} transcript mutations; CPU model only')
