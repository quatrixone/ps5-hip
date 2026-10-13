"""Exact saved-evidence omission and oracle rejection tests."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import check_clr_adapter as checker
p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=checker.check(a.proof);a.output.mkdir(parents=True,exist_ok=False)
# Hash all actual bytes once; mutations below change receipt assertions only.
cache={str(Path(k).resolve()):v for k,v in r['inputs'].items()}
cache.update({str((a.proof.parent/k).resolve()):v for k,v in r['artifacts'].items()})
real=checker.H;checker.H=lambda p:cache.get(str(Path(p).resolve())) or real(p)
mutations=[]
for field in ('inputs','artifacts','commands','modes'):
 q=copy.deepcopy(r);q[field]={} if field in ('inputs','artifacts') else [];mutations.append((field+'-empty',q))
for name in ('normal.stdout','noop.stderr','adapter.o','clr-0.d','commands.json'):
 q=copy.deepcopy(r);del q['artifacts'][name];mutations.append(('omit-'+name,q))
q=copy.deepcopy(r);key=next(k for k in q['inputs'] if k.endswith('/tools/prepare_clr_adapter.py'));del q['inputs'][key];mutations.append(('omit-preparation-helper',q))
q=copy.deepcopy(r);q['commands'][0]['exit_code']=1;mutations.append(('compile-failure',q))
q=copy.deepcopy(r);next(c for c in q['commands'] if c['label']=='noop')['exit_code']=0;mutations.append(('oracle-did-not-reject',q))
q=copy.deepcopy(r);q['gpu_executed']=True;mutations.append(('false-gpu-claim',q))
for tool in (checker.tool(next(c for c in r['commands'] if c['label']=='adapter')['argv'][0]),checker.tool('ld.lld-18')):
 q=copy.deepcopy(r);del q['inputs'][tool];mutations.append(('omit-tool-'+Path(tool).name,q))
for label in ('adapter','link'):
 q=copy.deepcopy(r);c=next(c for c in q['commands'] if c['label']==label);c['argv']=[v for v in c['argv'] if not v.startswith('-fsanitize=')];mutations.append(('uninstrumented-'+label,q))
q=copy.deepcopy(r);q['sanitizer']='claimed';mutations.append(('unknown-sanitizer',q))
q=copy.deepcopy(r);next(c for c in q['commands'] if c['label']=='adapter')['argv'].append('-fno-sanitize=all');mutations.append(('disabled-instrumentation',q))
for name,q in mutations:
 try:checker.check(a.proof,q)
 except (AssertionError,KeyError,ValueError):pass
 else:raise AssertionError('accepted '+name)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'classification':'strict-clr-adapter-saved-checker-pass','proof_sha256':h(a.proof),'checker_sha256':h(checker.__file__),'test_sha256':h(__file__),'mutations':[n for n,_ in mutations],'passed':len(mutations),'gpu_executed':False}
(a.output/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(h(a.output/'receipt.json'))
