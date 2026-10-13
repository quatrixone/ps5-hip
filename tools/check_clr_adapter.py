"""Replay saved adapter evidence without executing GPU or model code."""
if not __debug__: raise RuntimeError("optimized Python disables required evidence assertions")
from pathlib import Path
import argparse,hashlib,json,shlex,shutil
MODES=['normal','callback','badbinary','truncated','unsupported','destroy','submit','throw','kick','timeout','postflush','cleanup','noop','corrupt']
UNITS=['clr-'+str(i) for i in range(28)]+['adapter','model']+['core-'+x for x in ('code_object','fat_binary','host_launch','launch','native_memory','native_data','native_launch','submission','hip_launch_profile')]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tool=lambda name:str(Path(shutil.which(name) or name).resolve())
def check(path,report=None):
 path=Path(path).resolve();O=path.parent;r=json.loads(path.read_text()) if report is None else report
 assert r['classification']=='source-bound-real-clr-ps5-adapter-model'
 assert r['modes']==MODES and r['source_units']==39 and not r['gpu_executed'] and not r['native_linked']
 assert r['sanitizer'] in ('address','thread','none')
 assert r['inputs'] and r['artifacts']
 for name,digest in r['inputs'].items():assert H(name)==digest,name
 for name,digest in r['artifacts'].items():assert H(O/name)==digest,name
 commands=r['commands'];by={c['label']:c for c in commands}
 labels={x for u in UNITS for x in (u,u+'-headers')}|{'link'}|set(MODES)
 assert len(commands)==len(labels) and set(by)==labels
 compiler=tool(by['adapter']['argv'][0]);linker=tool('ld.lld-18')
 assert compiler in r['inputs'] and linker in r['inputs']
 sanitizer=[] if r['sanitizer']=='none' else ['-fsanitize='+('address,undefined' if r['sanitizer']=='address' else 'thread')]
 for label in labels-set(MODES):
  args=by[label]['argv'];assert tool(args[0])==compiler
  assert [v for v in args if v.startswith('-fsanitize=')]==sanitizer
  assert not any(v.startswith(('-fno-sanitize=', '-fsanitize-ignorelist=', '-fsanitize-blacklist=')) for v in args)
  if sanitizer:assert '-fno-omit-frame-pointer' in args
 assert '-fuse-ld=lld-18' in by['link']['argv'] and '-Wl,-z,defs' in by['link']['argv']
 assert json.loads((O/'commands.json').read_text())==commands
 required={'commands.json','model'}|{u+ext for u in UNITS for ext in ('.o','.d')}|{label+ext for label in labels for ext in ('.stdout','.stderr')}
 assert set(r['artifacts'])==required
 for u in UNITS:
  c=by[u];assert c['exit_code']==0 and c['expected_exit']==0
  source=c['argv'][c['argv'].index('-c')+1];assert source in r['inputs']
  assert c['argv'][-2:]==['-o',str(O/(u+'.o'))]
  assert by[u+'-headers']['exit_code']==0
  for dep in shlex.split((O/(u+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]):assert str(Path(dep).resolve()) in r['inputs'],dep
 assert by['link']['exit_code']==0
 R=Path(by['adapter']['argv'][by['adapter']['argv'].index('-c')+1]).parents[2]
 for name in ('tools/qualify_clr_adapter.py','tools/prepare_clr_adapter.py','tools/patch_clr_lifetime.py','tools/patch_clr_lifetime_v6.py','tools/patch_clr_lifetime_v7.py','dependencies.json'):
  assert str(R/name) in r['inputs'],name
 for u in UNITS:assert str(O/(u+'.o')) in by['link']['argv']
 for mode in MODES:
  c=by[mode];out=(O/(mode+'.stdout')).read_text();err=(O/(mode+'.stderr')).read_text()
  assert c['argv'][0]==str(O/'model') and c['argv'][-1]==mode
  assert H(c['argv'][1])==r['genuine_code_object_sha256']
  assert not any(x in err for x in ('runtime error:','ERROR: AddressSanitizer','WARNING: ThreadSanitizer','LeakSanitizer:'))
  if mode in ('noop','corrupt'):
   assert c['exit_code']<0 and 'CLR_OUTPUT_ORACLE_REJECT mode='+mode+' index=0' in err
   assert 'CLR_ADAPTER_MODEL assertion' not in err and not out
  else:
   assert c['exit_code']==0 and 'CLR_PS5_ADAPTER_MODEL_PASS mode='+mode+' gpu_execution=false' in out
   if mode not in ('badbinary','truncated','unsupported'):
    if mode in ('normal','callback','destroy'):
     n=3 if mode=='callback' else 2
     assert f'allocations={n} frees={n} retained=0 releases={n+4} uncertain=0' in out
    else:
     tail='retained=3 releases=3 uncertain=0' if mode=='cleanup' else 'retained=6 releases=0 uncertain=1'
     assert 'allocations=2 frees=0 '+tail in out
  if r['sanitizer']=='address':
   leak='1' if mode in ('normal','callback','badbinary','truncated','unsupported') else '0'
   assert c['sanitizer_environment']['ASAN_OPTIONS']=='protect_shadow_gap=0:detect_leaks='+leak
   assert c['sanitizer_environment']['UBSAN_OPTIONS']=='halt_on_error=1:print_stacktrace=1'
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);a=p.parse_args();r=check(a.proof)
 print('CLR_ADAPTER_SAVED_PASS modes=14 units=39 sanitizer='+r['sanitizer'])
