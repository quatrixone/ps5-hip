#!/usr/bin/env python3
"""Strict saved compiler authority and omission rejection for reduction."""
if not __debug__: raise RuntimeError('optimized Python is forbidden')
import argparse,copy,hashlib,json,pathlib,sys,shlex
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from probe_hip_routes import audit_object
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCE_NAMES={'tests/upstream-reduction/warp_size_reduction.hip','tests/upstream-reduction/popcount.hpp','tests/upstream-reduction/source.json','tools/probe_hip_reduction.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py'}
def check(out,r,rehash=True):
 assert r['classification']=='bound-upstream-reduction-compiler-proof' and r['runtime_or_hardware_executed'] is False
 assert r['upstream_source']['sha256']=='42ffe47b3d8dfdd59d2c8ebaabcf7ae6451c97f1055fe9ca6fafebdcc274a57a'
 assert set(r['sources'])=={str(ROOT/n) for n in SOURCE_NAMES}
 assert set(r['tools'])=={'clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm','llvm-config'}
 assert len(r['inputs'])>100 and r['artifacts']
 sdk=pathlib.Path(r['sdk_root']);assert sdk.is_absolute()
 wrappers={str(sdk/n) for n in ('bin/prospero-clang++','bin/clang++','bin/prospero-llvm-config','.ps5-sdk-revision')}
 assert set(r['wrapper_inputs'])==wrappers and wrappers<=set(r['inputs'])
 assert all(r['wrapper_inputs'][n]==r['inputs'][n] for n in wrappers)
 assert r['sdk_revision']==json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision']
 required={f'{host}/{name}' for host in ('linux','ps5') for name in ('reduction.o','reduction.bundle','reduction.hsaco','observed.json')}
 required|={f'{h}-{s}.d' for h in ('linux','ps5') for s in ('host','device')}
 required|={f'{h}-{s}.{stream}' for h in ('linux','ps5') for s in ('driver','host-dependencies','device-dependencies','compile','metadata','isa','symbols','host','device-symbols') for stream in ('stdout','stderr')}
 assert required<=set(r['artifacts'])
 labels={f'{h}-{s}' for h in ('linux','ps5') for s in ('driver','host-dependencies','device-dependencies','compile','metadata','isa','symbols','host','device-symbols')}
 assert {x['label'] for x in r['commands']}==labels and len(r['commands'])==len(labels)
 assert all(x['exit_code']==0 and x['environment']['LLVM_CONFIG']==r['tools']['llvm-config']['path'] for x in r['commands'])
 required_inputs=set()
 for host in ('linux','ps5'):
  for side in ('host','device'):
   words=shlex.split((out/(host+'-'+side+'.d')).read_text().replace('\\\n',' '));assert words[0].endswith(':')
   for word in words[1:]:
    hp=pathlib.Path(word);required_inputs.add(str((hp if hp.is_absolute() else ROOT/hp).resolve()))
 assert required_inputs<=set(r['inputs']), 'missing consumed header'
 for c in r['commands']:
  if c['label'].endswith('-compile'):
   argv=c['argv'];libpath=next(x.split('=',1)[1] for x in argv if x.startswith('--rocm-device-lib-path='))
   for x in argv:
    if x.startswith('--hip-device-lib='):assert str(pathlib.Path(libpath)/x.split('=',1)[1]) in r['inputs']
 # Negative receipt-omission checks reuse this same structural admission, not costly unchanged binary reads.
 if not rehash:return
 for n,h in r['sources'].items():assert sha(pathlib.Path(n))==h,n
 for v in r['tools'].values():assert sha(pathlib.Path(v['path']))==v['sha256']
 for n,h in r['inputs'].items():assert sha(pathlib.Path(n))==h,n
 assert (sdk/'.ps5-sdk-revision').read_text().strip()==r['sdk_revision']
 for n,h in r['artifacts'].items():
  p=(out/n).resolve();assert p.is_relative_to(out.resolve()) and sha(p)==h,n
 for host in ('linux','ps5'):
  assert 'clang version 18.1.8' in (out/(host+'-driver.stderr')).read_text()
  a=audit_object((out/host/'reduction.hsaco').read_bytes(),(out/(host+'-metadata.stdout')).read_text(),(out/(host+'-isa.stdout')).read_text(),('_Z12block_reducePiPmS_m',))
  assert a==json.loads((out/host/'observed.json').read_text())
  d=a['descriptors']['_Z12block_reducePiPmS_m'];assert d=={'group_bytes':0,'private_bytes':0,'kernarg_bytes':288,'rsrc1':0x60af0080,'rsrc2':0x8c,'rsrc3':0,'properties':0x409,'preload':0}
  ops=a['isa']['_Z12block_reducePiPmS_m']['opcodes'];assert {'s_barrier','ds_bpermute_b32','ds_read_b32','ds_write_b32'}<=set(ops)
  assert not (out/(host+'-device-symbols.stdout')).read_text().strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--compiler',type=pathlib.Path,required=True);a=p.parse_args();out=a.compiler.resolve();r=json.loads((out/'receipt.json').read_text());check(out,r);cases=[]
 for field in ('sources','tools','inputs','artifacts','commands','wrapper_inputs'):
  bad=copy.deepcopy(r);bad[field]={} if field!='commands' else [];cases.append(bad)
 for n in r['wrapper_inputs']:
  bad=copy.deepcopy(r);del bad['inputs'][n];cases.append(bad)
 bad=copy.deepcopy(r);del bad['tools']['llvm-config'];cases.append(bad)
 bad=copy.deepcopy(r);del bad['inputs'][next(n for n in r['inputs'] if n.endswith('/hip/hip_runtime.h'))];cases.append(bad)
 for name in sorted(r['artifacts']):
  if name.endswith(('.o','.hsaco','.bundle','.d')):
   bad=copy.deepcopy(r);del bad['artifacts'][name];cases.append(bad)
 for bad in cases:
  try:check(out,bad,rehash=False)
  except (AssertionError,KeyError):pass
  else:raise AssertionError('receipt omission accepted')
 print('REDUCTION_COMPILER_PASS inventory_mutations='+str(len(cases)))
if __name__=='__main__':main()
