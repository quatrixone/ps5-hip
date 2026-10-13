"""Inspect each unchanged genuine module independently for the native two-TU title."""
import argparse,hashlib,json,pathlib,shutil,struct,subprocess,sys
import probe_hip_routes as shared
from fat_binary_fixture_test import section
R=pathlib.Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXPECTED='1e39ead5c5388cff501366e0d0fa287db7938f8ac6ca3317302f56a6ade2beb9'
def extract(bundle):
 magic=b'__CLANG_OFFLOAD_BUNDLE__';assert bundle.startswith(magic)
 count=struct.unpack_from('<Q',bundle,len(magic))[0];assert count==2
 at=len(magic)+8;found=[]
 for _ in range(count):
  offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24
  name=bundle[at:at+length];at+=length
  assert offset<=len(bundle) and size<=len(bundle)-offset
  if name==b'hipv4-amdgcn-amd-amdhsa--gfx1013':found.append(bundle[offset:offset+size])
 assert len(found)==1 and found[0][:4]==b'\x7fELF';return found[0]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 proof=a.proof.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 assert H(proof/'receipt.json')==EXPECTED
 r=json.loads((proof/'receipt.json').read_text())
 sources=dict(r['sources']);sources[str(pathlib.Path(__file__).resolve().relative_to(R))]=H(pathlib.Path(__file__))
 sources['tests/hip_multi_recipe_test.py']=H(R/'tests/hip_multi_recipe_test.py')
 assert all(H(R/n)==h for n,h in sources.items())
 inputs=r['inputs']|{str(proof/n):h for n,h in r['artifacts'].items()}|{str(proof/'receipt.json'):EXPECTED}
 assert all(H(pathlib.Path(n))==h for n,h in inputs.items())
 subprocess.run([sys.executable,str(R/'tests/hip_multi_recipe_test.py'),'--check',str(proof)],check=True)
 commands=[];cases=[]
 def run(label,*args):
  q=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=60)
  for suffix,text in (('.stdout',q.stdout),('.stderr',q.stderr)):(out/(label+suffix)).write_text(text)
  commands.append(dict(label=label,argv=list(map(str,args)),exit_code=q.returncode));assert q.returncode==0,q.stderr
  return q.stdout
 for host in ('linux','ps5'):
  (out/host).mkdir()
  for actor in 'ab':
   obj=out/host/(actor+'.o');bundle=out/host/(actor+'.bundle');code=out/host/(actor+'.hsaco')
   shutil.copyfile(proof/host/(actor+'.o'),obj);bundle.write_bytes(section(obj.read_bytes(),b'.hip_fatbin'));code.write_bytes(extract(bundle.read_bytes()))
   metadata=run(host+actor+'-metadata','/usr/bin/llvm-readobj-18','--notes',code)
   isa=run(host+actor+'-isa','/usr/bin/llvm-objdump-18','--disassemble','--mcpu=gfx1013',code)
   audit=shared.audit_object(code.read_bytes(),metadata,isa,('multi_'+actor,))
   d=audit['descriptors']['multi_'+actor]
   assert (d['group_bytes'],d['private_bytes'],d['kernarg_bytes'],d['rsrc1'],d['rsrc2'],d['rsrc3'],d['properties'],d['preload'])==(0,0,280,1622081600,0x8c,0,0x409,0)
   ops=audit['isa']['multi_'+actor]['opcodes'];assert not any(op.startswith(('flat_','scratch_','s_swappc')) for op in ops)
   assert any(op.startswith('global_load') for op in ops) and any(op.startswith('global_store') for op in ops)
   cases.append(dict(host=host,module=actor,files={p.name:H(p) for p in (obj,bundle,code)},audit=audit))
 for actor in 'ab':assert section((out/'linux'/(actor+'.hsaco')).read_bytes(),b'.text')==section((out/'ps5'/(actor+'.hsaco')).read_bytes(),b'.text')
 assert all(H(R/n)==h for n,h in sources.items()) and all(H(pathlib.Path(n))==h for n,h in inputs.items())
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 result=dict(classification='offline-two-module-compiler-inspection-pass',sources=sources,inputs=inputs,upstream_sha256=EXPECTED,cases=cases,hardware_executed=False,artifacts={str(p.relative_to(out)):H(p) for p in out.rglob('*') if p.is_file()})
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',H(out/'receipt.json'))
if __name__=='__main__':main()
