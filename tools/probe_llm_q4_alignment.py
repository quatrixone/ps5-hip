"""Offline Q4 load-alignment experiment; never authorizes a console launch."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
from build_llm_q4_fixture import metadata_contract,PROFILES
from probe_hip_routes import audit_object
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
f=args.fixture.resolve(); out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((f/'receipt.json').read_text())
for name in ['q4.bc','q4.ll','q4.hsaco','q4-isa.stdout']:
 assert sha(f/name)==r['artifacts'][name]['sha256'],name
inputs={str(f/n):sha(f/n) for n in ['q4.bc','q4.ll','q4.hsaco','q4-isa.stdout','receipt.json']}
for n in ['tools/probe_llm_q4_alignment.py','tools/build_llm_q4_fixture.py','tools/probe_hip_routes.py']:
 inputs[str(root/n)]=sha(root/n)
tool_versions={}
for name in ['llc-18','ld.lld-18','llvm-readobj-18','llvm-objdump-18']:
 tool=Path(shutil.which(name)).resolve()
 inputs[str(tool)]=sha(tool)
 version=subprocess.check_output([str(tool),'--version'],text=True)
 assert '18.1.8' in version,name
 tool_versions[name]=version.strip()
commands=[]
def run(name,args):
 p=subprocess.run(args,cwd=root,capture_output=True,text=True)
 (out/(name+'.stdout')).write_text(p.stdout)
 (out/(name+'.stderr')).write_text(p.stderr)
 commands.append(dict(args=list(map(str,args)),returncode=p.returncode))
 if p.returncode:raise RuntimeError(name)
 return p.stdout
results={}
for name in ['baseline','volatile-u16']:
 obj=out/(name+'.o'); module=out/(name+'.hsaco')
 source=f/'q4.bc'
 if name=='volatile-u16':
  source=out/'q4-volatile-u16.ll'
  ir=(f/'q4.ll').read_text()
  assert ir.count('load i16, ptr addrspace(1)')==4
  source.write_text(ir.replace('load i16, ptr addrspace(1)', 'load volatile i16, ptr addrspace(1)'))
 run(name+'-compile',['llc-18','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx1013',*flags,'-filetype=obj',str(source),'-o',str(obj)])
 run(name+'-link',['ld.lld-18','-shared',str(obj),'-o',str(module)])
 notes=run(name+'-notes',['llvm-readobj-18','--notes',str(module)])
 isa=run(name+'-isa',['llvm-objdump-18','--disassemble','--mcpu=gfx1013',str(module)])
 metadata=metadata_contract(notes,PROFILES['q4'])
 audit=audit_object(module.read_bytes(),notes,isa,[PROFILES['q4']['symbol']])
 descriptor=audit['descriptors'][PROFILES['q4']['symbol']]
 results[name]=dict(sha256=sha(module),metadata=metadata,descriptor=descriptor,global_loads=[line.strip() for line in isa.splitlines() if 'global_load_' in line])
assert results['baseline']['sha256']==sha(f/'q4.hsaco'),'baseline mismatch'
assert results['baseline']['metadata']==results['volatile-u16']['metadata']
assert sum('global_load_ushort' in x for x in results['volatile-u16']['global_loads'])==5
assert results['baseline']['sha256']!=results['volatile-u16']['sha256']
for path,digest in inputs.items():
 assert sha(Path(path))==digest,'input changed: '+path
(out/'receipt.json').write_text(json.dumps(dict(status='PASS_OFFLINE_ONLY',gpu_executed=False,inputs=inputs,tool_versions=tool_versions,commands=commands,results=results,artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},limits=['Volatile i16 loads change cache flags and scheduling; not a proven hardware fix.', 'Resource descriptor changes require independent runtime admission and native audit.', 'No numerical tolerance or original fixture modified.']),indent=2)+'\n')
print('PASS offline alignment experiment: '+str(out/'receipt.json'))
