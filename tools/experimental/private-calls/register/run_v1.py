"""Focused MIR operands; diagnostic inputs only, no hardware execution."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,difflib
root=Path(__file__).resolve().parent
repo=root.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(root.parent/'g3-private-backend-validation'))
from diagnostics import check_diagnostics
source=root.parent/'g3-private-backend-mir-controls/inputs/unchanged-gfx1013.mir'
text=source.read_text();start=text.index('name:            leaf\n');end=text.index('\n...\n',start)
leaf=text[start:end];marker='    renamable '
assert marker in leaf
variants={
 'implicit-full':'S_NOP 0, implicit undef $flat_scr',
 'implicit-low':'S_NOP 0, implicit undef $flat_scr_lo',
 'implicit-high':'S_NOP 0, implicit undef $flat_scr_hi',
 'explicit-low':'dead $sgpr5 = S_MOV_B32 undef $flat_scr_lo',
}
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--baseline-compiler',type=Path,required=True)
parser.add_argument('--patched-compiler',type=Path,required=True)
args=parser.parse_args()
compilers={'baseline':args.baseline_compiler.resolve(strict=True),'patched-on':args.patched_compiler.resolve(strict=True)}
identities={mode:sha(path) for mode,path in compilers.items()}
for mode,digest in identities.items():assert json.loads((repo/('validation-'+mode+'.json')).read_text())['compiler']['sha256']==digest,'compiler differs from completed mode'
out=root/'runs-v1';out.mkdir(exist_ok=False);records=[];commands=[]
try:
 for kind,instruction in variants.items():
  folder=out/kind;folder.mkdir();fixture=folder/'input.mir'
  changed=leaf.replace(marker,'    '+instruction+'\n'+marker,1)
  fixture.write_text(text[:start]+changed+text[end:])
  record={'name':kind,'input_sha256':sha(fixture),'source_sha256':sha(source),'modes':{}};records.append(record)
  for mode,compiler in compilers.items():
   dest=folder/mode;dest.mkdir();base=[str(compiler),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx1013','-O2','-verify-machineinstrs','-start-before=si-fix-vgpr-copies']
   if mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
   status={};record['modes'][mode]=status
   for stage,extra in [('pre-pei',['-stop-before=prologepilog','-o',str(dest/'post-proof.mir')]),('assembly',['-filetype=asm','-o',str(dest/'output.s')])]:
    command=base+extra+[str(fixture)];run=subprocess.run(command,capture_output=True,text=True)
    (dest/(stage+'.log')).write_text(run.stdout+run.stderr);commands.append({'command':command,'exit_code':run.returncode})
    status[stage]=run.returncode
    assert run.returncode==0,(kind,mode,stage,run.stderr)
    if stage=='pre-pei':
     mir=(dest/'post-proof.mir').read_text();assert instruction.split(' = ')[-1] in mir,(kind,'operand disappeared before proof')
    if mode=='patched-on':
     status[stage+'_diagnostics']=check_diagnostics(run.stderr,fixture.read_text().replace('\n  define','\ndefine'),{'leaf':'rejected','middle':'rejected','stress':'rejected'},['leaf','middle','stress'])
     assert 'buffer-call-proof register-use leaf explicit 1 with-mask 1' in run.stderr
   asm=(dest/'output.s').read_text();body=re.search(r'(?ms)^stress:.*?(?=^\.Lfunc_end)',asm);assert body
   assert body[0].count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_')==2
   if mode=='patched-on':assert (dest/'output.s').read_bytes()==(folder/'baseline/output.s').read_bytes()
   status['files']={f.name:sha(f) for f in dest.iterdir()}
  (out/'partial.json').write_text(json.dumps({'cases':records,'commands':commands},indent=2)+'\n')
 assert all(sha(path)==identities[mode] for mode,path in compilers.items())
 report={'all_four_passed':True,'cases':records,'commands':commands,'compiler_sha256':identities,'recipe_sha256':sha(Path(__file__)),'hardware_validated':False,'limits':['MIR diagnostics deliberately add undef physical operands; they verify compiler admission/register accounting, not useful GPU instruction semantics.','Only gfx1013 tested for these four operand forms; full two-target ordinary matrix is separate.']}
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(out/'result.json'))
except Exception as error:
 (out/'failure.json').write_text(json.dumps({'error':repr(error),'cases':records,'commands':commands,'compiler_sha256':identities},indent=2)+'\n');raise
