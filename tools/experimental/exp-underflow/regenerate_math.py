#!/usr/bin/env python3
"""Regenerate retained six-operation OCML objects with isolated baseline/patched llc."""
import argparse,hashlib,json,pathlib,re,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from fat_binary_fixture_test import section
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(yes,why):
 if not yes:raise ValueError(why)
def audit_assembly(before,after):
 # The actual linked OCML kernel has exp(x) and tanh's exp(2*abs(x)).
 require(before.count('0xc2ce8ed0')==2 and after.count('0xc2cff1b4')==2,'exactly two natural-exp cutoff sites required')
 require(before.replace('0xc2ce8ed0','0xc2cff1b4')==after,'unexpected six-op assembly change')
 return {'natural_exp_literal_sites_changed':2,'remaining_assembly_exact':True}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=pathlib.Path,required=True);p.add_argument('--original',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
 experiment=args.experiment.resolve();original=args.original.resolve();out=args.output.resolve();require(not out.exists() and out.is_relative_to(ROOT/'results'),'fresh isolated output required');out.mkdir(parents=True)
 inputs={};commands=[];profiles=[]
 def bind(path):
  path=pathlib.Path(path).resolve();digest=sha(path);require(str(path) not in inputs or inputs[str(path)]==digest,'input drift');inputs[str(path)]=digest;return path
 def run(label,argv):
  print('START '+label,flush=True);argv=list(map(str,argv));z=subprocess.run(argv,cwd=out,capture_output=True,timeout=180)
  (out/(label+'.stdout')).write_bytes(z.stdout);(out/(label+'.stderr')).write_bytes(z.stderr);commands.append({'label':label,'argv':argv,'exit':z.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(z.returncode==0,label+' failed '+z.stderr.decode(errors='replace')[-2000:]);return z.stdout.decode()
 try:
  bind(HERE/'regenerate_math.py');bind(ROOT/'tests/fat_binary_fixture_test.py');receipt=bind(experiment/'receipt.json');record=json.loads(receipt.read_text())
  for tool in ('/usr/bin/llvm-readobj-18','/usr/bin/llvm-objdump-18'):bind(tool)
  require(record['classification']=='passed-isolated-exp-cutoff-compiler-matrix','compiler experiment failed')
  compilers={'baseline':pathlib.Path('/home/denis/ps5hip-llvm18-private-backend/build-baseline/bin/llc'),'noop':experiment/'noop/llc','patched':experiment/'patched/llc'}
  for mode,path in compilers.items():
   bind(path)
   expected=record['inputs'][str(path)] if mode=='baseline' else record[mode]['llc_sha256'];require(sha(path)==expected,'compiler identity mismatch')
  bind(original/'receipt.json');bind(ROOT/'tools/math_oracle_contract.json');require(sha(ROOT/'tools/math_oracle_contract.json')=='373c94b8ad8dac7326401ad54a1299d750298a5922c1a226d0d4816c854430b8','oracle changed')
  for target in ('gfx1013','gfx1030'):
   for opt in ('O1','O2'):
    label=target+'-'+opt;driver=bind(original/(label+'-driver.stderr'));lines=[shlex.split(x) for x in driver.read_text().splitlines() if x.startswith(' "')];require(len(lines)==2 and '-cc1' in lines[0] and '-flavor' in lines[1],'retained driver structure')
    cc,link=lines;require(cc.count('-emit-obj')==1 and cc[-2]=='hip','expected HIP cc1 command');source=bind(cc[-1]);require(sha(source)=='31bfd82b70cd2a55d1b0579d9e1e2c70c2d11b91cf60d892fe36d79c771fbe54','fixture changed')
    require(pathlib.Path(cc[0]).resolve()==pathlib.Path('/usr/lib/llvm-18/bin/clang').resolve(),'unexpected compiler');bind(cc[0]);bind(link[0])
    libs=[bind(cc[i+1]) for i,x in enumerate(cc) if x=='-mlink-builtin-bitcode'];require(len(libs)==9,'exact device libraries required')
    directory=out/label;directory.mkdir();bc=directory/'linked-optimized.bc';dep=directory/'headers.d'
    cc[cc.index('-o')+1]=str(bc);cc[cc.index('-dumpdir')+1]=str(directory/'diagnostic-');cc[cc.index('-emit-obj')]='-emit-llvm-bc';cc+=['-dependency-file',str(dep),'-MT','linked-optimized.bc','-sys-header-deps']
    pre=cc.copy();pre[pre.index('-emit-llvm-bc')]='-E';pre[pre.index('-o')+1]=str(directory/'preprocessed.hip')
    run(label+'-dependencies',pre)
    text=dep.read_text().replace('\\\n',' ')
    for path in shlex.split(text.split(':',1)[1]):bind(path)
    run(label+'-linked-ir',cc)
    existing=bind(original/(label+'.hsaco'));objects={};assemblies={}
    for mode,compiler in compilers.items():
     obj=directory/(mode+'.o');code=directory/(mode+'.hsaco');asm=directory/(mode+'.s')
     common=[compiler,'-march=amdgcn','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-'+opt,'-relocation-model=pic','-amdgpu-internalize-symbols','-amdhsa-code-object-version=5',bc]
     run(label+'-'+mode+'-object',common+['-filetype=obj','-o',obj]);run(label+'-'+mode+'-assembly',common+['-filetype=asm','-o',asm])
     command=link.copy();command[command.index('-o')+1]=str(code);indices=[i for i,x in enumerate(command) if x.endswith('.o')];require(len(indices)==1,'one native object required');command[indices[0]]=str(obj)
     run(label+'-'+mode+'-link',command);objects[mode]=code.read_bytes();assemblies[mode]=asm.read_text()
     run(label+'-'+mode+'-notes',['/usr/bin/llvm-readobj-18','--notes',code]);run(label+'-'+mode+'-isa',['/usr/bin/llvm-objdump-18','--disassemble','--mcpu='+target,code])
    require(objects['baseline']==objects['noop'],'no-op changed six-op ELF')
    audit_assembly(assemblies['baseline'],assemblies['patched'])
    # Require the new compilation pipeline reproduces actual old executable
    # instructions, descriptor and metadata before qualifying a replacement.
    comparisons={name:section(existing.read_bytes(),name.encode())==section(objects['baseline'],name.encode()) for name in ('.text','.rodata','.note')}
    require(all(comparisons.values()),'baseline cc1-IR/llc regeneration differs from frozen sections '+str(comparisons))
    require(section(objects['baseline'],b'.rodata')==section(objects['patched'],b'.rodata') and section(objects['baseline'],b'.note')==section(objects['patched'],b'.note'),'descriptor/metadata changed')
    profiles.append({'target':target,'optimization':opt,'original_sha256':sha(existing),'linked_ir_sha256':sha(bc),'objects':{m:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for m,v in objects.items()},'original_sections_reproduced':comparisons,'only_natural_exp_cutoff_changed':True})
  # Bind inspection tools too and enforce all consumed identities after use.
  for tool in ('/usr/bin/llvm-readobj-18','/usr/bin/llvm-objdump-18'):bind(tool)
  require(all(sha(path)==digest for path,digest in inputs.items()),'consumed input changed')
  report={'schema':1,'classification':'passed-six-operation-object-regeneration','hardware_validated':False,'numeric_accuracy_qualified':False,'oracle_changed':False,'inputs':inputs,'commands':commands,'profiles':profiles,'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'limits':['Same host and retained supplied LLVM/OCML components; not a clean/hermetic compiler build.','Instruction and clamp correction is offline evidence; native EXP/FLDEXP numerical execution remains untested.','No replacement package, console run, or oracle relaxation.']}
  (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS '+sha(out/'receipt.json'),flush=True)
 except BaseException as error:
  (out/'failure.json').write_text(json.dumps({'schema':1,'error':repr(error),'inputs':inputs,'commands':commands,'profiles':profiles,'hardware_validated':False},indent=2)+'\n');raise
if __name__=='__main__':main()
