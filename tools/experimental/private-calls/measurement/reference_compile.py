from pathlib import Path
import hashlib,json,subprocess,shutil
out=Path(__file__).resolve().parent;source_root=out.parent/'g3-private-backend-validation-v4';old=out.parent/'g3-private-backend-validation-v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
control=json.loads((old/'runs/baseline/result.json').read_text());compiler=Path(control['compiler']);assert sha(compiler)==control['compiler_sha256']
matrix=json.loads((source_root/'matrix.json').read_text());cases=[c for c in matrix['cases'] if c['kind'] in ('ordinary','volatile') and c['order']=='original'];assert len(cases)==8
versions={};tools={}
for name,path in [('llc',compiler)]+[(n,Path(shutil.which(n))) for n in ['ld.lld-18','llvm-objdump-18','llvm-readobj-18']]:
 tools[name]={'path':str(path.resolve()),'sha256':sha(path.resolve())};versions[name]=subprocess.check_output([str(path),'--version'],text=True)
results=[];commands=[]
for case in cases:
 d=out/case['id'];d.mkdir();source=source_root/case['input'];assert sha(source)==case['sha256'];base=[str(compiler),'-enable-ipra=false','-mtriple=amdgcn-amd-amdhsa','-mcpu='+case['target'],'-'+case['optimization'],'-verify-machineinstrs']
 for name,cmd in [('asm',base+['-filetype=asm',str(source),'-o',str(d/'output.s')]),('object',base+['-filetype=obj',str(source),'-o',str(d/'output.o')]),('link',['ld.lld-18','-shared','--no-undefined',str(d/'output.o'),'-o',str(d/'output.hsaco')]),('isa',['llvm-objdump-18','-d','--mcpu='+case['target'],str(d/'output.hsaco')]),('metadata',['llvm-readobj-18','--notes','--symbols','--relocations',str(d/'output.hsaco')])]:
  p=subprocess.run(cmd,capture_output=True,text=True);log=d/(name+'.txt' if name in ('isa','metadata') else name+'.log');log.write_text(p.stdout+p.stderr);commands.append({'case':case['id'],'argv':cmd,'exit_code':p.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');p.check_returncode()
 results.append({'case':case,'source_path':str(source),'files':{p.name:sha(p) for p in d.iterdir()}});print(case['id'],'pass',flush=True)
assert sha(compiler)==control['compiler_sha256']
report={'classification':'independent clean baseline with explicit IPRA disabled; no proof patch or native execution','compiler':tools['llc'],'paired_old_baseline_receipt_sha256':sha(old/'runs/baseline/result.json'),'tools':tools,'versions':versions,'matrix_sha256':sha(source_root/'matrix.json'),'cases':results,'script_sha256':sha(Path(__file__)),'hardware_validated':False,'patched_acceptance':False}
(out/'compile-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(sha(out/'compile-receipt.json'))
