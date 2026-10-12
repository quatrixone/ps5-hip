from pathlib import Path
import argparse,hashlib,json,re,subprocess
from diagnostics import check_diagnostics
out=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);p.add_argument('--run-label');p.add_argument('--fixture',default='late-spill');p.add_argument('--expect-proof',choices=['proven','rejected'],default='rejected');args=p.parse_args()
suffix='' if args.fixture=='late-spill' else '-'+args.fixture
root=out/'stress-runs'/(args.run_label or args.mode+suffix);root.mkdir(parents=True,exist_ok=False)
source=out/'stress-inputs'/(args.fixture+'.ll');manifest=out/('stress-input.json' if args.fixture=='late-spill' else args.fixture+'-input.json');assert sha(source)==json.loads(manifest.read_text())['sha256']
compiler=args.compiler.resolve();before=sha(compiler);records=[];commands=[]
for target in ['gfx1013','gfx1030']:
 d=root/target;d.mkdir();base=[str(compiler),'-enable-ipra=false','-mtriple=amdgcn-amd-amdhsa','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-O2','-verify-machineinstrs']
 if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
 for name,extra in [('output.s',['-filetype=asm']),('output.o',['-filetype=obj']),('pre-pei.mir',['-stop-before=prologepilog']),('post-pei.mir',['-stop-after=prologepilog'])]:
  cmd=base+extra+[str(source),'-o',str(d/name)];r=subprocess.run(cmd,capture_output=True,text=True);(d/(name+'.log')).write_text(r.stdout+r.stderr);commands.append({'command':cmd,'exit_code':r.returncode});r.check_returncode()
 asm=(d/'output.s').read_text();pre=(d/'pre-pei.mir').read_text();post=(d/'post-pei.mir').read_text()
 def opcounts(text):return {'buffer_load':len(re.findall(r'\bBUFFER_LOAD_\w+',text)),'buffer_store':len(re.findall(r'\bBUFFER_STORE_\w+',text)),'sgpr_vgpr_spill':len(re.findall(r'\bSI_SPILL_S\d+_TO_VGPR',text)),'vgpr_save':len(re.findall(r'\bSI_SPILL_V\d+_SAVE',text)),'flat':len(re.findall(r'\bFLAT_\w+',text)),'scratch':len(re.findall(r'\bSCRATCH_\w+',text))}
 record={'target':target,'pre_pei':opcounts(pre),'post_pei':opcounts(post),'assembly_buffer_load':len(re.findall(r'^\s*buffer_load_\w+',asm,re.M)),'assembly_buffer_store':len(re.findall(r'^\s*buffer_store_\w+',asm,re.M)),'calls':asm.count('s_swappc_b64'),'flat_init':asm.count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_')}
 assert record['calls']==2 and record['assembly_buffer_load'] and record['assembly_buffer_store']
 assert not re.search(r'^\s*(?:flat_|scratch_)\w+|src_private_',asm,re.M)
 assert record['flat_init']==(0 if args.mode=='patched-on' and args.expect_proof=='proven' else 2)
 if args.mode=='patched-on':record['diagnostics']=check_diagnostics((d/'output.s.log').read_text(),source.read_text(),{'stress':args.expect_proof},('stress','middle','leaf'))
 if args.mode=='patched-off':
  for f in ('output.s','output.o'):assert sha(d/f)==sha(out/'stress-runs'/('baseline'+suffix)/target/f)
 record['files']={f.name:sha(f) for f in d.iterdir()};records.append(record)
assert sha(compiler)==before
report={'mode':args.mode,'compiler_sha256':before,'source_sha256':sha(source),'fixture':args.fixture,'expected_proof':args.expect_proof,'cases':records,'commands':commands,'hardware_validated':False,'patched_acceptance_complete':False,'limits':'Stress intent is not late-spill coverage: inspect actual pre/post-PEI difference. No semantics or hardware claim.'}
(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(records,indent=2));print('receipt',sha(root/'result.json'))
