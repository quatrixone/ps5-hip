from pathlib import Path
import argparse,hashlib,json,re,subprocess
out=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);args=p.parse_args()
root=out/'reload-runs'/args.mode;root.mkdir(parents=True,exist_ok=False);compiler=args.compiler.resolve();before=sha(compiler);records=[];commands=[]
matrix=json.loads((out/'matrix.json').read_text())
for case in matrix['cases']:
 if case['kind']!='ordinary' or case['optimization']!='O2' or 'reverse' in case['id']:continue
 d=root/case['id'];d.mkdir();source=out/case['input'];assert sha(source)==case['sha256']
 base=[str(compiler),'-enable-ipra=false','-mtriple=amdgcn-amd-amdhsa','-mcpu='+case['target'],'-O2','-verify-machineinstrs']
 if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
 for stage,extra in [('save',['-stop-before=prologepilog',str(source),'-o',str(d/'pre-pei.mir')]),('reload',['-start-before=prologepilog','-filetype=asm',str(d/'pre-pei.mir'),'-o',str(d/'reloaded.s')])]:
  cmd=base+extra;r=subprocess.run(cmd,capture_output=True,text=True);(d/(stage+'.log')).write_text(r.stdout+r.stderr);commands.append({'command':cmd,'exit_code':r.returncode});r.check_returncode()
 asm=(d/'reloaded.s').read_text();body=re.search(r'(?ms)^nested_private:.*?(?=^\.Lfunc_end)',asm);assert body
 assert body[0].count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_')==2,'reloaded MIR retained an unauthorized proof'
 assert 'buffer-call-proof result' not in (d/'reload.log').read_text(),'reload unexpectedly recomputed proof: assess separately'
 assert 'ProvenBufferOnly' not in (d/'pre-pei.mir').read_text(),'proof unexpectedly serialized'
 records.append({'id':case['id'],'reloaded_root_flat_init':2,'proof_recomputed':False,'files':{f.name:sha(f) for f in d.iterdir()}})
assert len(records)==2 and sha(compiler)==before
report={'mode':args.mode,'compiler_sha256':before,'cases':records,'commands':commands,'hardware_validated':False,'patched_acceptance_complete':False,'scope':'Only MIR reload started at PEI with no proof recomputation: initialization must be conservative.'}
(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(records),sha(root/'result.json'))
