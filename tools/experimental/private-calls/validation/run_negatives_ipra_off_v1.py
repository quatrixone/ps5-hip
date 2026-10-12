from pathlib import Path
import argparse,hashlib,json,re,subprocess
from diagnostics import check_diagnostics
out=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);p.add_argument('--run-label');args=p.parse_args()
matrix=read(out/'negative-matrix.json');folder=out/'negative-runs'/(args.run_label or args.mode);folder.mkdir(parents=True,exist_ok=False)
compiler=args.compiler.resolve();before=sha(compiler);results=[];commands=[]
for case in matrix['cases']:
 for target in matrix['targets']:
  name=case['name']+'-'+target;print(args.mode,name,flush=True);d=folder/name;d.mkdir();source=out/case['input'];assert sha(source)==case['sha256']
  base=[str(compiler),'-enable-ipra=false','-mtriple=amdgcn-amd-amdhsa','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-O2','-verify-machineinstrs']
  if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
  failed=None
  for stage,extra in [('assembly',['-filetype=asm','-o',str(d/'output.s')]),('object',['-filetype=obj','-o',str(d/'output.o')]),('pre-pei',['-stop-before=prologepilog','-o',str(d/'pre-pei.mir')])]:
   cmd=base+extra+[str(source)];r=subprocess.run(cmd,capture_output=True,text=True);(d/(stage+'.log')).write_text(r.stdout+r.stderr);commands.append({'case':name,'stage':stage,'command':cmd,'exit_code':r.returncode})
   if r.returncode:failed={'stage':stage,'exit_code':r.returncode};break
  record={'name':name,'compiler_failed':failed}
  if failed:
   assert case['name']=='dynamic' and 'unsupported dynamic alloca' in (d/'assembly.log').read_text(),(name,'unexpected compiler failure')
   record['classification']='unsupported before proof; coverage gap, not passed rejection'
  if not failed:
   asm=(d/'output.s').read_text();kernel=re.search(r'(?ms)^negative:.*?(?=^\.Lfunc_end)',asm);assert kernel,name
   count=kernel[0].count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_');record['flat_init_count']=count
   record['pre_pei_calls']=[s.strip() for s in (d/'pre-pei.mir').read_text().splitlines() if 'SI_CALL' in s]
   record['flat_initialization_retained']=count==2
   if args.mode!='patched-on':assert count==2,(name,'baseline lost initialization')
   if args.mode=='patched-on':
    canonical_alias=case['name']=='alias'
    if canonical_alias:
     mir=(d/'pre-pei.mir').read_text();assert '@alias' not in mir and '@safe' in mir,'alias unexpectedly retained'
    expected={'negative':'proven' if canonical_alias else 'rejected'}
    record['diagnostics']=check_diagnostics((d/'assembly.log').read_text(),source.read_text(),expected,['negative'])
    assert count==(0 if canonical_alias else 2),(name,'unexpected suppression')
    record['classification']='canonicalized-safe alias control; not alias rejection' if canonical_alias else 'actual proof rejection'
   if args.mode=='patched-off':
    for file in ('output.s','output.o'):assert sha(d/file)==sha(out/'negative-runs/baseline'/name/file),(name,'defaultoffchanged')
  record['files']={f.name:sha(f) for f in d.iterdir()};results.append(record)
  (folder/'partial.json').write_text(json.dumps({'cases':results,'commands':commands},indent=2)+'\n')
assert sha(compiler)==before
report={'mode':args.mode,'compiler_sha256':before,'matrix_sha256':sha(out/'negative-matrix.json'),'cases':results,'commands':commands,'hardware_validated':False,'patched_acceptance_complete':False,'limits':['Dynamic alloca compiler failure is unsupported coverage, not an accepted proof rejection.','No linking required: unresolved externals/indirect cases intentionally test codegen admission and retained initialization only.','Source alias canonicalizes away before proof and is a safe-direct-call control, not residual alias rejection. Other successful cases require actual diagnostic rejection and retained flat initialization.'],'script_sha256':sha(Path(__file__))}
(folder/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',len(results),'cases; backendfailures',sum(bool(r['compiler_failed']) for r in results),sha(folder/'result.json'))
