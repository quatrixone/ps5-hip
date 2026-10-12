from pathlib import Path
import argparse,hashlib,json,re,subprocess
from diagnostics import check_diagnostics
out=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);args=p.parse_args()
root=out/'ineligible-runs'/args.mode;root.mkdir(parents=True,exist_ok=False);compiler=args.compiler.resolve();before=sha(compiler)
source=out/'stress-inputs/late-callee-unbounded.ll';base=[str(compiler),'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx900','-O2','-verify-machineinstrs']
if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
commands=[]
for kind,ext in [('asm','s'),('obj','o')]:
 cmd=base+['-filetype='+kind,str(source),'-o',str(root/('output.'+ext))];r=subprocess.run(cmd,capture_output=True,text=True);(root/(kind+'.log')).write_text(r.stdout+r.stderr);commands.append({'command':cmd,'exit_code':r.returncode});r.check_returncode()
asm=(root/'output.s').read_text();assert asm.count('s_swappc_b64')==2 and 'flat_scratch_lo' in asm and 'flat_scratch_hi' in asm
diagnostics=None
if args.mode=='patched-on':diagnostics=check_diagnostics((root/'asm.log').read_text(),source.read_text(),{n:'rejected' for n in ['stress','middle','leaf']},['stress','middle','leaf'])
if args.mode in ('patched-off','patched-on'):
 for ext in ('s','o'):assert sha(root/('output.'+ext))==sha(out/'ineligible-runs/baseline'/('output.'+ext))
assert sha(compiler)==before
report={'mode':args.mode,'target':'gfx900','compiler_sha256':before,'source_sha256':sha(source),'calls':2,'flat_scratch_init_retained':True,'diagnostics':diagnostics,'commands':commands,'files':{f.name:sha(f) for f in root.iterdir()},'hardware_validated':False,'patched_acceptance_complete':False}
(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(root/'result.json'))
