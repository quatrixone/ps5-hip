from pathlib import Path
import argparse,hashlib,json,re,subprocess
from diagnostics_v2 import check_diagnostics, DEAD_ORDINARY_ORIGINALS
from frame_audit_v3 import inspect_case
out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
 p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);p.add_argument('--run-label');args=p.parse_args()
 measured_path=out/'ipra-frame-contracts.json'
 assert sha(measured_path)=='33179c0a773c1072602b5c6b6842a430a9a81a4e9a1ad4eb65751f7680cf5699', 'independent frame measurements changed'
 measured=read(measured_path)
 matrix=read(out/'matrix.json');folder=out/'runs'/(args.run_label or args.mode);folder.mkdir(parents=True,exist_ok=False)
 compiler=args.compiler.resolve();before=sha(compiler);results=[];commands=[]
 if args.mode=='baseline':assert before==measured['compiler']['sha256'], 'baseline compiler differs from independent frame measurement'
 for profile in measured['profiles']:
  original=next(c for c in matrix['cases'] if c['kind']==profile['kind'] and c['target']==profile['target'] and c['optimization']==profile['optimization'] and c['order']=='original')
  assert original['sha256']==profile['input_sha256'], 'measurement input differs from matrix'
 baseline=read(out/'runs/baseline/result.json') if args.mode.startswith('patched') else None
 if args.mode=='patched-on':
  off=read(out/'runs/patched-off/result.json')
  assert off['same_build_baseline_identical'] is True
  assert off['compiler_sha256']==before, 'enabled compiler differs from verified disabled compiler'
  assert off['matrix_sha256']==baseline['matrix_sha256']==sha(out/'matrix.json'), 'control matrix differs'
 try:
  for case in matrix['cases']:
   print(args.mode,case['id'],flush=True);d=folder/case['id'];d.mkdir();source=out/case['input'];assert sha(source)==case['sha256']
   base=[str(compiler),'-enable-ipra=false','-mtriple=amdgcn-amd-amdhsa','-mcpu='+case['target'],'-'+case['optimization'],'-verify-machineinstrs']
   if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
   for name,cmd in [('asm',base+['-filetype=asm',str(source),'-o',str(d/'output.s')]),('object',base+['-filetype=obj',str(source),'-o',str(d/'output.o')]),('link',['ld.lld-18','-shared','--no-undefined',str(d/'output.o'),'-o',str(d/'output.hsaco')]),('isa',['llvm-objdump-18','-d','--mcpu='+case['target'],str(d/'output.hsaco')]),('metadata',['llvm-readobj-18','--notes','--symbols','--relocations',str(d/'output.hsaco')])]:
    r=subprocess.run(cmd,capture_output=True,text=True);(d/(name+'.log' if name not in ('isa','metadata') else name+'.txt')).write_text(r.stdout+r.stderr)
    commands.append({'case':case['id'],'command':cmd,'exit_code':r.returncode});(folder/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');r.check_returncode()
   roots=inspect_case(case,d,args.mode=='patched-on' and case['kind']=='ordinary',measured)
   if baseline:
    basecase=next(c for c in baseline['cases'] if c['id']==case['id']);basedir=out/'runs/baseline'/case['id']
    assert sha(basedir/'metadata.txt')==basecase['files']['metadata.txt'], 'baseline metadata changed'
   diagnostics=None
   if args.mode=='patched-on':
    expected={root['root']:('proven' if case['kind']=='ordinary' else 'rejected') for root in roots}
    reachable={name for root in roots for name in root['reachable']}
    if case['kind']=='ordinary':expected.update({name:'proven' for name in reachable})
    diagnostics=check_diagnostics((d/'asm.log').read_text(),source.read_text(),expected,reachable,
      linked_metadata=(d/'metadata.txt').read_text(),baseline_metadata=(basedir/'metadata.txt').read_text(),
      allowed_dead=DEAD_ORDINARY_ORIGINALS if case['kind']=='ordinary' else ())
   if baseline:
    basecase=next(c for c in baseline['cases'] if c['id']==case['id']);basedir=out/'runs/baseline'/case['id']
    if args.mode=='patched-off':
     for name in ('output.s','output.o','output.hsaco'):assert sha(d/name)==sha(basedir/name),(case['id'],name,'defaultoffdiff')
    else:
     assert [r['descriptor'] for r in roots]==[r['descriptor'] for r in basecase['roots']],(case['id'],'resourcechange')
   results.append({'id':case['id'],'roots':roots,'diagnostics':diagnostics,'files':{p.name:sha(p) for p in sorted(d.iterdir())}})
  assert sha(compiler)==before
  report={'mode':args.mode,'compiler':str(compiler),'compiler_sha256':before,'matrix_sha256':sha(out/'matrix.json'),'hardware_validated':False,'patched_acceptance_complete':False,'same_build_baseline_identical':args.mode=='patched-off','cases':results,'limits':['Installed compiler controls are not evidence for patched compiler behavior.','Matrix checks selected observable ordinary/generic/volatile graphs only. Full unknown/external/alias/recursive/inlineasm/dynamicframe/forgedattr/MIR reload negative acceptance remains separate.','Even passing patched matrix does not alone establish pre-PEI all-machine-functions barrier instrumentation or authorize GPU use.'],'script_sha256':sha(Path(__file__)), 'diagnostic_parser_sha256':sha(out/'diagnostics_v2.py'),'frame_audit_sha256':sha(out/'frame_audit_v3.py'),'measured_frames_sha256':sha(measured_path),'enable_ipra':False}
  (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(results),'cases',sha(folder/'result.json'))
 except Exception as e:
  (folder/'failure.json').write_text(json.dumps({'error':repr(e),'completed_cases':len(results),'commands':commands},indent=2)+'\n');raise
if __name__=='__main__':main()
