"""Versioned fail-closed MIR controls; original runner/evidence stay immutable."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys
p=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
sys.path.insert(0,str(p.parent/'g3-private-backend-validation'))
from diagnostics import check_diagnostics

def require_complete(root, mode, matrix, compiler_hash=None):
 folder=root/'runs-v2'/mode
 report=json.loads((folder/'result.json').read_text())
 assert report['mode']==mode and report['all_controls_passed'] is True
 assert report['matrix_sha256']==sha(root/'matrix.json')
 assert len(matrix['cases'])==10 and [c['name'] for c in report['cases']]==[c['name'] for c in matrix['cases']]
 if compiler_hash is not None:assert report['compiler_sha256']==compiler_hash,'patched compiler changed since default-off'
 for case in report['cases']:
  assert case['stages']=={'pre-pei':0,'assembly':0},'incomplete prerequisite case'
  assert {'pre-pei.log','assembly.log','post-proof.mir','output.s'} <= case['files'].keys()
  for file,digest in case['files'].items():
   assert Path(file).name==file and sha(folder/case['name']/file)==digest,'prerequisite artifact changed'
 return report

def run(args, root=p):
 out=root/'runs-v2'/(args.run_label or args.mode)
 out.mkdir(parents=True,exist_ok=False)
 records=[];commands=[];matrix={};before=None;active=None
 def partial():
  if active is not None:
   directory=out/active['name'];active['files']={f.name:sha(f) for f in directory.iterdir() if f.is_file()}
  (out/'partial.json').write_text(json.dumps({'cases':records,'commands':commands,'all_controls_passed':False},indent=2)+'\n')
 try:
  matrix=json.loads((root/'matrix.json').read_text());assert len(matrix['cases'])==10
  compiler=args.compiler.resolve();before=sha(compiler)
  if args.mode.startswith('patched'):
   require_complete(root,'baseline',matrix)
   if args.mode=='patched-on':require_complete(root,'patched-off',matrix,before)
  for case in matrix['cases']:
   d=out/case['name'];d.mkdir();source=root/case['file'];assert sha(source)==case['sha256']
   base=[str(compiler),'-mtriple=amdgcn-amd-amdhsa','-mcpu='+case['target'],'-O2','-verify-machineinstrs','-start-before=si-fix-vgpr-copies']
   if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
   active={'name':case['name'],'stages':{}};records.append(active)
   for stage,extra in [('pre-pei',['-stop-before=prologepilog','-o',str(d/'post-proof.mir')]),('assembly',['-filetype=asm','-o',str(d/'output.s')])]:
    cmd=base+extra+[str(source)];r=subprocess.run(cmd,capture_output=True,text=True)
    (d/(stage+'.log')).write_text(r.stdout+r.stderr)
    commands.append({'command':cmd,'exit_code':r.returncode});active['stages'][stage]=r.returncode;partial()
    if r.returncode:
     raise RuntimeError(f"known-valid control regressed: {case['name']} {stage} exit {r.returncode}")
    if stage=='pre-pei':
     mir=(d/'post-proof.mir').read_text();active['mir_verifier_passed']=True
     if case['kind']=='variable-frame':assert 'type: variable-sized' in mir
     if case['kind']=='residual-alias':assert re.search(r'SI_CALL[^\n]*@late_alias,',mir)
     if case['kind']=='bundle':assert 'BUNDLE {' in mir
     if case['kind']=='unknown-spill-pseudo':assert 'SI_SPILL_V32_SAVE' in mir
     if args.mode=='patched-on':
      active['diagnostics']=check_diagnostics(r.stderr,source.read_text().replace('\n  define','\ndefine'),{'stress':case['expected_root_proof']},['stress','middle','leaf'])
    else:
     asm=(d/'output.s').read_text();body=re.search(r'(?ms)^stress:.*?(?=^\.Lfunc_end)',asm);assert body
     count=body[0].count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_');active['root_flat_init']=count
     assert count==(0 if args.mode=='patched-on' and case['kind']=='unchanged' else 2),(case['name'],count)
     if args.mode=='patched-off':assert sha(d/'output.s')==sha(root/'runs-v2/baseline'/case['name']/'output.s')
   assert active['stages']=={'pre-pei':0,'assembly':0}
   active['classification']='valid MIR and codegen';partial()
  assert len(records)==10 and sha(compiler)==before
  report={'mode':args.mode,'compiler_sha256':before,'matrix_sha256':sha(root/'matrix.json'),'cases':records,'commands':commands,'all_controls_passed':True,'hardware_validated':False,'patched_acceptance_complete':False,'script_sha256':sha(Path(__file__)),'limit':'Parent must separately bind same-source/configuration baseline build provenance; this runner verifies matching fixture and artifact identities.'}
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS10controls',sha(out/'result.json'));return 0
 except Exception as error:
  partial()
  failure={'mode':args.mode,'error':repr(error),'all_controls_passed':False,'compiler_sha256':before,'cases':records,'commands':commands,'hardware_validated':False,'script_sha256':sha(Path(__file__))}
  (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
  print(str(error),file=sys.stderr);return 1

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--compiler',type=Path,required=True);parser.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);parser.add_argument('--run-label');sys.exit(run(parser.parse_args()))
