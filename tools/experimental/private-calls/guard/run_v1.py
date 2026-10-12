"""Focused effective-IPRA guard tests; no complete backend acceptance claim."""
from pathlib import Path
import hashlib,json,re,resource,subprocess,sys
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parent/'g3-private-backend-validation'))
from diagnostics import check_diagnostics
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
llvm=Path('/home/denis/ps5hip-llvm18-private-backend')
compilers={'baseline':llvm/'build-baseline/bin/llc','patched':llvm/'build-patched/bin/llc'}
identities={name:sha(path) for name,path in compilers.items()}
assert identities['patched']=='881acfaaf52d08d04bf90e7f74bbb764e0c261c6e01cc1f930f7b81b63497d42'
source=p.parent/'g3-private-backend-validation/negative-inputs/alias.ll'
patch=p.parent/'g3-private-backend-design/buffer-call-proof-v4.patch'
actual=llvm/'llvm-project-patched/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp'
text=actual.read_text();body=text.split('void GCNPassConfig::addPostRegAlloc() {',1)[1].split('\n}',1)[0]
guard='if (ProveBufferCalls && TM->Options.EnableIPRA)\n    report_fatal_error("experimental buffer-call proof requires -enable-ipra=false");'
assert guard in body and body.index(guard)<body.index('addPass(')
assert body.index(guard)<body.index('addPass(createAMDGPUProveBufferCallsPass())')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
out=p/'runs-v1';out.mkdir(exist_ok=False);cases=[];commands=[]
expected_error='LLVM ERROR: experimental buffer-call proof requires -enable-ipra=false'
configs=[('baseline-default','baseline',[],False,False),('baseline-ipra-off','baseline',['-enable-ipra=false'],False,False),('feature-off-default','patched',[],False,False),('feature-off-ipra-off','patched',['-enable-ipra=false'],False,False),('proof-default','patched',[],True,True),('proof-ipra-on','patched',['-enable-ipra=true'],True,True),('proof-ipra-off','patched',['-enable-ipra=false'],True,False)]
try:
 for target in ('gfx1013','gfx1030'):
  d=out/target;d.mkdir()
  for name,compiler,flags,proof,must_fail in configs:
   folder=d/name;folder.mkdir();base=[str(compilers[compiler]),'-mtriple=amdgcn-amd-amdhsa','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-O2','-verify-machineinstrs',*flags]
   if proof:base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
   record={'target':target,'configuration':name,'expected_guard_failure':must_fail,'stages':{}};cases.append(record)
   for kind,ext in ([('asm','s')] if must_fail else [('asm','s'),('obj','o')]):
    output=folder/('output.'+ext);cmd=base+['-filetype='+kind,str(source),'-o',str(output)]
    run=subprocess.run(cmd,capture_output=True,text=True);(folder/(kind+'.log')).write_text(run.stdout+run.stderr)
    commands.append({'command':cmd,'exit_code':run.returncode});record['stages'][kind]=run.returncode
    if must_fail:
     assert run.returncode!=0 and run.stderr.splitlines()[0]==expected_error,(name,run.returncode,run.stderr)
     assert 'buffer-call-proof inventory' not in run.stderr and (not output.exists() or output.stat().st_size==0)
    else:
     assert run.returncode==0,(name,run.returncode,run.stderr)
     assert output.stat().st_size>0
     if proof:record[kind+'_diagnostics']=check_diagnostics(run.stderr,source.read_text(),{'negative':'proven','safe':'proven'},['negative','safe'])
   if not must_fail:
    assembly=(folder/'output.s').read_text();root=re.search(r'(?ms)^negative:.*?(?=^\.Lfunc_end)',assembly);assert root
    assert root[0].count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_')==(0 if proof else 2)
   record['files']={str(f.relative_to(out)):sha(f) for f in folder.iterdir() if f.is_file()}
  for current,baseline in [('feature-off-default','baseline-default'),('feature-off-ipra-off','baseline-ipra-off')]:
   for file in ('output.s','output.o'):assert (d/current/file).read_bytes()==(d/baseline/file).read_bytes(),(target,current,file,'feature-off changed')
 assert all(sha(path)==identities[name] for name,path in compilers.items())
 report={'guard_regressions_passed':True,'feature_off_each_configuration_byte_identical':True,'cases':cases,'commands':commands,'compiler_sha256':identities,'source_sha256':sha(source),'patch_sha256':sha(patch),'actual_target_machine_source_sha256':sha(actual),'recipe_sha256':sha(Path(__file__)),'accepted_compiler':False,'hardware_validated':False,'limits':['This guard deliberately checks effective IPRA configuration, not CLI token spelling. AMDGPU defaults IPRA on.','Explicit IPRA-off proof success is one alias-canonicalized fixture on two targets. Full freshly paired IPRA-off acceptance, resources and semantics remain unqualified.']}
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS14configurations',sha(out/'result.json'))
except Exception as error:
 (out/'failure.json').write_text(json.dumps({'error':repr(error),'cases':cases,'commands':commands,'compiler_sha256':identities},indent=2)+'\n');raise
