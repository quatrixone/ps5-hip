"""Synthetic subprocess-status tests only; never claim a real LLVM invocation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib,hashlib,io,json,subprocess,tempfile
import run_v2 as runner

out=Path(__file__).resolve().parent;cases=[]
def prepare(root):
 compiler=root/'synthetic-compiler';compiler.write_text('not an executable; subprocess always mocked')
 source=root/'test.mir';source.write_text('diagnostic only')
 matrix={'cases':[{'name':'case'+str(i),'kind':'unchanged','target':'gfx1013','file':'test.mir','sha256':runner.sha(source),'expected_root_proof':'proven'} for i in range(10)]}
 (root/'matrix.json').write_text(json.dumps(matrix));return compiler,matrix

def fake_success(cmd,**kwargs):
 output=Path(cmd[cmd.index('-o')+1])
 output.write_text('unverified synthetic MIR' if '-stop-before=prologepilog' in cmd else 'stress:\n s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO)\n s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI)\n.Lfunc_end0:\n')
 return subprocess.CompletedProcess(cmd,0,'','synthetic stage, no compiler executed')

for failed_stage,exit_code in [('pre-pei',7),('assembly',-11)]:
 with tempfile.TemporaryDirectory(dir=out) as temporary:
  root=Path(temporary);compiler,_=prepare(root)
  def fake(cmd,**kwargs):
   stage='pre-pei' if '-stop-before=prologepilog' in cmd else 'assembly'
   if stage==failed_stage:return subprocess.CompletedProcess(cmd,exit_code,'synthetic stdout','synthetic failure diagnostics')
   return fake_success(cmd,**kwargs)
  with patch.object(runner.subprocess,'run',side_effect=fake),contextlib.redirect_stderr(io.StringIO()):
   result=runner.run(SimpleNamespace(compiler=compiler,mode='baseline',run_label=None),root)
  folder=root/'runs-v2/baseline';failure=json.loads((folder/'failure.json').read_text())
  assert result!=0 and not (folder/'result.json').exists() and (folder/'partial.json').exists()
  assert failure['cases'][0]['stages'][failed_stage]==exit_code and not failure['all_controls_passed']
  assert 'synthetic failure diagnostics' in (folder/'case0'/(failed_stage+'.log')).read_text()
  cases.append({'case':failed_stage+'-failure','exit_code_preserved':exit_code,'runner_nonzero':True,'partial_and_logs_preserved':True})

for mode in ['patched-off','patched-on']:
 with tempfile.TemporaryDirectory(dir=out) as temporary:
  root=Path(temporary);compiler,_=prepare(root)
  with patch.object(runner.subprocess,'run',side_effect=AssertionError('must not invoke compiler')) as process,contextlib.redirect_stderr(io.StringIO()):
   result=runner.run(SimpleNamespace(compiler=compiler,mode=mode,run_label=None),root)
  assert result!=0 and process.call_count==0
  cases.append({'case':mode+'-missing-baseline','runner_nonzero':True,'compiler_calls':0})

with tempfile.TemporaryDirectory(dir=out) as temporary:
 root=Path(temporary);compiler,matrix=prepare(root)
 with patch.object(runner.subprocess,'run',side_effect=fake_success) as process,contextlib.redirect_stdout(io.StringIO()):
  assert runner.run(SimpleNamespace(compiler=compiler,mode='baseline',run_label=None),root)==0
 assert process.call_count==20
 runner.require_complete(root,'baseline',matrix)
 report_path=root/'runs-v2/baseline/result.json';original=report_path.read_text();report=json.loads(original)
 for mutation in ['missing-case','failed-stage','matrix','false-success','artifact']:
  modified=json.loads(original)
  if mutation=='missing-case':modified['cases'].pop()
  if mutation=='failed-stage':modified['cases'][0]['stages']['assembly']=9
  if mutation=='matrix':modified['matrix_sha256']='0'*64
  if mutation=='false-success':modified['all_controls_passed']=False
  if mutation=='artifact':modified['cases'][0]['files']['output.s']='0'*64
  report_path.write_text(json.dumps(modified))
  with patch.object(runner.subprocess,'run',side_effect=AssertionError('must not invoke compiler')) as process,contextlib.redirect_stderr(io.StringIO()):
   result=runner.run(SimpleNamespace(compiler=compiler,mode='patched-off',run_label='patched-off-'+mutation),root)
  assert result!=0 and process.call_count==0
  cases.append({'case':'bad-baseline-'+mutation,'runner_nonzero':True,'compiler_calls':0})
 cases.append({'case':'complete-synthetic-baseline','stages':20,'runner_zero':True,'actual_compiler_executed':False})

report={'actual_compiler_executed':False,'cases':cases,'runner_sha256':runner.sha(out/'run_v2.py'),'test_sha256':runner.sha(Path(__file__))}
dest=out/'run-v2-synthetic-tests.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(cases),'synthetic cases',runner.sha(dest))
