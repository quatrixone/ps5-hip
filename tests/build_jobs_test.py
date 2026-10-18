"""Exercise explicit job budgets and real executor ordering without invoking compilers."""
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_jobs as policy
import build_public_hip as b


class JobPolicy(unittest.TestCase):
 def test_make_exports_overrides_and_validates_budget(self):
  with tempfile.TemporaryDirectory() as directory:
   makefile=Path(directory)/'Makefile'
   text=(ROOT/'Makefile').read_text().replace('tools/build_jobs.py',str(ROOT/'tools/build_jobs.py'))
   makefile.write_text(text+'\nprint-jobs:\n\t@$(PYTHON) '+str(ROOT/'tools/build_jobs.py')+'\n')
   env=dict(os.environ,PS5_HIP_BUILD_JOBS='5')
   # Test independent Make invocations without the parent recipe's overrides.
   for key in ('JOBS','MAKEFLAGS','MFLAGS','MAKEOVERRIDES'):env.pop(key,None)
   command=['make','--no-print-directory','-s','-f',str(makefile),'print-jobs']
   result=subprocess.run(command+['JOBS=3'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
   self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),'3')
   result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
   self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),'5')
   result=subprocess.run(command+['JOBS=0'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
   self.assertNotEqual(result.returncode,0);self.assertIn('integer from 1 to 64',result.stderr)
 def test_explicit_budget_and_invalid_values(self):
  for value in ('1','3','24','64'):
   with patch.dict(os.environ,{'PS5_HIP_BUILD_JOBS':value}),self.subTest(value=value):self.assertEqual(policy.build_jobs(),int(value))
  for value in ('','0','-1','65','1.5',' 2','2 ','02','+2','2 3','unlimited'):
   with patch.dict(os.environ,{'PS5_HIP_BUILD_JOBS':value}),self.subTest(value=value),self.assertRaises(ValueError):policy.build_jobs()
 def test_default_cpu_memory_and_unknown_limits(self):
  with patch.dict(os.environ,{},clear=True),patch.object(policy.os,'sched_getaffinity',return_value=set(range(24))):
   for available,expected in ((27*1024**2,8),(6*1024**2,2),(512*1024,1)):
    with patch.object(Path,'read_text',return_value=f'MemAvailable: {available} kB\n'):self.assertEqual(policy.build_jobs(),expected)
   with patch.object(Path,'read_text',side_effect=OSError):self.assertEqual(policy.build_jobs(),2)
  with patch.dict(os.environ,{},clear=True),patch.object(policy.os,'sched_getaffinity',side_effect=OSError),patch.object(policy.os,'cpu_count',return_value=None),patch.object(Path,'read_text',return_value='bad'):
   self.assertEqual(policy.build_jobs(),1)
 def test_public_workers_and_stable_object_order(self):
  for jobs in (1,3):
   with self.subTest(jobs=jobs),tempfile.TemporaryDirectory() as directory:
    output=Path(directory);source=output/'source';(source/'hipamd/src').mkdir(parents=True)
    for i in range(30):(source/'hipamd/src'/f'unit{i:02d}.cpp').touch()
    title_done=False;active=peak=started=0;lock=threading.Lock();barrier=threading.Barrier(jobs)
    def compile_unit(out,path,obj,flags,label,source_root=None):
     nonlocal title_done,active,peak,started
     if path.name=='public_hip_bootstrap_console_title.cpp':
      self.assertFalse(title_done);self.assertEqual(active,0);title_done=True
     else:
      self.assertTrue(title_done)
      with lock:active+=1;peak=max(peak,active);started+=1;number=started
      if number<=jobs:barrier.wait(timeout=5)
      with lock:active-=1
     return {'object':str(obj.relative_to(out))}
    with patch.dict(os.environ,{'PS5_HIP_BUILD_JOBS':str(jobs)}),patch.object(b,'compiler_flags',return_value=['compiler']),patch.object(b,'compile_bound_unit',side_effect=compile_unit),patch.object(b,'_plan_bound_unit',side_effect=lambda *a,**kw: ({},lambda _: compile_unit(*a,**kw))):
     records=b.compile_units(output,source)
    self.assertEqual(peak,jobs);self.assertEqual(len(records),76)
    self.assertEqual([int(Path(r['object']).name.split('-')[0]) for r in records],list(range(76)))
 def test_invalid_budget_and_title_failure_do_not_start_closure(self):
  with tempfile.TemporaryDirectory() as directory:
   output=Path(directory);source=output/'source';(source/'hipamd/src').mkdir(parents=True)
   for i in range(30):(source/'hipamd/src'/f'unit{i:02d}.cpp').touch()
   with patch.dict(os.environ,{'PS5_HIP_BUILD_JOBS':'0'}),patch.object(b,'compile_bound_unit') as compile_unit,self.assertRaises(ValueError):b.compile_units(output,source)
   compile_unit.assert_not_called()
   with patch.dict(os.environ,{'PS5_HIP_BUILD_JOBS':'3'}),patch.object(b,'compiler_flags',return_value=['compiler']),patch.object(b,'compile_bound_unit',side_effect=RuntimeError('title failed')) as compile_unit,self.assertRaisesRegex(RuntimeError,'title failed'):b.compile_units(output,source)
   self.assertEqual(compile_unit.call_count,1)
   self.assertEqual(compile_unit.call_args.args[1].name,'public_hip_bootstrap_console_title.cpp')


if __name__=='__main__':unittest.main()
