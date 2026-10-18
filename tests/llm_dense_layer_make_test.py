import os,subprocess,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Delivery(unittest.TestCase):
 def make(self,*args):
  return subprocess.run(['make','-f',str(ROOT/'Makefile'),'JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x',*args,'llm-dense-layer-tests'],cwd=ROOT,env={k:v for k,v in os.environ.items() if k not in ('MAKEFLAGS','MFLAGS','MAKEOVERRIDES','LLM_DENSE_LAYER_OUTPUT_DIR','LLM_CPU_BUILD_DIR','PYTHON','MAKE')},text=True,capture_output=True,timeout=20)
 def test_dryrun(self):
  p=self.make('-n','MAKE=echo','PYTHON=echo','LLM_CPU_BUILD_DIR=/tmp/dense selected')
  self.assertEqual(p.returncode,0,p.stderr)
  self.assertIn('--numeric',p.stdout)
  self.assertIn('--output "results/llm-dense-layer"',p.stdout)
  self.assertIn('--cpu-build "/tmp/dense selected"',p.stdout)
  self.assertIn('if ! test -f "/tmp/dense selected/receipt.json"',p.stdout)
 def test_parent_make_options_are_isolated(self):
  with patch.dict(os.environ,LLM_DENSE_LAYER_OUTPUT_DIR='/tmp/exported-output',LLM_CPU_BUILD_DIR='/tmp/exported-cpu',MAKEFLAGS='-- LLM_DENSE_LAYER_OUTPUT_DIR=/tmp/parent-output',MFLAGS='--silent',MAKEOVERRIDES='LLM_DENSE_LAYER_OUTPUT_DIR=/tmp/parent-output'):
   self.test_dryrun()
 def test_receipt_reuses(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'receipt.json').write_text('{}')
   p=self.make('PYTHON=echo','MAKE=false','LLM_CPU_BUILD_DIR='+d)
   self.assertEqual(p.returncode,0,p.stderr)
   self.assertIn('tools/build_llm_dense_layer_test.py',p.stdout)
 def test_missing_receipt_requests_cpu(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.make('PYTHON=echo','MAKE=echo','LLM_CPU_BUILD_DIR='+d)
   self.assertEqual(p.returncode,0,p.stderr)
   self.assertIn('llm-cpu-reference LLM_CPU_BUILD_DIR='+d,p.stdout)
if __name__=='__main__':unittest.main()
