import os,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Delivery(unittest.TestCase):
 def run_make(self,*args):
  env={k:v for k,v in os.environ.items() if k not in ('MAKEFLAGS','MFLAGS','MAKEOVERRIDES','LLM_CPU_BUILD_DIR','LLM_DENSE_REFERENCE_DIR','LLM_NATIVE_DENSE_TEST_DIR','PYTHON','MAKE')}
  return subprocess.run(['make','-f',str(ROOT/'Makefile'),'JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x','PYTHON=echo',*args,'llm-native-dense-layer-tests'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
 def test_dryrun(self):
  p=self.run_make('-n','LLM_CPU_BUILD_DIR=/tmp/cpu selected','LLM_DENSE_REFERENCE_DIR=/tmp/ref selected')
  self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--cpu-build "/tmp/cpu selected"',p.stdout);self.assertIn('--reference "/tmp/ref selected"',p.stdout);self.assertIn('--output "results/llm-native-dense-helper"',p.stdout);self.assertNotIn('llm-cpu-reference',p.stdout)
 def test_missing_prerequisite_rejects(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.run_make('LLM_CPU_BUILD_DIR='+d);self.assertNotEqual(p.returncode,0);self.assertIn('receipt required',p.stderr)
 def test_existing_inputs_reused(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'receipt.json').write_text('{}');Path(d,'reference.json').write_text('{}')
   p=self.run_make('MAKE=false','LLM_CPU_BUILD_DIR='+d,'LLM_DENSE_REFERENCE_DIR='+d);self.assertEqual(p.returncode,0,p.stderr);self.assertIn('build_llm_native_dense_layer_test.py',p.stdout)
if __name__=='__main__':unittest.main()
