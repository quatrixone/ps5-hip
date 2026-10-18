import ast,re,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Delivery(unittest.TestCase):
 def make(self,*args):
  return subprocess.run(['make','-f',str(ROOT/'Makefile'),'JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x',*args,'llm-dense-projection-test'],cwd=ROOT,text=True,capture_output=True,timeout=20)
 def test_exports(self):
  tree=ast.parse((ROOT/'tools/build_llm_layer_fixture.py').read_text())
  expected=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYMBOLS' for t in n.targets))
  admission=re.findall(r'\{"(layer_\w+)"', (ROOT/'tests/llm_layer_admission.cpp').read_text())
  self.assertEqual(set(expected),set(admission));self.assertEqual(len(expected),16)
  for enabled in [False,True]:
   cmd=['clang++-18','-E','-P','-x','c++']+(['-DPS5HIP_ENABLE_DENSE_F16=1'] if enabled else [])+[str(ROOT/'kernels/llm_layer.hip')]
   out=subprocess.check_output(cmd,text=True,timeout=20)
   exports=re.findall(r'__attribute__\(\(global\)\)\s+void\s+(\w+)',out)
   self.assertCountEqual(exports,list(expected)+(['layer_dense_f16'] if enabled else []))
 def test_dryrun(self):
  p=self.make('-n','PYTHON=echo','LLM_CPU_BUILD_DIR=/tmp/dense selected')
  self.assertEqual(p.returncode,0,p.stderr)
  self.assertIn('--cpu-build "/tmp/dense selected"',p.stdout)
  self.assertIn('if ! test -f "/tmp/dense selected/receipt.json"',p.stdout)
 def test_receipt_reuses(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'receipt.json').write_text('{}')
   p=self.make('PYTHON=echo','MAKE=false','LLM_CPU_BUILD_DIR='+d)
   self.assertEqual(p.returncode,0,p.stderr)
   self.assertIn('tests/llm_dense_projection_test.py',p.stdout)
 def test_missing_receipt_requests_cpu(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.make('PYTHON=echo','MAKE=echo','LLM_CPU_BUILD_DIR='+d)
   self.assertEqual(p.returncode,0,p.stderr)
   self.assertIn('llm-cpu-reference LLM_CPU_BUILD_DIR='+d,p.stdout)
if __name__=='__main__':unittest.main()
