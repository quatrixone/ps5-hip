import os,subprocess,tempfile,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'dependencies.json').is_file())
MAKEFILE=Path(os.environ.get('PS5HIP_PREFIX_TEST_MAKEFILE',ROOT/'Makefile'))
class PrefixMake(unittest.TestCase):
 def run_make(self,target,*variables):
  result=subprocess.run(['make','--no-print-directory','-n','-f',str(MAKEFILE),'-o','deps-build','-o','deps-fetch','-o','llm-deps-fetch',target,'MAKE=echo','JOBS=1','UFS_REV=test','ZLIB_VERSION=test','TITLE_ID=PPSA99014',*variables],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr);return result.stdout
 def test_prefix_app_prepares_selected_aggregate(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'aggregate'
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE=model-prefix','LLM_PREFIX_FIXTURE_DIR='+str(path))
   self.assertIn('llm-prefix-assets LLM_PREFIX_FIXTURE_DIR="'+str(path)+'"',text)
   self.assertIn('--prefix-fixture "'+str(path)+'"',text)
   self.assertLess(text.index('llm-prefix-assets'),text.index('tools/build_llm_native_app.py'))
   custom=Path(temp)/'selected'
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE=model-prefix','LLM_PREFIX_FIXTURE_DIR='+str(path),'LLM_NATIVE_APP_PREFIX_FIXTURE='+str(custom))
   self.assertIn('--prefix-fixture "'+str(custom)+'"',text)
 def test_nonprefix_app_does_not_prepare_aggregate(self):
  for phase in ('vocab','model-gate'):
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE='+phase)
   self.assertNotIn('llm-prefix-assets',text);self.assertNotIn('--prefix-fixture',text)
 def test_reuse_is_guarded_and_validated(self):
  cases=[('llm-getrows-fixture','LLM_GETROWS_FIXTURE_DIR','tools/build_llm_q4_fixture.py --get-rows-only','fixture_inputs'),('llm-prefix-reference','LLM_PREFIX_REFERENCE_DIR','tools/build_llm_prefix_reference.py','authenticate_prefix_reference'),('llm-prefix-assets','LLM_PREFIX_FIXTURE_DIR','tools/build_public_hip_prefix_assets.py','prefix_inputs')]
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp);(folder/'receipt.json').write_text('{}');model=folder/'model.gguf';model.touch()
   for target,variable,producer,validator in cases:
    text=self.run_make(target,variable+'='+str(folder),'LLM_PREFIX_MODEL='+str(model))
    line=next(line for line in text.splitlines() if producer in line)
    self.assertTrue(line.startswith('if ! test -f "'+str(folder)+'/receipt.json"; then '),line)
    self.assertIn('; fi',line)
    consumer=next(line for line in text.splitlines() if 'from ' in line and validator+';' in line)
    self.assertTrue(consumer.startswith('python3 -c '));self.assertIn(str(folder),consumer)
 def test_reference_selected_model_and_host_target(self):
  with tempfile.TemporaryDirectory() as temp:
   model=Path(temp)/'external.gguf';model.touch()
   text=self.run_make('llm-prefix-reference','LLM_PREFIX_MODEL='+str(model))
   self.assertIn('--model "'+str(model)+'"',text)
  text=self.run_make('llm-prefix-host-tests')
  self.assertIn('tools/build_llm_prefix_host_tests.py',text)
  self.assertNotIn('source_console.py',text)
 def test_regressions_run_on_relevant_targets(self):
  for target,tests in (('llm-getrows-fixture',('llm_getrows_validation_test.py',)),('llm-prefix-host-tests',('llm_prefix_readbacks_test.py',))):
   text=self.run_make(target)
   for test in tests:self.assertIn('python3 tests/'+test,text)
  tests=('source_model_prefix_freeze_test.py','source_prefix_console_test.py','source_prefix_authentication_test.py')
  for phase in ('vocab','model-gate','model-prefix'):
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE='+phase)
   for test in tests:
    if phase=='model-prefix':
     self.assertIn('python3 tests/'+test,text)
     self.assertLess(text.index('tests/'+test),text.index('tools/build_llm_native_app.py'))
    else:self.assertNotIn('tests/'+test,text)
if __name__=='__main__':unittest.main()
