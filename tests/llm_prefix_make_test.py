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
 def test_prepared_source_override_and_default(self):
  text=self.run_make('llm-native-app')
  self.assertNotIn('--prepared-source-build',text)
  text=self.run_make('llm-native-app','LLM_NATIVE_APP_PREPARED_SOURCE_BUILD=build/prior source')
  line=next(line for line in text.splitlines() if 'tools/build_llm_native_app.py' in line)
  self.assertIn('--prepared-source-build "build/prior source"',line)
class LayerMake(unittest.TestCase):
 run_make=PrefixMake.run_make
 def test_layer_selected_aggregate_and_no_unrelated_phase(self):
  text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE=model-layer','LLM_NATIVE_APP_LAYER_FIXTURE=results/selected-layer')
  self.assertIn('llm-layer-assets LLM_LAYER_ASSETS_DIR="results/selected-layer"',text)
  self.assertIn('--layer-fixture "results/selected-layer"',text)
  self.assertNotIn('--prefix-fixture',text)
  for test in ('source_model_layer_freeze_test.py','source_layer_console_test.py','source_layer_authentication_test.py','native_layer_transcript_test.py','llm_layer_readbacks_test.py'):
   self.assertLess(text.index('tests/'+test),text.index('tools/build_llm_native_app.py'))
  for phase in ('vocab','model-gate','model-prefix'):
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE='+phase)
   self.assertNotIn('--layer-fixture',text)
 def test_layer_receipts_guard_creation_but_not_authentication(self):
  for target,variable,producer,validator in [('llm-layer-fixture','LLM_LAYER_FIXTURE_DIR','build_llm_layer_fixture.py','fixture_inputs'),('llm-layer-reference','LLM_LAYER_REFERENCE_DIR','build_llm_layer_reference.py','authenticate_layer_reference'),('llm-layer-assets','LLM_LAYER_ASSETS_DIR','build_public_hip_layer_assets.py','layer_inputs')]:
   text=self.run_make(target,variable+'=results/reused-layer')
   line=next(x for x in text.splitlines() if producer in x)
   self.assertTrue(line.startswith('if ! test -f "results/reused-layer/receipt.json"; then '))
   self.assertIn('import '+validator+'; '+validator,text)
  text=self.run_make('llm-layer-host-tests');self.assertIn('tools/build_llm_layer_host_test.py',text);self.assertNotIn('source_console.py',text)

class FullGraphMake(unittest.TestCase):
 run_make=PrefixMake.run_make
 def test_selected_inputs_and_authenticated_reuse(self):
  with tempfile.TemporaryDirectory() as folder:
   model=Path(folder)/'selected.gguf';model.touch()
   text=self.run_make('llm-full-graph-trace','LLM_FULL_GRAPH_TRACE_DIR=results/selected-full-graph','LLM_FULL_GRAPH_MODEL='+str(model))
   self.assertIn('--model "'+str(model)+'"',text)
  self.assertIn('if ! test -f "results/selected-full-graph/receipt.json"; then python3 tools/build_llm_full_graph_trace.py',text)
  self.assertIn('import authenticate_full_graph_trace; authenticate_full_graph_trace',text)
  self.assertIn('tests/llm_full_graph_trace_test.py',text)
  self.assertIn('tests/llm_full_graph_trace_data_test.py --raw "results/selected-full-graph/raw"',text)
  self.assertNotIn('source_console.py',text)
 def test_admission_consumes_selected_trace(self):
  text=self.run_make('llm-full-graph-admission','LLM_FULL_GRAPH_TRACE_DIR=results/selected-full-graph','LLM_FULL_GRAPH_ADMISSION_DIR=results/selected-admission')
  self.assertIn('tools/build_llm_full_graph_admission.py --trace "results/selected-full-graph" --output "results/selected-admission"',text)

class FullExecutionMake(unittest.TestCase):
 run_make=PrefixMake.run_make
 def test_host_targets_use_selected_inputs_without_native_build(self):
  for target,script,variable,folder in [('llm-full-host-tests','build_llm_full_backend_test.py','LLM_FULL_HOST_TEST_DIR','backend'),('llm-full-reference','build_llm_full_model_reference.py','LLM_FULL_REFERENCE_DIR','reference'),('llm-full-helper-tests','build_llm_full_backend_test.py','LLM_FULL_HELPER_TEST_DIR','helper')]:
   with tempfile.TemporaryDirectory() as directory:
    model=Path(directory)/'selected-full.gguf';model.touch()
    text=self.run_make(target,variable+'=results/selected-'+folder,'LLM_FULL_GRAPH_MODEL='+str(model))
    self.assertIn('--model "'+str(model)+'" --output "results/selected-'+folder+'"',text)
   self.assertIn('tools/'+script,text)
   self.assertNotIn('build_llm_native_app.py',text)
   self.assertNotIn('source_console.py',text)
 def test_full_native_phase_uses_its_own_aggregate(self):
  with tempfile.TemporaryDirectory() as directory:
   receipt=Path(directory)/'receipt.json';receipt.write_text('{"profile":"smoke"}')
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE=model-full','LLM_NATIVE_APP_FULL_FIXTURE='+directory)
  self.assertIn('--phase "model-full"',text)
  self.assertIn('--full-fixture "'+directory+'"',text)
  self.assertNotIn('--layer-fixture',text)
  self.assertNotIn('--prefix-fixture',text)
  self.assertLess(text.index('tests/public_hip_full_assets_test.py'),text.index('tools/build_llm_native_app.py'))
 def test_generation_assets_select_the_generation_oracle(self):
  text=self.run_make('llm-full-assets','LLM_FULL_PROFILE=generation','LLM_GENERATION_REFERENCE_DIR=results/selected-generation-reference')
  self.assertIn('llm-generation-reference',text);self.assertIn('--reference "results/selected-generation-reference" --profile "generation"',text);self.assertIn('results/llm-generation-assets',text);self.assertNotIn('then echo llm-full-reference',text)
  smoke=self.run_make('llm-full-assets');self.assertIn('--profile "smoke"',smoke);self.assertIn('results/llm-full-assets',smoke)
 def test_generation_app_checks_existing_profile_and_runs_its_tests(self):
  with tempfile.TemporaryDirectory() as directory:
   (Path(directory)/'receipt.json').write_text('{"profile":"generation"}')
   text=self.run_make('llm-native-app','LLM_NATIVE_APP_PHASE=model-full','LLM_FULL_PROFILE=generation','LLM_NATIVE_APP_FULL_FIXTURE='+directory)
  self.assertIn('requested full profile differs',text);self.assertIn('tests/native_generation_application_test.py',text);self.assertIn('tests/llm_generation_readbacks_test.py',text);self.assertIn('tests/source_generation_console_test.py',text)
 def test_numerical_target_preserves_selected_reference(self):
  with tempfile.TemporaryDirectory() as directory:
   receipt=Path(directory)/'receipt.json';receipt.write_text('{}')
   text=self.run_make('llm-full-numeric-tests','LLM_FULL_REFERENCE_DIR='+directory,'LLM_FULL_NUMERIC_TEST_DIR=results/selected-numeric')
  self.assertIn('--reference "'+directory+'" --output "results/selected-numeric"',text)
  self.assertIn('tests/llm_full_numeric_test.py',text)
  self.assertIn('tools/build_llm_full_numeric_test.py',text)
  self.assertNotIn('build_llm_native_app.py',text)
 def test_dispatch_target_uses_selected_trace(self):
  text=self.run_make('llm-full-dispatch-tests','LLM_FULL_GRAPH_TRACE_DIR=results/selected-full-graph','LLM_FULL_DISPATCH_TEST_DIR=results/selected-dispatch')
  self.assertIn('--graph "results/selected-full-graph/raw/run-0-pos-0.graph.json" --output "results/selected-dispatch"',text)
  self.assertIn('tools/build_llm_full_dispatch_test.py',text)

if __name__=='__main__':unittest.main()
