from pathlib import Path
import json,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_prefix_reference as r

class Authentication(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.folder=self.root/'reference';self.folder.mkdir()
  self.source=self.root/'producer.py';self.source.write_bytes(b'source');self.model=self.root/'model.gguf';self.model.write_bytes(b'model')
  self.inputs={str(p):r.provenance.sha(p) for p in (self.source,self.model)}
  for name in ('compile.d','headers.d','ldd.stdout','run-1.stdout','run-2.stdout'):(self.folder/name).write_bytes(b'fixture')
  for n in (1,2):
   run=self.folder/f'run-{n}';run.mkdir();(run/'tensor').write_bytes(b'same')
  self.data={'schema':1,'status':'PASS','classification':r.CLASSIFICATION,'native_executed':False,'gpu_executed':False,'tokens':list(r.TOKENS),'nodes':6,'repeat_bitwise':True,'inputs':self.inputs,'artifacts':r.inventory(self.folder),'model':{'path':str(self.model),'bytes':5,'sha256':r.provenance.sha(self.model)},'cpu_build':str(self.root/'cpu'),'metrics':{}}
  self.save();self.addCleanup(patch.stopall)
  patch.object(r,'sources',return_value=[self.source]).start()
  patch.object(r.provenance,'cpu_inputs',return_value={str(self.model):r.provenance.sha(self.model)}).start()
  patch.object(r.provenance,'dependencies',return_value=set()).start()
  patch.object(r.provenance,'runtime_inputs',return_value={}).start()
  self.semantic=patch.object(r,'validate_reference',return_value={}).start()
 def save(self): (self.folder/'receipt.json').write_text(json.dumps(self.data))
 def call(self):return r.authenticate_prefix_reference(self.folder)
 def test_complete_closure(self):
  data,inputs=self.call();self.assertEqual(data['model'],self.data['model']);self.assertEqual(data['reference'],str(self.folder/'run-1'))
  self.assertEqual(inputs[data['receipt']],data['receipt_sha256']);self.assertIn(str(self.folder/'run-1/tensor'),inputs)
 def test_changed_source(self):
  self.source.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'source missing or stale'):self.call()
 def test_omitted_source(self):
  del self.inputs[str(self.source)];self.save()
  with self.assertRaisesRegex(ValueError,'source missing or stale'):self.call()
 def test_missing_tampered_extra_artifacts(self):
  for mutation in ('missing','tampered','extra'):
   case=Authentication();case.setUp()
   try:
    path=case.folder/'run-1/tensor'
    if mutation=='missing':path.unlink()
    elif mutation=='tampered':path.write_bytes(b'bad')
    else:(case.folder/'extra').write_bytes(b'extra')
    with self.assertRaisesRegex(ValueError,'artifact identity or inventory'):case.call()
   finally:case.doCleanups()
 def test_wrong_model(self):
  self.model.write_bytes(b'other')
  with self.assertRaisesRegex(ValueError,'model identity'):self.call()
 def test_gpu_claim_rejected(self):
  self.data['gpu_executed']=True;self.save()
  with self.assertRaisesRegex(ValueError,'contract'):self.call()
 def test_repeated_outputs_differ(self):
  path=self.folder/'run-2/tensor';path.write_bytes(b'changed');self.data['artifacts']['run-2/tensor']=r.provenance.sha(path);self.save()
  with self.assertRaisesRegex(ValueError,'repetitions'):self.call()
 def test_missing_cpu_closure(self):
  with patch.object(r.provenance,'cpu_inputs',return_value={'/omitted':'0'*64}),self.assertRaisesRegex(ValueError,'CPU/model'):self.call()
 def test_numerical_rejection_propagates(self):
  self.semantic.side_effect=ValueError('embedding decode differs')
  with self.assertRaisesRegex(ValueError,'embedding decode'):self.call()
 def test_metrics_receipt_tampering(self):
  self.data['metrics']={'wrong':True};self.save()
  with self.assertRaisesRegex(ValueError,'metrics'):self.call()
 def test_receipt_mutation_during_validation(self):
  def mutate(*args):
   self.data['extra']='changed';self.save();return {}
  self.semantic.side_effect=mutate
  with self.assertRaisesRegex(ValueError,'receipt changed'):self.call()
 def test_symlink_artifact(self):
  (self.folder/'link').symlink_to(self.model)
  with self.assertRaisesRegex(ValueError,'symlink'):self.call()

class SourceScope(unittest.TestCase):
 def test_unrelated_imports_do_not_change_sources(self):
  before=r.sources();extra=types.ModuleType('unrelated_consumer');extra.__file__=str(r.ROOT/'tools/unrelated.py')
  with patch.dict(sys.modules,unrelated_consumer=extra):self.assertEqual(r.sources(),before)
  self.assertIn(r.ROOT/'tests/llm_prefix_reference_test.py',before)
  self.assertIn(r.ROOT/'ports/ggml/model_prefix.h',before)
 def test_no_ignored_paths_in_producer_contract(self):
  self.assertEqual(r.HARNESS,r.ROOT/'tests/llm_actual_prefix_reference.cpp')
  self.assertEqual(r.NUMERICAL,r.ROOT/'tools/verify_llm_prefix_readbacks.py')
 def test_capture_contract_rejected_before_files(self):
  with self.assertRaisesRegex(ValueError,'four-token/six-node'):r.validate_reference(Path('/not-used'),'')
 def test_fresh_results_only(self):
  with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'fresh results'):r.build(Path(folder),Path(folder),Path(folder))

if __name__=='__main__':unittest.main()
