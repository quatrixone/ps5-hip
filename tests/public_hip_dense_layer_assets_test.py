import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import public_hip_dense_layer_assets as a
VALIDATE_MODULE=a.validate_module
REPO=a.ROOT

class DenseAssets(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.parents={};self.closures={}
  for name in a.SOURCES:self.write(name,b'source')
  for kind,names in a.FILES.items():
   folder=self.root/kind;self.parents[kind]=folder
   for name in ('receipt.json',*names):self.write(kind+'/'+name,b'{}')
   expected='source-only-bounded-dense-f16-layer-fixture' if kind=='layer' else a.KINDS[kind]
   self.write(kind+'/receipt.json',json.dumps({'classification':expected}).encode())
   self.closures[kind]={str(p):a.sha(p) for p in folder.iterdir()}
  self.reference=self.root/'reference'
  for name in ('receipt.json','reference.bin','reference.json'):self.write('reference/'+name,b'{}')
  self.refinputs={str(p):a.sha(p) for p in self.reference.iterdir()}
  self.addCleanup(patch.stopall);patch.object(a,'ROOT',self.root).start()
  self.module=patch.object(a,'validate_module',side_effect=lambda k,f:self.closures[k]).start()
  self.ref=patch.object(a,'validate_reference',side_effect=lambda f:{'inputs':self.refinputs}).start()
 def write(self,name,data):
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
 def call(self):return a.assemble(self.parents,self.reference)
 def test_exact_assets_and_reauthentication(self):
  out=self.root/'aggregate';data=a.build(self.parents,self.reference,out)
  self.assertEqual(data,a.dense_inputs(out));self.assertEqual(data['classification'],a.CLASSIFICATION)
  self.assertEqual(set(data['assets']),{'dense-layer/'+n+'.hsaco' for n in ('getrows','normalized','weighted','q8','q4','layer')})
  self.assertNotIn('model',data);self.assertEqual(list(out.iterdir()),[out/'receipt.json'])
  with self.assertRaises(FileExistsError):a.build(self.parents,self.reference,out)
  receipt=out/'receipt.json';altered=json.loads(receipt.read_text());altered['assets']['dense-layer/q4.hsaco']='wrong';receipt.write_text(json.dumps(altered))
  with self.assertRaisesRegex(ValueError,'altered'):a.dense_inputs(out)
 def test_missing_or_unbound_modules(self):
  del self.parents['q4']
  with self.assertRaisesRegex(ValueError,'five'):self.call()
  self.parents['q4']=self.root/'q4';del self.closures['q4'][str(self.root/'q4/q8.hsaco')]
  with self.assertRaisesRegex(ValueError,'module'):self.call()
 def test_reference_conflict_and_mutation(self):
  self.refinputs[str(self.root/'q4/q8.hsaco')]='0'*64
  with self.assertRaisesRegex(ValueError,'conflicting'):self.call()
  del self.refinputs[str(self.root/'q4/q8.hsaco')]
  def mutate(folder):
   self.write(a.SOURCES[0],b'changed');return {'inputs':self.refinputs}
  self.ref.side_effect=mutate
  with self.assertRaisesRegex(ValueError,'changed'):self.call()
 def test_unbound_reference_receipt(self):
  del self.refinputs[str(self.reference/'receipt.json')]
  with self.assertRaisesRegex(ValueError,'reference receipt'):self.call()
 def test_rejected_module_propagates(self):
  self.module.side_effect=ValueError('wrong dense profile')
  with self.assertRaisesRegex(ValueError,'dense profile'):self.call()
 def test_wrong_module_classification(self):
  self.write('getrows/receipt.json',b'{"classification":"diagnostic-q4-halfword-fixture"}')
  with self.assertRaisesRegex(ValueError,'classification'):self.call()
 def test_new_receipt_mutation(self):
  output=self.root/'aggregate';original=a.sha
  def mutate(path):
   if Path(path)==output/'receipt.json':Path(path).write_bytes(b'changed')
   return original(path)
  with patch.object(a,'sha',side_effect=mutate):
   with self.assertRaisesRegex(ValueError,'new aggregate receipt changed'):a.build(self.parents,self.reference,output)
 def test_dense_module_explicit_profile(self):
  import build_llm_layer_fixture as fixture
  with patch.object(fixture,'fixture_inputs',return_value={}) as validator:
   VALIDATE_MODULE('layer',self.root/'layer')
   validator.assert_called_once_with(self.root/'layer','dense-f16')
 def test_make_reuses_selected_inputs(self):
  folder=self.root/'selected inputs';folder.mkdir();(folder/'receipt.json').write_text('{}')
  variables=('LLM_RMSNORM_FIXTURE_DIR','LLM_GETROWS_FIXTURE_DIR','LLM_WEIGHTED_RMSNORM_FIXTURE_DIR','LLM_Q4_FIXTURE_DIR','LLM_Q4_HALFWORD_FIXTURE_DIR','LLM_DENSE_LAYER_FIXTURE_DIR','LLM_DENSE_REFERENCE_DIR','LLM_DENSE_ASSETS_DIR')
  env={k:v for k,v in os.environ.items() if k not in ('MAKEFLAGS','MFLAGS','MAKEOVERRIDES','MAKE','PYTHON')}
  argv=['make','llm-dense-layer-assets','PYTHON=echo','MAKE=false','JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x',*[n+'='+str(folder) for n in variables]]
  run=subprocess.run(argv,cwd=REPO,env=env,capture_output=True,text=True,timeout=20)
  self.assertEqual(run.returncode,0,run.stderr);self.assertIn('dense_inputs',run.stdout)
  run=subprocess.run([*argv,'-n'],cwd=REPO,env=env,capture_output=True,text=True,timeout=20)
  self.assertEqual(run.returncode,0,run.stderr);self.assertIn('--reference "'+str(folder)+'"',run.stdout)

if __name__=='__main__':unittest.main()
