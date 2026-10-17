from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'tools'))
sys.path.append(str(HERE.parents[1]/'tools'))
import public_hip_layer_assets as a
from build_public_hip_layer_assets import build

class LayerAssets(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  for name in a.SOURCES:self.write(name,b'source')
  self.parents={};self.closures={}
  for kind in a.KINDS:
   folder=self.root/kind;self.parents[kind]=folder
   self.write(kind+'/receipt.json',json.dumps({'classification':a.KINDS[kind]}).encode())
   for name in a.FILES[kind]:self.write(kind+'/'+name,(kind+name).encode())
   self.closures[kind]={str(p):a.sha(p) for p in folder.iterdir()}
  self.reference=self.root/'reference';self.write('reference/receipt.json',b'{}');self.write('reference/run-1/value',b'oracle')
  model=self.write('model.gguf',b'model')
  self.refinputs={str(p):a.sha(p) for p in (model,self.reference/'receipt.json',self.reference/'run-1/value')}
  self.refdata={'model':{'path':str(model),'bytes':5,'sha256':a.sha(model)},'receipt':str(self.reference/'receipt.json'),
    'receipt_sha256':a.sha(self.reference/'receipt.json'),'reference':str(self.reference/'run-1')}
  self.addCleanup(patch.stopall);patch.object(a,'ROOT',self.root).start()
  self.module=patch.object(a,'validate_module',side_effect=lambda k,f:self.closures[k]).start()
  self.ref=patch.object(a,'validate_reference',side_effect=lambda f:(self.refdata,self.refinputs)).start()
 def write(self,name,data):
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
 def call(self):return a.assemble(self.parents,self.reference)
 def test_exact_six_and_external_model(self):
  out=self.root/'aggregate';result=build(self.parents,self.reference,out)
  self.assertEqual(set(result['assets']),{'model-layer/'+n+'.hsaco' for n in ('getrows','normalized','weighted','q8','q4','layer')})
  self.assertEqual(list(out.iterdir()),[out/'receipt.json']);self.assertEqual(result['reference'],str(self.reference))
  self.assertEqual(result['model'],self.refdata['model']);self.assertEqual(result,a.layer_inputs(out))
 def test_missing_parent(self):
  del self.parents['q4']
  with self.assertRaisesRegex(ValueError,'five'):self.call()
 def test_missing_module(self):
  (self.parents['q4']/'q8.hsaco').unlink()
  with self.assertRaisesRegex(ValueError,'module'):self.call()
 def test_wrong_module_classification(self):
  self.write('getrows/receipt.json',b'{"classification":"q4"}')
  with self.assertRaisesRegex(ValueError,'classification'):self.call()
 def test_validator_rejects_metadata(self):
  self.module.side_effect=ValueError('metadata mismatch')
  with self.assertRaisesRegex(ValueError,'metadata'):self.call()
 def test_unbound_module(self):
  del self.closures['q4'][str(self.parents['q4']/'q8.hsaco')]
  with self.assertRaisesRegex(ValueError,'module'):self.call()
 def test_unbound_parent_receipt(self):
  del self.closures['q4'][str(self.parents['q4']/'receipt.json')]
  with self.assertRaisesRegex(ValueError,'receipt'):self.call()
 def test_conflicting_identity(self):
  self.refinputs[str(self.parents['q4']/'q8.hsaco')]='0'*64
  with self.assertRaisesRegex(ValueError,'conflicting'):self.call()
 def test_stale_model(self):
  self.write('model.gguf',b'other')
  with self.assertRaisesRegex(ValueError,'changed'):self.call()
 def test_wrong_model_size(self):
  self.refdata['model']['bytes']=7
  with self.assertRaisesRegex(ValueError,'model'):self.call()
 def test_unbound_model(self):
  del self.refinputs[str(self.root/'model.gguf')]
  with self.assertRaisesRegex(ValueError,'model'):self.call()
 def test_reference_receipt_identity(self):
  self.refdata['receipt_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'receipt'):self.call()
 def test_reference_location(self):
  self.refdata['reference']=str(self.root/'elsewhere')
  with self.assertRaisesRegex(ValueError,'directory'):self.call()
 def test_changed_reference_artifact(self):
  self.write('reference/run-1/value',b'changed')
  with self.assertRaisesRegex(ValueError,'changed'):self.call()
 def test_mutation_during_validation(self):
  def mutate(folder):
   self.write(a.SOURCES[0],b'changed');return self.refdata,self.refinputs
  self.ref.side_effect=mutate
  with self.assertRaisesRegex(ValueError,'changed'):self.call()
 def test_altered_receipt_mapping(self):
  out=self.root/'aggregate';build(self.parents,self.reference,out)
  path=out/'receipt.json';data=json.loads(path.read_text());data['assets']['model-layer/q4.hsaco']=str(self.root/'model.gguf');path.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'altered'):a.layer_inputs(out)
 def test_stale_aggregate_source(self):
  out=self.root/'aggregate';build(self.parents,self.reference,out);self.write(a.SOURCES[0],b'new implementation')
  with self.assertRaisesRegex(ValueError,'stale'):a.layer_inputs(out)
 def test_symlink_module(self):
  path=self.parents['q4']/'q8.hsaco';contents=path.read_bytes();path.unlink();target=self.write('outside',contents);path.symlink_to(target)
  with self.assertRaisesRegex(ValueError,'module'):self.call()
 def test_no_overwrite(self):
  out=self.root/'aggregate';build(self.parents,self.reference,out)
  with self.assertRaises(FileExistsError):build(self.parents,self.reference,out)

if __name__=='__main__':unittest.main()
