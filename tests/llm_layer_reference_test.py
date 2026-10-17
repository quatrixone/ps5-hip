from pathlib import Path
import sys,json,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_layer_reference as r
class Reference(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.folder=self.root/'reference';self.folder.mkdir();self.model=self.root/'model';self.model.write_bytes(b'model')
  for number in (1,2):
   folder=self.folder/f'run-{number}';folder.mkdir()
   for name,size in r.contract.FILES.items():(folder/name).write_bytes(bytes(size))
   (self.folder/f'run-{number}.stdout').write_text(r.LINE)
  for name in ('compile.d','headers.d','ldd.stdout'):(self.folder/name).write_text('')
  self.data={'schema':1,'status':'PASS','classification':r.CLASSIFICATION,'cpu_build':str(self.root/'cpu'),'model':{'path':str(self.model),'bytes':5,'sha256':r.provenance.sha(self.model)},'repeat_bitwise':True,'metrics':r.validate_reference(self.folder/'run-1',r.LINE),'inputs':{str(self.model):r.provenance.sha(self.model)},'artifacts':r.inventory(self.folder),'native_executed':False,'gpu_executed':False}
  self.save()
  for target,kwargs in [('sources',{'return_value':[]})]:
   mock=patch.object(r,target,**kwargs);mock.start();self.addCleanup(mock.stop)
  for name,value in [('cpu_inputs',{}),('dependencies',set()),('runtime_inputs',{})]:
   mock=patch.object(r.provenance,name,return_value=value);mock.start();self.addCleanup(mock.stop)
 def save(self):(self.folder/'receipt.json').write_text(json.dumps(self.data))
 def test_valid_closure(self):
  data,inputs=r.authenticate_layer_reference(self.folder);self.assertEqual(data['reference'],str(self.folder/'run-1'));self.assertIn(str(self.folder/'receipt.json'),inputs);self.assertEqual(data['metrics']['files'],151)
 def test_missing_and_tampered_raw_rejected(self):
  p=next((self.folder/'run-1').iterdir());p.write_bytes(b'changed')
  with self.assertRaises(ValueError):r.authenticate_layer_reference(self.folder)
  p.unlink()
  with self.assertRaises(ValueError):r.authenticate_layer_reference(self.folder)
 def test_model_and_required_source_rejected(self):
  self.model.write_bytes(b'other')
  with self.assertRaises(ValueError):r.authenticate_layer_reference(self.folder)
  self.model.write_bytes(b'model')
  with patch.object(r,'sources',return_value=[Path(__file__)]),self.assertRaises(ValueError):r.authenticate_layer_reference(self.folder)
 def test_reset_and_completion_rejected(self):
  p=self.folder/'run-1/run-1-pos-0-out-00.f32le';p.write_bytes(b'\0\0\x80?'+bytes(p.stat().st_size-4))
  with self.assertRaisesRegex(ValueError,'reset'):r.validate_reference(self.folder/'run-1',r.LINE)
  with self.assertRaisesRegex(ValueError,'completion'):r.validate_reference(self.folder/'run-2','')
 def test_symlink_and_extra_inventory_rejected(self):
  (self.folder/'extra').symlink_to(self.model)
  with self.assertRaisesRegex(ValueError,'symlink'):r.authenticate_layer_reference(self.folder)
 def test_compiler_inventory_mismatch_rejected(self):
  with patch.object(r.provenance,'dependencies',side_effect=[set(),{'different'}]),self.assertRaisesRegex(ValueError,'compiler dependency'):r.authenticate_layer_reference(self.folder)
if __name__=='__main__':unittest.main()
