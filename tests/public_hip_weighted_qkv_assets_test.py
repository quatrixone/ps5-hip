from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import public_hip_weighted_qkv_assets as a
import build_llm_weighted_rmsnorm_fixture as weighted
import build_llm_q4_halfword_fixture as halfword
import verify_weighted_qkv_readbacks as verifier

class Assets(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  for name in ('tools/public_hip_weighted_qkv_assets.py','tests/public_hip_weighted_qkv_assets_test.py','tools/verify_weighted_qkv_readbacks.py','tests/weighted_qkv_readbacks_test.py'):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'source')
  self.w=self.root/'weighted';self.q=self.root/'halfword';self.r=self.root/'reference'
  for folder in (self.w,self.q,self.r):folder.mkdir()
  self.modules={}
  for folder,name in ((self.w,'weighted.hsaco'),(self.q,'q8.hsaco'),(self.q,'q4.hsaco')):
   p=folder/name;p.write_bytes(name.encode());self.modules[str(p)]=a.sha(p)
  self.data={};self.identities={}
  for name in [*a.REFERENCE_FILES.values(),'receipt.json']:
   p=self.r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());self.identities[str(p)]=a.sha(p)
   if name!='receipt.json':self.data[name]=p.read_bytes()
  self.addCleanup(patch.stopall);patch.object(a,'ROOT',self.root).start()
  patch.object(weighted,'fixture_inputs',return_value={str(self.w/'weighted.hsaco'):self.modules[str(self.w/'weighted.hsaco')]}).start()
  self.qmock=patch.object(halfword,'fixture_inputs',return_value={n:h for n,h in self.modules.items() if str(self.q) in n}).start()
  self.refmock=patch.object(verifier,'authenticate',return_value=(self.data,self.identities)).start()
 def call(self):return a.assets(self.w,self.q,self.r)
 def test_exact_mapping(self):
  result=self.call();self.assertEqual(len(result['assets']),10);self.assertEqual(set(result['assets']),{'weighted-qkv/'+n for n in ('weighted.hsaco','q8.hsaco','q4.hsaco',*a.REFERENCE_FILES)})
 def test_required_variant_and_closure(self):
  self.qmock.side_effect=ValueError('wrong halfword classification')
  with self.assertRaisesRegex(ValueError,'halfword'):self.call()
  self.qmock.side_effect=None;self.qmock.return_value={}
  with self.assertRaisesRegex(ValueError,'authenticated closure'):self.call()
 def test_missing_extra_and_changed_assets(self):
  self.data['unexpected']=b'x'
  with self.assertRaisesRegex(ValueError,'seven'):self.call()
  del self.data['unexpected'];p=self.r/'input.f32le';p.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.call()
 def test_conflicting_and_mutating_implementation(self):
  self.identities[str(self.w/'weighted.hsaco')]='0'*64
  with self.assertRaisesRegex(ValueError,'conflicting'):self.call()
  del self.identities[str(self.w/'weighted.hsaco')]
  def changed(_):
   (self.root/'tools/public_hip_weighted_qkv_assets.py').write_bytes(b'new code');return self.data,self.identities
  self.refmock.side_effect=changed
  with self.assertRaisesRegex(ValueError,'changed during'):self.call()
 def test_symlink_asset_rejected(self):
  p=self.r/'input.f32le';data=p.read_bytes();p.unlink();target=self.root/'outside';target.write_bytes(data);p.symlink_to(target)
  with self.assertRaises(ValueError):self.call()

if __name__=='__main__':unittest.main()
