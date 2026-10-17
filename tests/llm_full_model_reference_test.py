import json,sys,tempfile,unittest,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_full_model_reference as ref
class Reference(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  for name,size in ref.FILES.items():(self.root/name).write_bytes(bytes(size))
  self.steps=[dict(run=r,position=p,token=(0,1,2,31999)[p%4],status=0,committed_position=p) for r,n in ((0,512),(1,2)) for p in range(n)]
  self.save()
 def save(self):(self.root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.steps))
 def test_complete(self):self.assertEqual(len(ref.validate(self.root)),168)
 def test_same_size_reset_mutation(self):
  p=self.root/'run-1-pos-0-layer-21.f32le';p.write_bytes(struct.pack('<f',1)+p.read_bytes()[4:])
  with self.assertRaisesRegex(ValueError,'reset boundary'):ref.validate(self.root)
 def test_nonfinite(self):
  p=self.root/'run-0-pos-255-logits.f32le';p.write_bytes(struct.pack('<f',float('nan'))+p.read_bytes()[4:])
  with self.assertRaisesRegex(ValueError,'nonfinite'):ref.validate(self.root)
 def test_missing_boundary(self):
  (self.root/'run-0-pos-511-layer-0.f32le').unlink()
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate(self.root)
 def test_wrong_commit(self):
  self.steps[256]['committed_position']=255;self.save()
  with self.assertRaisesRegex(ValueError,'committed progression'):ref.validate(self.root)
 def test_symlink(self):
  p=self.root/'run-0-pos-1-norm.f32le';p.unlink();p.symlink_to(self.root/'run-0-pos-0-norm.f32le')
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate(self.root)
if __name__=='__main__':unittest.main()
