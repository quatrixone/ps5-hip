import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_q4_reference as b

class Builder(unittest.TestCase):
 def test_exact_asset_size_and_identity_drift(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'asset';p.write_bytes(b'abcd');self.assertEqual(b.exact(p,4),b'abcd')
   for size in (3,5):
    with self.assertRaises(ValueError):b.exact(p,size)
   inputs={};b.bind(inputs,p);p.write_bytes(b'abce')
   with self.assertRaises(ValueError):b.bind(inputs,p)
   with self.assertRaises(ValueError):b.unchanged(inputs)
   p.unlink()
   with self.assertRaises(ValueError):b.exact(p,4)
 def test_wrong_model_rejects(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'evidence').mkdir();p=root/'model';p.write_bytes(b'model')
   lock=root/'evidence/offline-llm-model-conversion.json';lock.write_text(json.dumps({'outputs':{'tinyllama-q4_0.gguf':{'bytes':5,'sha256':b.sha(p)}}}))
   with patch.object(b,'ROOT',root):
    b.verify_model(p);p.write_bytes(b'other')
    with self.assertRaises(ValueError):b.verify_model(p)
    p.write_bytes(b'shorter')
    with self.assertRaises(ValueError):b.verify_model(p)
 def test_fresh_output_and_finite_outputs(self):
  with tempfile.TemporaryDirectory() as d,patch.object(b,'ROOT',Path(d)):
   root=Path(d);out=root/'results/old';out.mkdir(parents=True)
   with self.assertRaises(ValueError):b.build(root,root,root,out)
   with self.assertRaises(ValueError):b.build(root,root,root,root/'elsewhere')
  for value in (float('nan'),float('inf')):
   with self.assertRaises(ValueError):b.pack([value],'d')

if __name__=='__main__':unittest.main()
