import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_llm_native_dense_layer_test import capture_inputs,sha
class Binding(unittest.TestCase):
 def test_binding_and_drift(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'reference.bin';b=root/'reference.json';a.write_bytes(b'raw');b.write_bytes(b'index');oracle={'sha256':sha(a),'manifest_sha256':sha(b)}
   self.assertEqual(len(capture_inputs([a,b],root,oracle)),2)
   a.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'validated reference changed'):capture_inputs([a,b],root,oracle)
 def test_missing_required_input(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileNotFoundError):capture_inputs([Path(d)/'missing'],Path(d),{})
if __name__=='__main__':unittest.main()
