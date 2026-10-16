import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('native_build',Path(__file__).resolve().parents[1]/'tools/build_llm_native_client.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class NativeBuild(unittest.TestCase):
 def test_actual_locked_selection_has_unique_complete_source_units(self):
  units,models=b.selection(b.ROOT/'.deps/sources/llama-client')
  self.assertEqual(len(units),59);self.assertEqual(len(set(units)),59);self.assertEqual(len(models),161)
  self.assertEqual(len(units)+(len(models)+15)//16+1,71)
  self.assertIn('ggml-backend-reg.cpp',[p.name for p in units]);self.assertEqual(sum(p.name=='ggml-backend-reg.cpp' for p in units),1)
 def test_identity_drift_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'source';p.write_bytes(b'old');inputs={};b.bind(inputs,p);b.bind(inputs,p);p.write_bytes(b'new')
   with self.assertRaisesRegex(ValueError,'input changed'):b.bind(inputs,p)
 def test_fresh_scoped_output_required(self):
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaisesRegex(ValueError,'fresh'):b.build(Path(directory)/'outside')
  with self.assertRaisesRegex(ValueError,'fresh'):b.build(b.ROOT/'results')
 def test_cmake_path_injection_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   client=Path(directory);(client/'src').mkdir();(client/'src/CMakeLists.txt').write_text('set(LLAMA_CORE_SOURCES ../outside.cpp)')
   with self.assertRaisesRegex(ValueError,'unexpected core'):b.selection(client)
if __name__=='__main__':unittest.main()
