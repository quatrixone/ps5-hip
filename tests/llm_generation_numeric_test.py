import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_generation_numeric_test as test

class NumericGeneration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.raw=self.root/'raw';self.ref=self.root/'reference';self.raw.mkdir();self.ref.mkdir()
  metadata={'prompt_tokens':[1],'runs':[{'run':r,'processed':[1],'generated':[2]} for r in range(2)]}
  for folder in (self.raw,self.ref):
   (folder/'generation.json').write_text(json.dumps(metadata));(folder/'steps.jsonl').write_text('steps\n')
  for r in range(2):
   for stage in [*(f'layer-{i}' for i in range(22)),'norm','logits']:
    name=f'run-{r}-pos-0-{stage}.f32le';size=128000 if stage=='logits' else 8192
    content=bytes(8)+struct.pack('<f',1.)+bytes(size-12) if stage=='logits' else bytes(size)
    (self.raw/name).write_bytes(content)
    if stage=='logits':(self.ref/name).write_bytes(content)
  (self.raw/'backend.json').write_text(json.dumps(dict(requests=2,nodes=1596,launches=1466,CPU_arithmetic=0,scheduler_forwarded=2,allocations=16,frees=16,instances_created=1,instances_freed=1)))
 def compare(self):return test.compare_generation(self.raw,self.ref)
 def test_pass_inventory_and_budget(self):
  rows,files,requests=self.compare();self.assertEqual((len(rows),len(files),requests),(2,48,2));self.assertTrue(all(r['pass_budget'] for r in rows));self.assertEqual((test.ABS_TOL,test.REL_TOL,test.L2_TOL),(2e-5,2e-4,2e-4))
 def test_feedback_metadata(self):
  p=self.raw/'generation.json';p.write_text(p.read_text().replace('"generated": [2]','"generated": [3]'))
  with self.assertRaisesRegex(ValueError,'feedback'):self.compare()
 def test_extra_inventory(self):
  (self.raw/'unexpected').write_bytes(b'')
  with self.assertRaisesRegex(ValueError,'inventory'):self.compare()
 def test_boundary_nonfinite_and_short(self):
  p=self.raw/'run-0-pos-0-layer-0.f32le';p.write_bytes(struct.pack('<f',float('nan'))+bytes(8188))
  with self.assertRaisesRegex(ValueError,'nonfinite'):self.compare()
  p.write_bytes(b'')
  with self.assertRaisesRegex(ValueError,'size'):self.compare()
 def test_reset_all_boundaries(self):
  p=self.raw/'run-1-pos-0-norm.f32le';p.write_bytes(struct.pack('<f',1e-8)+bytes(8188))
  with self.assertRaisesRegex(ValueError,'reset'):self.compare()
 def test_logit_error_fails_unchanged_gate(self):
  for r in range(2):(self.raw/f'run-{r}-pos-0-logits.f32le').write_bytes(bytes(8)+struct.pack('<f',1.)+bytes(8)+struct.pack('<f',.1)+bytes(127976))
  rows,_,_=self.compare();self.assertFalse(all(r['pass_budget'] for r in rows))
 def test_no_cpu_or_owner_leak(self):
  p=self.raw/'backend.json';original=json.loads(p.read_text())
  for key,value in [('CPU_arithmetic',1),('frees',15),('scheduler_forwarded',1),('CPU_arithmetic',False)]:
   p.write_text(json.dumps({**original,key:value}))
   with self.subTest(key=key),self.assertRaises(ValueError):self.compare()
 def test_within_budget_near_tie_wrong_greedy(self):
  for r in range(2):
   name=f'run-{r}-pos-0-logits.f32le'
   (self.ref/name).write_bytes(bytes(8)+struct.pack('<ff',1.,1.-1e-6)+bytes(127984))
   (self.raw/name).write_bytes(bytes(8)+struct.pack('<ff',1.-1e-6,1.)+bytes(127984))
  with self.assertRaisesRegex(ValueError,'argmax'):self.compare()
 def test_symlink_boundary(self):
  p=self.raw/'run-0-pos-0-logits.f32le';p.unlink();p.symlink_to(self.ref/p.name)
  with self.assertRaisesRegex(ValueError,'kind'):self.compare()

if __name__=='__main__':unittest.main()
