import math,struct,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_llm_layer_readbacks import compare,FILES,score_oracle
class LayerReadbacks(unittest.TestCase):
 def setUp(self):
  self.expected={n:struct.pack('<'+str(s//(2 if n.endswith('.f16le') else 4))+('e' if n.endswith('.f16le') else 'f'),*([1.]*(s//(2 if n.endswith('.f16le') else 4)))) for n,s in FILES.items()}
  for name,size in FILES.items():
   if name.endswith('out-08.f32le'):
    position=int(name.split('-')[3]);extent=256 if position<256 else 512
    values=[64. if slot<=position else 0. for head in range(32) for slot in range(extent)]
    self.expected[name]=struct.pack('<'+str(len(values))+'f',*values)
 def test_score_gqa_rounding_and_future_slots(self):
  query=[0.]*2048;cache=[0.]*131072
  for head in range(32):query[head*64]=1.0001
  for slot in range(512):
   for head in range(4):cache[slot*256+head*64]=head+1
  raw={'run-0-cache-k.f16le':struct.pack('<131072e',*cache),'run-0-pos-0-out-04.f32le':struct.pack('<2048f',*query)}
  targets,budgets=score_oracle('run-0-pos-0-out-08.f32le',raw)
  for head in range(32):
   self.assertEqual(targets[head*256],head//8+1)
   self.assertTrue(all(x==0 for x in targets[head*256+1:(head+1)*256]))
   self.assertGreater(budgets[head*256],0)
 def test_score_arithmetic(self):
  with self.assertRaisesRegex(ValueError,'chained score'):compare(self.mutate('run-0-pos-511-out-08.f32le',64.001),self.expected)
 def mutate(self,name,value):
  actual=dict(self.expected);fmt='e' if name.endswith('.f16le') else 'f';blob=bytearray(actual[name]);struct.pack_into('<'+fmt,blob,0,value);actual[name]=bytes(blob);return actual
 def test_exact(self):self.assertEqual(len(compare(self.expected,self.expected)),151)
 def test_missing(self):
  actual=dict(self.expected);actual.pop(next(iter(actual)))
  with self.assertRaisesRegex(ValueError,'inventory'):compare(actual,self.expected)
 def test_size(self):
  actual=dict(self.expected);actual[next(iter(actual))]=b''
  with self.assertRaisesRegex(ValueError,'size'):compare(actual,self.expected)
 def test_nonfinite(self):
  with self.assertRaisesRegex(ValueError,'nonfinite'):compare(self.mutate('run-0-pos-511-out-20.f32le',math.nan),self.expected)
 def test_error(self):
  with self.assertRaisesRegex(ValueError,'budget'):compare(self.mutate('run-0-pos-511-out-20.f32le',1.01),self.expected)
 def test_embedding_exact(self):
  with self.assertRaisesRegex(ValueError,'embedding'):compare(self.mutate('run-0-pos-511-out-00.f32le',1.000001),self.expected)
 def test_reset_exact(self):
  with self.assertRaisesRegex(ValueError,'reset'):compare(self.mutate('run-1-pos-0-out-20.f32le',1.000001),self.expected)
 def test_unused_cache_zero(self):
  name='run-1-cache-k.f16le';self.expected[name]=bytes(FILES[name])
  with self.assertRaisesRegex(ValueError,'zero cache'):compare(self.mutate(name,0.000001),self.expected)
 def test_mask_zero(self):
  name='run-0-pos-511-out-09.f32le';self.expected[name]=bytes(FILES[name])
  with self.assertRaisesRegex(ValueError,'masked'):compare(self.mutate(name,0.000001),self.expected)
if __name__=='__main__':unittest.main()
