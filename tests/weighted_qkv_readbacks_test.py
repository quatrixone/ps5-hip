import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_weighted_qkv_readbacks as v

class Readbacks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.assets={'input.f32le':bytes(8192),'run1/norm.f32le':struct.pack('<2048f',*([1.]*2048))}
  semantic,budget=v.reference.weighted_reference(self.assets['input.f32le'],self.assets['run1/norm.f32le'])
  self.assets.update({'weighted.fp64le':v.reference.pack(semantic,'d'),'weighted.budget.f64le':v.reference.pack(budget,'d')})
  for name,rows in v.reference.PROFILE['projections'].items():self.assets['run1/'+name+'.weights.q4_0']=bytes(rows*1152)
  self.chain=v.reference.chained_reference
  # Isolate buffer continuity checks from the separately tested full Q4 arithmetic.
  def projection(weights,q8,width,rows):return {'fp64':[0.]*rows,'budget':[1e-6]*rows,'lane_f32':[0.]*rows}
  self.mock=patch.object(v.reference.q4,'reference_q4',side_effect=projection);self.mock.start();self.addCleanup(self.mock.stop)
  self.write(bytes(8192))
 def write(self,weighted):
  q8=v.reference.q4.quantize_q8_1(weighted)
  for stage in v.STAGES:
   payload={'input':self.assets['input.f32le'],'norm-weight':self.assets['run1/norm.f32le'],'weighted':weighted,'q8':b'\xcc'*2304 if stage=='weighted' else q8,'projection-weight':self.assets['run1/'+(stage if stage in ('k','v') else 'q')+'.weights.q4_0'],'output':struct.pack('<I',0x7fc00000)*2048 if stage in ('weighted','q8') else bytes(v.reference.PROFILE['projections'][stage]*4)}
   for name,data in payload.items():(self.root/f'weighted-qkv-{stage}.{name}.raw').write_bytes(v.seeded(name,data))
 def mutate(self,stage,name,offset,data):
  p=self.root/f'weighted-qkv-{stage}.{name}.raw';raw=bytearray(p.read_bytes());raw[offset:offset+len(data)]=data;p.write_bytes(raw)
 def test_success_all_bytes(self):
  r=v.verify_buffers(self.root,self.assets);self.assertEqual(r['raw_files'],30);self.assertEqual(r['projection_values'],2560);self.assertFalse(r['gpu_origin_established'])
 def test_guards_inputs_weights_tails(self):
  for stage,name,offset in [('weighted','input',17),('q','norm-weight',17),('v','projection-weight',16+294912),('k','output',16+1024),('q8','weighted',0),('v','q8',2335)]:
   self.write(bytes(8192));self.mutate(stage,name,offset,b'\xff')
   with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
 def test_weighted_bad_before_projection_oracle(self):
  for bad in (float('nan'),1.):
   self.write(bytes(8192));self.mutate('weighted','weighted',16,struct.pack('<f',bad))
   with patch.object(v.reference,'chained_reference',side_effect=AssertionError('must reject weighted first')):
    with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
 def test_shared_substitution_and_q8_corruption(self):
  actual=struct.pack('<2048f',*([1e-5]*2048));self.write(actual)
  v.verify_buffers(self.root,self.assets)
  self.mutate('q8','weighted',16,bytes(8192))
  with self.assertRaisesRegex(ValueError,'shared'):v.verify_buffers(self.root,self.assets)
  self.write(actual);self.mutate('q','q8',20,b'\x00')
  with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
 def test_nan_and_wrong_projection(self):
  for value in (float('nan'),1.):
   self.write(bytes(8192));self.mutate('k','output',16,struct.pack('<f',value))
   with self.assertRaisesRegex(ValueError,'projection'):v.verify_buffers(self.root,self.assets)
 def test_missing_extra_and_size(self):
  p=self.root/'weighted-qkv-q.input.raw';data=p.read_bytes();p.unlink()
  with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
  p.write_bytes(data[:-1])
  with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
  p.write_bytes(data);(self.root/'extra').write_bytes(b'')
  with self.assertRaises(ValueError):v.verify_buffers(self.root,self.assets)
 def test_authentication_merge_preserves_first_identity(self):
  producer=str(v.ROOT/'tools/build_llm_weighted_qkv_reference.py')
  with patch.object(v,'authenticate',return_value=({}, {producer:'0'*64})),patch.object(v,'verify_buffers',side_effect=AssertionError('must reject conflict first')):
   with self.assertRaisesRegex(ValueError,'conflicting authenticated identity'):v.verify(self.root,self.root,self.root/'report.json')
 def test_authentication_rejects_empty_and_wrong_class(self):
  p=self.root/'receipt.json';p.write_text('{}')
  with self.assertRaises(ValueError):v.authenticate(self.root)
  p.write_text(json.dumps({'classification':'offline-weighted-rmsnorm-full-qkv-reference','status':'PASS','gpu_executed':False,'native_executed':False,'profile':v.reference.PROFILE,'inputs':{},'artifacts':{}}))
  with self.assertRaisesRegex(ValueError,'empty'):v.authenticate(self.root)

if __name__=='__main__':unittest.main()
