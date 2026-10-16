import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_weighted_qkv_reference as b

def f32(values):return struct.pack('<2048f',*values)

class Weighted(unittest.TestCase):
 def test_analytic_constant_signed_weights_and_budget_derivation(self):
  weights=([0.,1.,-2.,4.]*512);expected=1/math.sqrt(1+b.EPSILON)
  reference,budgets=b.weighted_reference(f32([1.]*2048),f32(weights))
  self.assertEqual(reference,[expected*w for w in weights])
  u=2**-24;norm_budget=2e-5+2e-4*expected
  for w,limit in zip(weights,budgets):
   self.assertEqual(limit,abs(w)*norm_budget+u/(1-u)*abs(w)*(expected+norm_budget)+2**-126)
  metrics=b.compare_weighted(f32(reference),reference,budgets)
  self.assertLess(metrics['maximum_budget_fraction'],1/32)
 def test_zero_and_small_values_finite(self):
  reference,budgets=b.weighted_reference(f32([0.]*2048),f32([-3.]*2048))
  self.assertEqual(reference,[0.]*2048);b.compare_weighted(f32([0.]*2048),reference,budgets)
  tiny=f32([2**-140]*2048);reference,budgets=b.weighted_reference(tiny,f32([2**-100]*2048))
  b.compare_weighted(f32([0.]*2048),reference,budgets)
 def test_malformed_nonfinite_overflow_and_corruption(self):
  valid=f32([1.]*2048)
  for bad in (valid[:-1],valid+b'x',f32([math.nan]*2048),f32([math.inf]*2048)):
   for args in ((bad,valid),(valid,bad)):
    with self.assertRaises(ValueError):b.weighted_reference(*args)
  huge=f32([float.fromhex('0x1.fffffep127')]*2048)
  with self.assertRaises(ValueError):b.weighted_reference(f32([1.]+[0.]*2047),huge)
  reference,budgets=b.weighted_reference(valid,valid)
  changed=list(reference);changed[17]+=4*budgets[17]
  with self.assertRaisesRegex(ValueError,'budget exceeded'):b.compare_weighted(f32(changed),reference,budgets)
  with self.assertRaises(ValueError):b.compare_weighted(valid,reference,[0.]*2048)
 def test_readback_quantization_discontinuity_not_cpu_equality(self):
  cpu=[127.,-127.,0.4999999701976776,0.]*512
  gpu=cpu.copy();gpu[2]=0.5
  self.assertLess(abs(cpu[2]-gpu[2]),2e-5)
  weights={name:b'' for name in ('q','k','v')}
  with patch.object(b.q4,'reference_q4',return_value={'marker':'only numerical projection mocked'}) as projection:
   frozen=b.chained_reference(f32(cpu),weights);actual=b.chained_reference(f32(gpu),weights)
   self.assertNotEqual(frozen['q8'],actual['q8'])
   for call in projection.call_args_list[-3:]:self.assertEqual(call.args[1],actual['q8'])
  with self.assertRaises(ValueError):b.chained_reference(f32(cpu),{'q':b''})
 def test_fresh_output_and_input_identity(self):
  with tempfile.TemporaryDirectory() as d,patch.object(b,'ROOT',Path(d)):
   root=Path(d);out=root/'results/old';out.mkdir(parents=True)
   for target in (out,root/'elsewhere',root/'results'):
    with self.assertRaisesRegex(ValueError,'fresh output'):b.build(root,root,target)
   p=root/'asset';p.write_bytes(b'abcd');inputs={};b.bind(inputs,p);p.write_bytes(b'abce')
   with self.assertRaises(ValueError):b.unchanged(inputs)

if __name__=='__main__':unittest.main()
