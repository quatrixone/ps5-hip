import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_llm_rmsnorm_reference import WIDTH,EPSILON,read_f32,fp64_reference,compare,build

class RmsNormReferenceTest(unittest.TestCase):
    def test_closed_form_zero_constant_and_tail(self):
        self.assertEqual(fp64_reference([0.0]*WIDTH),[0.0]*WIDTH)
        expected=1/math.sqrt(1+EPSILON)
        self.assertEqual(fp64_reference([1.0]*WIDTH),[expected]*WIDTH)
        values=[0.0]*WIDTH;values[-1]=10.0;out=fp64_reference(values)
        self.assertEqual(out[:-1],[0.0]*(WIDTH-1));self.assertAlmostEqual(out[-1],10/math.sqrt(100/WIDTH+EPSILON))
    def test_noop_corruption_and_nonfinite_rejected(self):
        values=[0.0]*WIDTH;values[-1]=10;reference=fp64_reference(values)
        self.assertEqual(compare(reference,reference)['maximum_absolute_error'],0)
        with self.assertRaisesRegex(ValueError,'mismatch'):compare(values,reference)
        bad=list(reference);bad[0]=1
        with self.assertRaisesRegex(ValueError,'mismatch'):compare(bad,reference)
        bad[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'non-finite'):compare(bad,reference)
        with self.assertRaisesRegex(ValueError,'shape'):compare([1],reference)
    def test_binary_extent_nonfinite_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'values';path.write_bytes(struct.pack('<ff',1.0,2.0));self.assertEqual(read_f32(path,2),(1.0,2.0))
            with self.assertRaisesRegex(ValueError,'byte count'):read_f32(path,3)
            path.write_bytes(struct.pack('<f',float('inf')))
            with self.assertRaisesRegex(ValueError,'non-finite'):read_f32(path,1)
            with self.assertRaisesRegex(ValueError,'fresh'):build(Path(directory),Path('unused'),Path('unused'))

if __name__=='__main__':unittest.main()
