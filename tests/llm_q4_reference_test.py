"""Numerical contract tests with hand-derived answers and corruption controls."""
from pathlib import Path
from fractions import Fraction
import math
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import llm_q4_reference as r

class Reference(unittest.TestCase):
    def test_fma_and_double_rounding(self):
        self.assertEqual(r.fma(1+2**-23, 1-2**-23, -1), -2**-46)
        self.assertEqual(r.fma(1+2**-23, 1.5, -2**-54), r.from_bits(r.bits(1.5)+1))
        self.assertEqual(r.f32(Fraction(1)+Fraction(1,2**24)), 1)
        self.assertEqual(r.f32(Fraction(1)+Fraction(3,2**24)), 1+2**-22)
        self.assertEqual(r.f32(Fraction(1)+Fraction(1,2**24)+Fraction(1,2**100)), 1+2**-23)

    def test_quantization_known_bytes(self):
        self.assertEqual(r.quantize_q8([0]*32), bytes(36))
        self.assertEqual(r.quantize_q8([-0.0]*32), bytes([0,0,0,128])+bytes(32))
        expected=struct.pack('<ee32b',1,127,*([127]+[0]*31))
        self.assertEqual(r.quantize_q8([127]+[0]*31), expected)
        for lane in range(32):
            values=[0]*32;values[lane]=-127
            qs=[0]*32;qs[lane]=-127
            self.assertEqual(r.quantize_q8(values),struct.pack('<ee32b',1,-127,*qs))
        values=[127,.5,-.5,1.5,-1.5]+[0]*27
        self.assertEqual(r.quantize_q8(values),struct.pack('<ee32b',1,127,127,1,-1,2,-2,*([0]*27)))
        # Original sum differs from sum of quantized integers (127 versus128).
        raw=r.quantize_q8([127,.49,.49]+[0]*29)
        self.assertEqual(struct.unpack_from('<e',raw,2)[0],128)
        self.assertEqual(sum(struct.unpack_from('<32b',raw,4)),127)
        self.assertEqual(r.quantize_q8([127],64),expected+bytes(36))

    def test_half_ties_and_scale_before_half(self):
        self.assertEqual(r.half(1+2**-11),1)
        self.assertEqual(r.half(1+3*2**-11),1+2**-9)
        raw=r.quantize_q8([1]*32)
        self.assertEqual(struct.unpack_from('<32b',raw,4),(127,)*32)
        self.assertEqual(struct.unpack_from('<e',raw,2)[0],32)

    def test_handencoded_semantics_and_nibble_order(self):
        q8=struct.pack('<ee32b',1,32,*([1]*32))
        q4=r.q4_block(2,list(range(16))*2)
        self.assertEqual(r.semantic_fp64(q4,q8),-32)
        self.assertEqual(r.lane_model(q4,q8),-32)
        q8=struct.pack('<ee32b',1,1,*([1]+[0]*31))
        q4=r.q4_block(2,[15]+[8]*31)
        self.assertEqual(r.semantic_fp64(q4,q8),14)
        self.assertEqual(r.lane_model(q4,q8),14)
        self.assertEqual(r.verify_output(14,q4,q8)['absolute_error'],0)

    def test_widths_cancellation_and_error_budget(self):
        for width in r.WIDTHS:
            values=[((i*37)%509-254)/64 for i in range(width)]
            q8=r.quantize_q8(values)
            weights=b''.join(r.q4_block((-1 if b%2 else 1)*2**((b%7)-8),[(i*3+b)%16 for i in range(32)]) for b in range(width//32))
            report=r.reference_q4(weights,q8,width,1)
            self.assertLessEqual(abs(report['fp64'][0]-report['lane_f32'][0]),report['budget'][0])
            self.assertGreater(report['budget'][0],0)
            with self.assertRaises(ValueError):r.verify_output(report['fp64'][0]+1+2*report['budget'][0],weights,q8)
        # Exact cancellation of large terms needs an absolute normwise budget.
        q8=struct.pack('<ee32b',1,32,*([1]*32))*64
        weights=b''.join(r.q4_block(1024 if i%2 else -1024,[15]*32) for i in range(64))
        self.assertEqual(r.semantic_fp64(weights,q8),0)
        self.assertEqual(r.lane_model(weights,q8),0)
        self.assertGreater(r.error_budget(weights,q8),1)

    def test_corruption_controls(self):
        q8=struct.pack('<ee32b',1,32,*([1]*32))*64
        weights=r.q4_block(1,[15]*32)*64
        expected=14336
        self.assertEqual(r.lane_model(weights,q8),expected)
        for value in (0,float('nan'),float('inf'),expected*.75):
            with self.assertRaises(ValueError):r.verify_output(value,weights,q8)
        for corrupted in (bytes([weights[0],weights[1],0])+weights[3:],weights[:-1]):
            if len(corrupted)==len(weights):self.assertNotEqual(r.semantic_fp64(corrupted,q8),expected)
            else:
                with self.assertRaises(ValueError):r.lane_model(corrupted,q8)
        corrupt=bytearray(q8);corrupt[2:4]=struct.pack('<e',16)
        self.assertNotEqual(r.semantic_fp64(weights,corrupt),expected)
        swapped=bytearray(q8);swapped[:4]=q8[2:4]+q8[:2]
        self.assertNotEqual(r.semantic_fp64(weights,swapped),expected)

    def test_half_subnormal_exact_bits_and_real_cancellation(self):
        for value, encoding in [(2**-24, 1), (-2**-23, 0x8002), (1023*2**-24, 0x3ff)]:
            self.assertEqual(struct.pack('<e', r.half(value)), struct.pack('<H', encoding))
        self.assertEqual(r.half(1.5*2**-24), 2*2**-24)
        values = [0.550297200679779, -0.550297200679779, -0.1834324151277542, 0.550297200679779, -0.1834324151277542, -0.7337296605110168, -0.1834324151277542, 1.100594401359558, -0.3668648302555084, 0.1834324151277542, 0.1834324151277542, -0.1834324151277542, -0.9171620607376099, 1.100594401359558, 0.9171620607376099, -0.1834324151277542, -0.550297200679779, -0.550297200679779, 0.0, 0.3668648302555084, 0.1834324151277542, -0.7337296605110168, 1.284026861190796, -0.1834324151277542, 0.3668648302555084, -0.1834324151277542, 0.1834324151277542, 0.1834324151277542, 0.3668648302555084, 0.1834324151277542, -0.550297200679779, -1.4674593210220337]
        self.assertEqual(r.butterfly(values), -2**-23)
        self.assertEqual(r.quantize_q8(values)[2:4], bytes([2,128]))
        # Half subnormal s is promoted exactly; correction must NOT flush to0.
        q8=struct.pack('<ee32b',1,-2**-23,*([0]*32))
        weights=r.q4_block(1,[8]*32)
        self.assertEqual(r.semantic_fp64(weights,q8),2**-20)
        self.assertEqual(r.lane_model(weights,q8),2**-20)

    def test_rejected_domain_and_lengths(self):
        for values in ([math.nan]*32,[math.inf]*32,[4096]*32,[2**-30]*32,[2**-140]*32):
            with self.assertRaises(ValueError):r.quantize_q8(values)
        for x in (2**-26,65536):
            with self.assertRaises(ValueError):r.half(x)
        with self.assertRaises(ValueError):r.quantize_q8([1],31)
        with self.assertRaises(ValueError):r.quantize_q8_1(bytes(4))
        with self.assertRaises(ValueError):r.reference_q4(bytes(18),bytes(36),2048,1)

if __name__=='__main__':unittest.main()
