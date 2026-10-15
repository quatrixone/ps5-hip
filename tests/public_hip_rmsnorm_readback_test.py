"""Synthetic byte-oracle tests; these are not native numerical evidence."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import verify_public_hip_rmsnorm as v


class ReadbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = {}
        for case, rows in v.CASES:
            payload = struct.pack('<f', 1.0) * (rows*v.WIDTH)
            for suffix in ('input', 'cpu'):
                name = f'rmsnorm/{case}.{suffix}.f32le'
                path = self.root/name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(payload)
                self.assets[name] = str(path)
            (self.root/f'rmsnorm-{case}.input-readback.raw').write_bytes(b'\xa5'*16 + payload + b'\xa5'*(v.TOTAL-16-len(payload)))
            (self.root/f'rmsnorm-{case}.output.raw').write_bytes(b'\x5a'*16 + payload + b'\x5a'*(v.TOTAL-16-len(payload)))
        self.target = self.root/'rmsnorm-stress-1.output.raw'

    def check(self):
        return v.verify_buffers(self.root, self.assets)

    def mutate(self, path, offset, value):
        original = path.read_bytes()
        changed = bytearray(original)
        changed[offset:offset+len(value)] = value
        path.write_bytes(changed)
        return original

    def test_complete_counts_and_tolerance(self):
        result = self.check()
        self.assertEqual(result['totals']['values'],81920)
        self.assertEqual(len(result['cases']),6)
        self.assertEqual(len(result['verified_files']),24)
        original = self.mutate(self.target,16,struct.pack('<f',1.00021))
        self.assertGreater(self.check()['cases'][0]['maximum_budget_fraction'],0.9)
        self.target.write_bytes(original)
        self.mutate(self.target,16,struct.pack('<f',1.00023))
        with self.assertRaisesRegex(ValueError,'numerical mismatch'):self.check()

    def test_zero_reference_absolute_budget(self):
        cpu=Path(self.assets['rmsnorm/stress-1.cpu.f32le'])
        self.mutate(cpu,0,struct.pack('<f',0.0))
        self.mutate(self.target,16,struct.pack('<f',1.9e-5))
        self.assertGreater(self.check()['cases'][0]['maximum_budget_fraction'],0.9)
        self.mutate(self.target,16,struct.pack('<f',2.1e-5))
        with self.assertRaisesRegex(ValueError,'numerical mismatch'):self.check()

    def test_nan_unwritten_infinite_and_corrupt(self):
        original = self.target.read_bytes()
        for value in (float('nan'),float('inf'),float('-inf'),2.0):
            self.target.write_bytes(original)
            self.mutate(self.target,16,struct.pack('<f',value))
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'non-finite output|numerical mismatch'):self.check()

    def test_missing_short_long_and_extra_dumps(self):
        original = self.target.read_bytes()
        self.target.unlink()
        with self.assertRaisesRegex(ValueError,'twelve case dumps'):self.check()
        for data in (original[:-1],original+b'\0'):
            self.target.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'wrong size'):self.check()
        self.target.write_bytes(original)
        (self.root/'rmsnorm-old-1.output.raw').write_bytes(original)
        with self.assertRaisesRegex(ValueError,'twelve case dumps'):self.check()

    def test_input_output_guards_and_inactive(self):
        for name,offset in (('output',0),('output',16+v.WIDTH*4),('output',v.TOTAL-1),
                            ('input-readback',0),('input-readback',16),('input-readback',v.TOTAL-1)):
            path=self.root/f'rmsnorm-stress-1.{name}.raw'
            original=self.mutate(path,offset,b'\xff')
            with self.subTest(name=name,offset=offset),self.assertRaisesRegex(ValueError,'guard|input'):self.check()
            path.write_bytes(original)

    def test_reference_asset_contract(self):
        removed=self.assets.pop('rmsnorm/stress-1.input.f32le')
        with self.assertRaisesRegex(ValueError,'twelve reference'):self.check()
        self.assets['rmsnorm/stress-1.input.f32le']=removed
        path=Path(self.assets['rmsnorm/stress-1.cpu.f32le'])
        self.mutate(path,0,struct.pack('<f',float('nan')))
        with self.assertRaisesRegex(ValueError,'non-finite reference'):self.check()


if __name__=='__main__':unittest.main()
