"""Raw readback corruption controls; synthetic files establish no GPU provenance."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_public_hip_q4 as v


class Dumps(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.dumps=self.root/'dumps';self.dumps.mkdir();self.assets={}
        for name,width,rows,padded in [(n,w,r,w) for n,w,r in v.MATRIX_CASES]+[(n,w,0,p) for n,w,p in v.Q8_CASES]:
            folder=self.root/name;folder.mkdir()
            # A trusted synthetic asset set for testing transport validation only.
            files={'input.f32le':struct.pack('<f',127)*width,
                   'q8.bin':struct.pack('<ee32b',1,4064,*([127]*32))*(padded//32)}
            if rows:
                files.update({'weights.q4_0':(struct.pack('<e',1)+bytes([0x99])*16)*(width//32)*rows,
                              'fp64le':struct.pack('<d',127*width)*rows,
                              'lane.f32le':struct.pack('<f',127*width)*rows,
                              'budget.f64le':struct.pack('<d',0)*rows})
            for suffix,data in files.items():
                path=folder/suffix;path.write_bytes(data);self.assets[f'q4/{name}/{suffix}']=path
        for name,stage,width,rows,padded in v.STAGES:
            for buffer,capacity in v.CAPACITIES.items():
                payload=b''
                if buffer=='input':payload=(self.root/name/'input.f32le').read_bytes()
                elif buffer=='weights' and rows:payload=(self.root/name/'weights.q4_0').read_bytes()
                elif buffer=='q8':payload=(self.root/name/'q8.bin').read_bytes()
                elif buffer=='output' and rows:
                    payload=struct.pack('<I',0x7fc00000)*rows if stage=='q8' else struct.pack('<f',127*width)*rows
                fill=b'\xa5' if buffer in ('input','weights') else b'\x5a'
                raw=fill*16+payload+fill*(capacity-len(payload)+16)
                (self.dumps/f'q4-{name}.{stage}.{buffer}.raw').write_bytes(raw)

    def verify(self):return v.verify_buffers(self.dumps,self.assets)

    def test_complete_comparison_has_no_provenance_claim(self):
        report=self.verify()
        self.assertEqual((report['stages'],report['raw_files'],report['q4_values'],report['q8_bytes']),
                         (28,112,592,80640))
        self.assertEqual(report['raw_bytes'],9097984)
        self.assertFalse(report['gpu_origin_established'])

    def test_each_buffer_guard_payload_and_inactive_corruption(self):
        for buffer in v.CAPACITIES:
            path=self.dumps/f'q4-attn-k-1.frozen.{buffer}.raw';original=path.read_bytes()
            for at in (0,16,len(original)-17,len(original)-1):
                with self.subTest(buffer=buffer,offset=at):
                    data=bytearray(original);data[at]^=1;path.write_bytes(data)
                    with self.assertRaises(ValueError):self.verify()
                    path.write_bytes(original)

    def test_nonfinite_noop_and_changed_unused_output(self):
        path=self.dumps/'q4-attn-k-1.chained.output.raw';original=path.read_bytes()
        for value in (0,float('nan'),float('inf')):
            path.write_bytes(original[:16]+struct.pack('<f',value)+original[20:])
            with self.assertRaises(ValueError):self.verify()
        path.write_bytes(original)
        path=self.dumps/'q4-attn-k-1.q8.output.raw';raw=path.read_bytes()
        path.write_bytes(raw[:16]+struct.pack('<f',260096)+raw[20:])
        with self.assertRaises(ValueError):self.verify()

    def test_missing_short_extra_symlink_and_bad_reference(self):
        path=self.dumps/'q4-q8-zero.q8.q8.raw';original=path.read_bytes();path.unlink()
        with self.assertRaises(ValueError):self.verify()
        path.write_bytes(original[:-1])
        with self.assertRaises(ValueError):self.verify()
        path.write_bytes(original);extra=self.dumps/'q4-extra.raw';extra.write_bytes(b'')
        with self.assertRaises(ValueError):self.verify()
        extra.unlink();path.unlink();path.symlink_to(self.assets['q4/q8-zero/q8.bin'])
        with self.assertRaises(ValueError):self.verify()
        path.unlink();path.write_bytes(original)
        budget=self.assets['q4/attn-k-1/budget.f64le'];budget.write_bytes(struct.pack('<d',-1))
        with self.assertRaises(ValueError):self.verify()

    def test_authenticated_receipt_and_changed_input(self):
        dependency=self.root/'dependency';dependency.write_bytes(b'locked')
        auth=dict(assets=self.assets,reference={'receipt':'reference'},
                  inputs={str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()})
        output=self.root/'verified.json'
        with patch('public_hip_q4_assets.assets',return_value=auth):
            result=v.verify('fixture','reference',self.dumps,output)
            self.assertEqual(result['status'],'PASS')
            self.assertFalse(json.loads(output.read_text())['gpu_origin_established'])
            with self.assertRaisesRegex(ValueError,'already exists'):
                v.verify('fixture','reference',self.dumps,output)
            dependency.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'input changed'):
                v.verify('fixture','reference',self.dumps,self.root/'changed.json')
            self.assertFalse((self.root/'changed.json').exists())

    def test_verifier_changed_during_authentication(self):
        implementation=self.root/'verifier.py';implementation.write_bytes(b'original')
        output=self.root/'changed-verifier.json'
        def authenticate(*args):
            implementation.write_bytes(b'changed')
            return dict(assets=self.assets,reference={},inputs={})
        with patch.object(v,'__file__',str(implementation)),patch('public_hip_q4_assets.assets',side_effect=authenticate):
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                v.verify('fixture','reference',self.dumps,output)
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
