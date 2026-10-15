"""Package validator controls; arithmetic is tested separately against real ISA."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import public_hip_q4_assets as a

class FakeOracle:
    PROFILE={'fixed':'test-only budget policy'}
    @staticmethod
    def quantize_q8_1(raw):return bytes(len(raw)//128*36)
    @staticmethod
    def quantize_q8(values,padded_width):return bytes(padded_width//32*36)
    @staticmethod
    def reference_q4(weight,q8,width,rows):return dict(fp64=[1.0]*rows,lane_f32=[1.0]*rows,budget=[0.01]*rows)

class Assets(unittest.TestCase):
    def corpus(self,folder):
        embedding=a.pack([2.0]*2048,'f');weights={n:bytes(size) for n,(_,size) in a.WEIGHTS.items()}
        report=dict(profile=a.declared_profile(FakeOracle),cases=[],q8_cases=[])
        (folder/'declared-profile.json').write_text(json.dumps(report['profile']))
        for name,width,rows in a.Q4_CASES:
            if name.startswith('attn-k'):activation=embedding;w=weights['attn-k'][:rows*width//32*18]
            elif name.startswith('ffn-down'):activation=a.pack([((i*73)%257-128)/64 for i in range(width)],'f');w=weights['ffn-down'][:rows*width//32*18]
            else:
                activation=a.pack([0.0]*width if name=='synthetic-zero' else [((i%32)-16)/8 for i in range(width)],'f')
                w=(a.pack([0.125],'e')+bytes(i|((15-i)<<4) for i in range(16)))*(width//32)
            data=dict(zip(a.FIELDS,[activation,w,bytes(width//32*36),a.pack([1.0]*rows,'d'),a.pack([1.0]*rows,'f'),a.pack([0.01]*rows,'d')]))
            (folder/name).mkdir();identity={}
            for n,v in data.items():(folder/name/n).write_bytes(v);identity[n]=dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
            report['cases'].append(dict(name=name,width=width,rows=rows,artifacts=identity))
        values={'q8-zero':[0.0]*256,'q8-constant':[1.0]*256,'q8-signed-ties':[127.,-127.,.5,-.5]*64,'q8-padding':[((i%31)-15)/8 for i in range(255)]}
        for name,width in a.Q8_CASES:
            (folder/name).mkdir();raw=a.pack(values[name],'f');q8=bytes(288)
            (folder/name/'input.f32le').write_bytes(raw);(folder/name/'q8.bin').write_bytes(q8)
            report['q8_cases'].append(dict(name=name,input_width=width,padded_width=256,input_sha256=hashlib.sha256(raw).hexdigest(),q8_sha256=hashlib.sha256(q8).hexdigest()))
        return report,embedding,weights

    def test_exact_assets_and_recomputed_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);report,e,w=self.corpus(folder)
            self.assertEqual(len(a.validate_corpus(folder,report,e,w,FakeOracle)),56)
            for name in ('input.f32le','weights.q4_0','q8.bin','fp64le','lane.f32le','budget.f64le'):
                p=folder/'attn-k-1'/name;original=p.read_bytes();altered=bytes([original[0]^1])+original[1:];p.write_bytes(altered)
                # Re-signing corrupted bytes must NOT bypass mathematical validation.
                old=report['cases'][0]['artifacts'][name];report['cases'][0]['artifacts'][name]=dict(bytes=len(altered),sha256=hashlib.sha256(altered).hexdigest())
                with self.subTest(name=name),self.assertRaises(ValueError):a.validate_corpus(folder,report,e,w,FakeOracle)
                p.write_bytes(original);report['cases'][0]['artifacts'][name]=old
            p=folder/'attn-k-1/q8.bin';p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):a.validate_corpus(folder,report,e,w,FakeOracle)

    def test_profile_inventory_and_padding_drift(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);report,e,w=self.corpus(folder)
            for mutate in (lambda j:j['profile']['oracle'].update(fixed='relaxed'),lambda j:j['cases'].pop(),lambda j:j['q8_cases'][-1].update(padded_width=512)):
                changed=json.loads(json.dumps(report));mutate(changed)
                with self.assertRaises(ValueError):a.validate_corpus(folder,changed,e,w,FakeOracle)
            p=folder/'q8-padding/input.f32le';p.write_bytes(p.read_bytes()+bytes(4))
            with self.assertRaises(ValueError):a.validate_corpus(folder,report,e,w,FakeOracle)

    def test_missing_closure_rejected_before_oracles(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);report=dict(status='PASS',classification='offline-gpu-format-q4-reference',native_executed=False,gpu_executed=False,weights_repeated_bitwise_equal=True,inputs={})
            (folder/'receipt.json').write_text(json.dumps(report))
            fake=types.SimpleNamespace(fixture_inputs=lambda folder:{'module':'hash'})
            with patch.dict(sys.modules,{'qualify_public_hip_q4':fake}):
                with self.assertRaisesRegex(ValueError,'missing numerical input closure'):a.assets(folder,folder)

    def test_asset_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for name in ('../escape','/tmp/absolute'):
                with self.assertRaises(ValueError):a.file_bytes(folder,name)

if __name__=='__main__':unittest.main()
