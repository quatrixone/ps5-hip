import json
from pathlib import Path
import sys,tempfile,unittest,struct
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_weighted_rmsnorm_fixture as w
from build_llm_rmsnorm_fixture import OFFSETS,SIZES,HIDDEN

class WeightedFixture(unittest.TestCase):
    def notes(self):
        args=[(o,s,'global_buffer' if i in (0,1,9,17) else 'by_value') for i,(o,s) in enumerate(zip(OFFSETS,SIZES))]+list(HIDDEN)
        text='  - .args:\n'+''.join(f'      - .offset: {o}\n        .size: {s}\n        .value_kind: {k}\n' for o,s,k in args)
        fields=dict(kernarg_segment_size=488,kernarg_segment_align=8,group_segment_fixed_size=0,private_segment_fixed_size=0,max_flat_workgroup_size=1024,wavefront_size=32)
        return text+''.join(f'    .{k}: {v}\n' for k,v in fields.items())+f'    .name: {w.SYMBOL}\namdhsa.target: amdgcn-amd-amdhsa--gfx1013\n'
    def test_complete_metadata(self):
        notes=self.notes();self.assertEqual(len(w.metadata_contract(notes)['arguments']),40)
        for old,new in [('gfx1013','gfx1030'),('.offset: 352','.offset: 356'),('.size: 12','.size: 16'),('.kernarg_segment_size: 488','.kernarg_segment_size: 480'),('.wavefront_size: 32','.wavefront_size: 64'),(w.SYMBOL,w.ORIGINAL_SYMBOL)]:
            with self.subTest(old=old),self.assertRaises(ValueError):w.metadata_contract(notes.replace(old,new,1))
        with self.assertRaises(ValueError):w.metadata_contract(notes+f'    .name: {w.SYMBOL}\namdhsa.target: amdgcn-amd-amdhsa--gfx1013\n')
    def test_object_architecture(self):
        data=bytearray(64);data[:9]=b'\x7fELF\x02\x01\x01\x40\x03';struct.pack_into('<H',data,18,224);struct.pack_into('<I',data,48,0x142)
        w.object_header(data)
        for offset in (4,7,8,18,48,49):
            bad=bytearray(data);bad[offset]^=1
            with self.subTest(offset=offset),self.assertRaises(ValueError):w.object_header(bad)
        with self.assertRaises(ValueError):w.object_header(data[:63])
    def test_exact_descriptor(self):
        d=dict(group_bytes=0,private_bytes=0,kernarg_bytes=488,rsrc1=0x60af0184,rsrc2=0x38c,rsrc3=0,properties=0x409,preload=0)
        w.descriptor_contract(d,w.PROFILE)
        for k in d:
            wrong=dict(d);wrong[k]^=1
            with self.subTest(k=k),self.assertRaises(ValueError):w.descriptor_contract(wrong,w.PROFILE)
    def test_parent_classification_and_symbol_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for report in ({'classification':w.CLASSIFICATION},{'selected':{'symbol':w.SYMBOL}},{}):
                (p/'receipt.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'original unweighted'):w.original_inputs(p)
    def test_identity_and_recursive_parent_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);report=dict(classification=w.CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,original_fixture=str(p),inputs={})
            (p/'receipt.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'recursive'):w.fixture_inputs(p)
            report['original_fixture']=str(p/'base');(p/'receipt.json').write_text(json.dumps(report))
            with patch.object(w,'original_inputs',return_value={'parent':'digest'}):
                with self.assertRaises(ValueError):w.fixture_inputs(p)
    def test_artifact_corruption_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for n in w.REQUIRED:(p/n).write_bytes(b'original')
            artifacts={n:dict(sha256=w.sha(p/n),bytes=8) for n in w.REQUIRED}
            report=dict(classification=w.CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,original_fixture=str(p/'base'),artifacts=artifacts)
            for bad in ('corrupt','escape'):
                if bad=='corrupt':(p/'weighted.hsaco').write_bytes(b'changed!')
                else:
                    (p/'weighted.hsaco').write_bytes(b'original');report['artifacts']['../escape']={}
                (p/'receipt.json').write_text(json.dumps(report))
                with patch.object(w,'original_inputs',return_value={'parent':'digest'}),patch.object(w,'checked_inputs',return_value={'parent':'digest'}),patch.object(w,'checked_tools',return_value={}):
                    with self.assertRaisesRegex(ValueError,'artifact changed|invalid artifact'):w.fixture_inputs(p)
    def test_lld_flavour(self):
        with patch.object(w.shutil,'which',return_value='/usr/bin/ld.lld-18'),patch.object(w.subprocess,'check_output',return_value='LLD18.1.8') as run,patch.object(w,'sha',return_value='digest'):
            self.assertEqual(w.discover_tool('ld.lld-18')['path'],'/usr/bin/ld.lld-18')
            run.assert_called_once_with(['/usr/bin/ld.lld-18','--version'],text=True)

if __name__=='__main__':unittest.main()
