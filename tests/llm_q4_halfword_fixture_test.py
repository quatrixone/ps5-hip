import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_q4_halfword_fixture as h

class HalfwordFixtureTest(unittest.TestCase):
    def test_linker_invocation_preserves_flavour(self):
        with patch.object(h.shutil,'which',return_value='/usr/bin/ld.lld-18'), patch.object(h.subprocess,'check_output',return_value='Ubuntu LLD 18.1.8') as run, patch.object(h,'sha',return_value='digest'):
            row=h.discover_tool('ld.lld-18')
            self.assertEqual(row['path'],'/usr/bin/ld.lld-18')
            run.assert_called_once_with(['/usr/bin/ld.lld-18','--version'],text=True)

    def test_exact_transform(self):
        source='prefix\n'+('  load i16, ptr addrspace(1) %p, align 2\n'*4)+'suffix\n'
        result=h.transform(source)
        self.assertEqual(result.replace('load volatile i16','load i16'),source)
        self.assertEqual(result.count('load volatile i16'),4)
        for bad in (source.replace('load i16','load i32',1),result,source+source):
            with self.assertRaises(ValueError): h.transform(bad)

    def test_descriptor_exact(self):
        expected=dict(group_bytes=384,private_bytes=0,kernarg_bytes=192,rsrc1=0x60af00c4,rsrc2=0xb8c,rsrc3=0,properties=0x409,preload=0)
        profile=dict(h.PROFILES['q4'],rsrc1=0x60af00c4)
        h.descriptor_contract(expected,profile)
        for key in expected:
            bad=dict(expected);bad[key]^=1
            with self.assertRaises(ValueError):h.descriptor_contract(bad,profile)

    def test_diagnostic_parent_rejected_before_recursion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'receipt.json').write_text(json.dumps({'classification':h.CLASSIFICATION}))
            with self.assertRaisesRegex(ValueError,'parent prohibited'):h.original_inputs(p)

    def test_empty_identity_and_recursive_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);report=dict(classification=h.CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,original_fixture=str(p),inputs={})
            receipt=p/'receipt.json';receipt.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'recursive'):h.fixture_inputs(p)
            report['original_fixture']=str(p/'original');receipt.write_text(json.dumps(report))
            with patch.object(h,'original_inputs',return_value={'parent':'digest'}):
                with self.assertRaises(ValueError):h.fixture_inputs(p)

    def test_artifact_identity_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);inputs={'parent':'digest'}
            for name in h.REQUIRED:(p/name).write_bytes(b'original')
            artifacts={name:dict(sha256=h.sha(p/name),bytes=8) for name in h.REQUIRED}
            report=dict(classification=h.CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,original_fixture=str(p/'original'),artifacts=artifacts)
            (p/'receipt.json').write_text(json.dumps(report));(p/'q4.hsaco').write_bytes(b'corrupt!')
            with patch.object(h,'original_inputs',return_value=inputs),patch.object(h,'checked_inputs',return_value=dict(inputs)),patch.object(h,'checked_tools',return_value={}):
                with self.assertRaisesRegex(ValueError,'artifact changed'):h.fixture_inputs(p)

if __name__=='__main__':unittest.main()
