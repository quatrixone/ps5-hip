"""Reject stale or damaged aggregate compiler evidence before any CLR linking."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import qualify_public_hip_aggregate as qualifier


class AggregateQualificationInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.compiler=self.root/'output';self.compiler.mkdir()
        root_patch=patch.object(qualifier,'ROOT',self.root);root_patch.start();self.addCleanup(root_patch.stop)
        tools_patch=patch.object(qualifier.shutil,'which',side_effect=lambda n:str(self.root/'tools-bin'/n));tools_patch.start();self.addCleanup(tools_patch.stop)
        inputs={}
        for name in qualifier.COMPILER_SOURCE_INPUTS+tuple('tools-bin/'+n for n in qualifier.COMPILER_TOOLS):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('controlled '+name).encode());inputs[str(p)]=qualifier.sha(p)
        inputs[sys.executable]=qualifier.sha(sys.executable)
        artifacts={}
        for name in qualifier.COMPILER_ARTIFACTS:
            p=self.compiler/name;p.write_bytes(('original '+name).encode());artifacts[name]=qualifier.sha(p)
        self.receipt=dict(classification='public-hip-genuine-aggregate-compiler-fixture',tests_passed=2,gpu_executed=False,inputs=inputs,artifacts=artifacts)
        self.save()

    def save(self):
        (self.compiler/'receipt.json').write_text(json.dumps(self.receipt))

    def test_corrupt_object_rejected(self):
        self.assertIn(str(self.compiler/'fixture.o'),qualifier.compiler_inputs(self.compiler))
        (self.compiler/'fixture.o').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'artifact changed'):qualifier.compiler_inputs(self.compiler)

    def test_missing_object_rejected(self):
        (self.compiler/'fixture.hsaco').unlink()
        with self.assertRaises(FileNotFoundError):qualifier.compiler_inputs(self.compiler)

    def test_stale_source_rejected(self):
        (self.root/'tests/hip_aggregate_fixture.hip').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'input changed'):qualifier.compiler_inputs(self.compiler)

    def test_required_identities_and_manifests(self):
        original=json.loads(json.dumps(self.receipt))
        for name in ['inputs','artifacts']:
            self.receipt=json.loads(json.dumps(original));self.receipt[name]={};self.save()
            with self.assertRaisesRegex(ValueError,'nonempty'):qualifier.compiler_inputs(self.compiler)
        for path in [self.root/'tests/hip_aggregate_fixture.hip',self.root/'tools-bin/clang++-18',self.root/'.deps/sources/hip-headers/include/hip/hip_runtime.h']:
            self.receipt=json.loads(json.dumps(original));del self.receipt['inputs'][str(path)];self.save()
            with self.assertRaisesRegex(ValueError,'required source'):qualifier.compiler_inputs(self.compiler)
        self.receipt=json.loads(json.dumps(original));del self.receipt['artifacts']['linux-device.d'];self.save()
        with self.assertRaisesRegex(ValueError,'dependency manifests'):qualifier.compiler_inputs(self.compiler)

    def test_artifact_path_confinement_and_claim(self):
        for name in ['../outside',str(self.compiler/'fixture.o')]:
            self.receipt['artifacts'][name]=hashlib.sha256(b'anything').hexdigest();self.save()
            with self.assertRaisesRegex(ValueError,'outside fixture|absolute compiler'):qualifier.compiler_inputs(self.compiler)
            del self.receipt['artifacts'][name]
        self.receipt['gpu_executed']=True;self.save()
        with self.assertRaisesRegex(ValueError,'unexpected compiler'):qualifier.compiler_inputs(self.compiler)


if __name__=='__main__':unittest.main()
