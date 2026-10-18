import sys
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_layer_fixture as m
class LayerFixtureTest(unittest.TestCase):
 def test_inventory(self):
  self.assertEqual(len(m.SYMBOLS),16);self.assertEqual(len(set(m.SYMBOLS)),16)
 def test_reject_outside_output(self):
  with self.assertRaisesRegex(ValueError,'outside'):m.build('/unused','/tmp/untrusted-layer-output')
 def test_reject_wrong_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'receipt.json').write_text('{"classification":"wrong"}')
   with self.assertRaisesRegex(ValueError,'invalid'):m.fixture_inputs(d)
 def test_undefined_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'undefined.stdout').write_text('U forbidden')
   with self.assertRaisesRegex(ValueError,'undefined'):m.contracts(d)
 def test_missing_symbol_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for n in ('undefined.stdout','notes.stdout','isa.stdout','admission.stdout'): (p/n).write_text('')
   (p/'layer.hsaco').write_bytes(b'fake')
   (p/'isa.stdout').write_text('<layer_q4_cpu>:\n global_load_ubyte '+ 'global_load_ubyte '*15)
   with patch.object(m,'audit_object',return_value={}):
    with self.assertRaisesRegex(ValueError,'admission'):m.contracts(d)
 def test_artifact_identity_and_missing_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);artifacts={}
   for n in m.REQUIRED:
    data=b'[]' if n=='commands.json' else b'fixture'
    (folder/n).write_bytes(data);artifacts[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   report=dict(classification=m.CLASSIFICATION,status='PASS',native_executed=False,original_fixture=str(folder/'parent'),artifacts=artifacts,commands=[])
   (folder/'receipt.json').write_text(json.dumps(report))
   with patch.object(m,'original_inputs',return_value={}),patch.object(m,'checked_inputs',return_value={}),patch.object(m,'checked_tools',return_value={}):
    (folder/'layer.hsaco').write_bytes(b'wrong')
    with self.assertRaisesRegex(ValueError,'artifact identity'):m.fixture_inputs(folder)
    del report['artifacts']['layer.hsaco'];(folder/'receipt.json').write_text(json.dumps(report))
    with self.assertRaisesRegex(ValueError,'required artifact'):m.fixture_inputs(folder)
if __name__=='__main__':unittest.main()

