import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import dense_layer_reference as ref
from build_llm_dense_layer_test import bind_reference
class Export(unittest.TestCase):
 def test_inventory(self):
  rows=ref.inventory();self.assertEqual(len(rows),11084);self.assertEqual(rows[-1]['offset']+rows[-1]['bytes'],154392576)
  self.assertEqual(sum(x['stage'].startswith('cache') for x in rows),38)
 def test_stream_integrity(self):
  rows=[dict(request=0,case='sequence',position=0,scale='1.f',stage='out-00',dtype='f32le',offset=0,bytes=4),dict(request=0,case='sequence',position=0,scale='1.f',stage='out-01',dtype='f32le',offset=4,bytes=4)]
  with tempfile.TemporaryDirectory() as folder,patch.object(ref,'BYTES',8),patch.object(ref,'inventory',return_value=rows):
   raw=Path(folder)/'reference.bin';manifest=Path(folder)/'reference.json';raw.write_bytes(bytes(8));ref.write_manifest(raw,manifest);ref.validate(raw,manifest)
   accepted=ref.validate(raw,manifest);bind_reference(raw,manifest,accepted)
   raw.write_bytes(b'1'+bytes(7))
   with self.assertRaises(ValueError):bind_reference(raw,manifest,accepted)
   raw.write_bytes(bytes(8));saved=manifest.read_bytes();manifest.write_bytes(saved+b' ')
   with self.assertRaises(ValueError):bind_reference(raw,manifest,accepted)
   manifest.write_bytes(saved)
   original=manifest.read_text()
   for modify in [lambda d:d['records'][1].update(offset=0),lambda d:d['records'][0].update(stage='unknown'),lambda d:d['records'].pop(),lambda d:d.update(unknown=True)]:
    data=json.loads(original);modify(data);manifest.write_text(json.dumps(data))
    with self.assertRaises(ValueError):ref.validate(raw,manifest)
   manifest.write_text(original.replace('"schema": 1','"schema": 1, "schema": 1'))
   with self.assertRaises(ValueError):ref.validate(raw,manifest)
   manifest.write_text(original)
   for blob in [bytes(7),bytes(9),b'1'+bytes(7)]:
    raw.write_bytes(blob)
    with self.assertRaises(ValueError):ref.validate(raw,manifest)
if __name__=='__main__':unittest.main()
