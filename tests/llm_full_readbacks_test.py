import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import verify_llm_full_readbacks as verifier


class FullReadbacks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.reference = self.root / 'reference'; self.raw = self.root / 'raw'
        (self.reference / 'raw').mkdir(parents=True); self.raw.mkdir()
        for name, size in verifier.FILES.items():
            (self.reference / 'raw' / name).write_bytes(bytes(size)); (self.raw / name).write_bytes(bytes(size))
        self.data = {'receipt_sha256': 'a' * 64, 'model': {'sha256': 'b' * 64}}
        self.auth = patch.object(verifier, 'authenticate_full_model_reference', return_value=(self.data, {}))
        self.auth.start(); self.addCleanup(self.auth.stop)

    def verify(self):
        return verifier.verify(self.reference, self.raw, self.root / 'comparison.json')

    def test_all_boundaries_and_claims(self):
        result = self.verify()
        self.assertEqual((result['layers'], result['raw_files'], result['raw_bytes']), (22, 72, 949248))
        self.assertFalse(result['gpu_origin_established']); self.assertFalse(result['inference'])
        self.assertEqual(json.loads((self.root / 'comparison.json').read_text()), result)

    def test_logits_error(self):
        name = 'run-0-pos-1-logits.f32le'
        (self.raw / name).write_bytes(struct.pack('<f', 1.) + bytes(verifier.FILES[name] - 4))
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.verify()

    def test_reset_must_be_bitwise(self):
        name = 'run-1-pos-0-layer-0.f32le'
        (self.raw / name).write_bytes(struct.pack('<f', 1e-8) + bytes(verifier.FILES[name] - 4))
        with self.assertRaisesRegex(ValueError, 'reset'):
            self.verify()

    def test_unexpected_inventory(self):
        (self.raw / 'unexpected').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.verify()

    def test_changed_reference_closure(self):
        path = self.root / 'identity'; path.write_bytes(b'changed')
        with patch.object(verifier, 'authenticate_full_model_reference', return_value=(self.data, {str(path): 'a' * 64})):
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                self.verify()

    def test_size_checked_before_hashing(self):
        (self.raw / next(iter(verifier.FILES))).write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'kind/size'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
