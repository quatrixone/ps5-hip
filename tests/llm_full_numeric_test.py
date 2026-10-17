import sys
from pathlib import Path
import struct
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_llm_full_numeric_test as numeric

class Comparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "raw"; self.raw.mkdir()
        self.reference = Path(self.temp.name) / "reference"; self.reference.mkdir()
        for name, size in numeric.FILES.items():
            for folder in (self.raw, self.reference):
                (folder / name).write_bytes(struct.pack("<f", 1.0) * (size // 4))

    def test_equal_and_complete(self):
        self.assertEqual(numeric.status(numeric.compare(self.raw, self.reference)), ("PASS", 0))

    def test_numerical_failure_is_nonzero(self):
        path = self.raw / "run-0-pos-1-logits.f32le"
        path.write_bytes(struct.pack("<f", 2.0) + path.read_bytes()[4:])
        self.assertEqual(numeric.status(numeric.compare(self.raw, self.reference)), ("NUMERICAL_FAIL", 1))

    def test_missing_extra_and_wrong_size(self):
        name = next(iter(numeric.FILES)); path = self.raw / name; saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError): numeric.compare(self.raw, self.reference)
        path.write_bytes(saved[:-4])
        with self.assertRaises(ValueError): numeric.compare(self.raw, self.reference)
        path.write_bytes(saved); (self.raw / "extra").write_bytes(b"")
        with self.assertRaises(ValueError): numeric.compare(self.raw, self.reference)

    def test_nonfinite_reference_and_result(self):
        name = next(iter(numeric.FILES))
        for folder in (self.raw, self.reference):
            path = folder / name; saved = path.read_bytes()
            path.write_bytes(struct.pack("<f", float("nan")) + saved[4:])
            with self.assertRaises(ValueError): numeric.compare(self.raw, self.reference)
            path.write_bytes(saved)

    def test_reset_drift_is_rejected_even_within_budget(self):
        path = self.raw / "run-1-pos-0-layer-0.f32le"
        path.write_bytes(struct.pack("<f", 1.000001) + path.read_bytes()[4:])
        with self.assertRaisesRegex(ValueError, "reset"): numeric.compare(self.raw, self.reference)

    def test_symlink_rejected(self):
        name = next(iter(numeric.FILES)); path = self.raw / name; path.unlink()
        path.symlink_to(self.reference / name)
        with self.assertRaises(ValueError): numeric.compare(self.raw, self.reference)

    def test_incomplete_status_rejected(self):
        with self.assertRaises(ValueError): numeric.status([])

if __name__ == "__main__":
    unittest.main()
