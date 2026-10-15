"""Reject package configurations that the console harness cannot use."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_public_hip import validate_results_directory


class ResultsDirectoryTest(unittest.TestCase):
    def test_absolute_directory(self):
        validate_results_directory("/data/ps5-hip/results")

    def test_invalid_directories(self):
        for value in ("results", "data/results", "", "/", "/data/", "/data//results",
                      "/data/../results", "/data/./results", "/data/\nresults",
                      "/data\\results", "/" + "a" * 240):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_results_directory(value)


if __name__ == "__main__":
    unittest.main()
