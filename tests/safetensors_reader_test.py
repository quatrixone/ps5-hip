"""Compile the bounded reader against the fetched client's existing JSON header."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SafetensorsReaderTest(unittest.TestCase):
    def test_bounded_reader_with_sanitizers(self):
        vendor = next((parent / '.deps/sources/llama-client/vendor'
                       for parent in (ROOT, *ROOT.parents)
                       if (parent / '.deps/sources/llama-client/vendor/nlohmann/json.hpp').is_file()), None)
        self.assertIsNotNone(vendor, 'Run make deps-fetch to obtain the existing llama-client JSON dependency')
        with tempfile.TemporaryDirectory(prefix='ps5hip-safetensors-') as directory:
            temporary = Path(directory)
            binary = temporary / 'reader-test'
            command = [*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O1',
                       '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                       '-fno-sanitize-recover=all', '-fno-omit-frame-pointer',
                       '-I', str(ROOT / 'ports/ggml'), '-I', str(vendor),
                       str(ROOT / 'tests/safetensors_reader_test.cpp'), '-o', str(binary)]
            subprocess.run(command, check=True, timeout=180)
            result = subprocess.run([str(binary), str(temporary / 'fixture.safetensors')],
                                    check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.stdout.strip(), '91 checks PASS')
            self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
