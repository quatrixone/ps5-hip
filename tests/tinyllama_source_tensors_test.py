"""Exercise strict TinyLlama source mapping with metadata-only fixtures."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TinyLlamaSourceTensorsTest(unittest.TestCase):
    def test_mapping_and_chunked_permutation(self):
        vendor = next((parent / '.deps/sources/llama-client/vendor'
                       for parent in (ROOT, *ROOT.parents)
                       if (parent / '.deps/sources/llama-client/vendor/nlohmann/json.hpp').is_file()), None)
        self.assertIsNotNone(vendor, 'Run make deps-fetch for the existing llama-client JSON dependency')
        with tempfile.TemporaryDirectory(prefix='ps5hip-source-tensors-') as directory:
            temporary = Path(directory)
            binary = temporary / 'tensor-test'
            command = [*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O1',
                       '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                       '-fno-sanitize-recover=all', '-fno-omit-frame-pointer',
                       '-I', str(ROOT / 'ports/ggml'), '-I', str(vendor),
                       str(ROOT / 'tests/tinyllama_source_tensors_test.cpp'), '-o', str(binary)]
            subprocess.run(command, check=True, timeout=180)
            fixtures = ROOT / 'tests/fixtures'
            result = subprocess.run([str(binary), str(fixtures / 'tinyllama_source_config.json'),
                                     str(fixtures / 'tinyllama_source_header.json'),
                                     str(temporary / 'weights'), str(temporary / 'mapping.json')],
                                    check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.stdout.strip(), '31 checks PASS; actual header inventory=201; Q heads=32 K heads=4')
            self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
