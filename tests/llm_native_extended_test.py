"""Exercise production native generation/sink with a tiny mocked llama ABI, no model."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class NativeExtended(unittest.TestCase):
    def test_production_generation_and_capture(self):
        source = (ROOT/'ports/ggml/native_full.h').read_text()
        source = source[source.index('namespace m0 {'):]
        fixture = (ROOT/'tests/llm_native_extended_fixture.cpp').read_text()
        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            unit = folder/'test.cpp'
            unit.write_text(fixture.replace('// PRODUCTION_HELPER', source))
            binary = folder/'test'
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',
                            '-I'+str(ROOT/'ports/ggml'),str(unit),'-o',str(binary)],check=True)
            for mode in ('outer-extended','outer-short','outer-corpus','extended','short','corpus','bad-prompt','bad-stop','bad-reset','overflow'):
                with self.subTest(mode=mode):
                    result = subprocess.run([str(binary),mode],capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('PASS '+mode,result.stdout)


if __name__ == '__main__':
    unittest.main()
