"""Compile the production path reader and check its boundary against packaging."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from external_model import DEFAULT_MODEL_PATH, encode_model_path


class ModelPathTest(unittest.TestCase):
    def test_native_reader_matches_packaging_boundary(self):
        source = (ROOT / 'tools/public_hip_llm_client_console_title.cpp').read_text()
        reader = source.split('static std::string configured_model_path(', 1)[1].split('\nint main()', 1)[0]
        harness = '''#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
[[noreturn]] void ps5hip_clr_gpu_fail(const char* r){throw std::runtime_error(r);}
static std::string configured_model_path(''' + reader + '''
int main(int argc,char** argv){
 try{std::cout<<configured_model_path(argv[1]);return 0;}catch(...){return 2;}
}
'''
        cases = [DEFAULT_MODEL_PATH, '/data/models/model.gguf',
                 '/user/homebrew/' + 'a' * 220 + '.gguf',
                 '/user/homebrew/' + 'a' * 221 + '.gguf',
                 '', '/app0/model.gguf', '/data/../model.gguf', '/data/./model.gguf',
                 '/data//model.gguf', '/data/model.gguf/', '/data/model.gguf\n',
                 '/data/model.gguf\x00', '/data/café.gguf', '/data/model.bin',
                 '/data/' + 'a' * 230 + '.gguf', '/data/a\\b.gguf']
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            cpp = directory / 'reader.cpp'
            binary = directory / 'reader'
            config = directory / 'model-path.txt'
            cpp.write_text(harness)
            subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(binary)], check=True)
            for path in cases:
                with self.subTest(path=path):
                    try:
                        encode_model_path(path)
                        valid = True
                    except ValueError:
                        valid = False
                    config.write_bytes(path.encode('utf-8'))
                    result = subprocess.run([str(binary), str(config)], capture_output=True)
                    self.assertEqual(result.returncode, 0 if valid else 2)
                    if valid:
                        self.assertEqual(result.stdout, path.encode('ascii'))
            config.unlink()
            self.assertEqual(subprocess.run([str(binary), str(config)]).returncode, 2)


if __name__ == '__main__':
    unittest.main()
