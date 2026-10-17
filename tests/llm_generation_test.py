"""Compile and exercise the bounded CPU sampling driver; no model/GPU claim."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class Generation(unittest.TestCase):
    def test_portable_driver(self):
        compiler=shutil.which('clang++-18') or shutil.which('clang++')
        self.assertIsNotNone(compiler, 'clang++ is required')
        with tempfile.TemporaryDirectory(prefix='ps5-hip-generation-') as tmp:
            binary=Path(tmp)/'generation'
            env={k:v for k,v in os.environ.items() if k not in ('LD_PRELOAD','LD_AUDIT','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH')}
            env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
            subprocess.run([compiler,'-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(ROOT/'ports/ggml'),str(ROOT/'tests/llm_generation_test.cpp'),'-o',str(binary)],check=True,env=env,timeout=60)
            subprocess.run([str(binary)],check=True,env=env,timeout=30)

if __name__=='__main__':unittest.main()
