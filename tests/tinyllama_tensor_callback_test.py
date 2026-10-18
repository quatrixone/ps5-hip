"""Actual ggml descriptors and bounded synthetic CPU uploads, no model weights."""
from pathlib import Path
import argparse,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpu-build',type=Path,default=ROOT/'build/llm-cpu-reference-v3')
parser.add_argument('--client',type=Path,default=ROOT/'.deps/sources/llama-client')
args=parser.parse_args();client=args.client.resolve();lib=args.cpu_build.resolve()/'cmake/bin'
with tempfile.TemporaryDirectory(prefix='tensor-callback-') as tmp:
 tmp=Path(tmp);binary=tmp/'test'
 command=['clang++-18','-std=c++17','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'ports/ggml'),'-I'+str(client/'vendor'),'-I'+str(client/'ggml/include'),str(ROOT/'tests/tinyllama_tensor_callback_test.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml-cpu','-lggml-base','-o',str(binary)]
 subprocess.run(command,check=True,timeout=120)
 subprocess.run([str(binary),str(ROOT/'tests/fixtures/tinyllama_source_config.json'),str(ROOT/'tests/fixtures/tinyllama_source_header.json'),str(tmp/'synthetic.safetensors')],check=True,timeout=30)
