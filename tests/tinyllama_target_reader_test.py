"""Portable bounded reader tests using existing ggml headers/libraries; no model needed."""
import argparse,pathlib,subprocess,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--cpu-build',type=pathlib.Path,default=ROOT/'build/llm-cpu-reference-v3')
 parser.add_argument('--client',type=pathlib.Path,default=ROOT/'.deps/sources/llama-client')
 args=parser.parse_args();libs=args.cpu_build.resolve()/'cmake/bin';client=args.client.resolve()
 with tempfile.TemporaryDirectory(prefix='tinyllama-target-reader-') as folder:
  folder=pathlib.Path(folder);binary=folder/'test'
  subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'ports/ggml'),'-I'+str(client/'vendor'),'-I'+str(client/'ggml/include'),str(ROOT/'tests/tinyllama_target_reader_test.cpp'),'-L'+str(libs),'-Wl,-rpath,'+str(libs),'-lggml-base','-o',str(binary)],check=True,timeout=120)
  subprocess.run([str(binary),str(folder/'fixture.safetensors')],check=True,timeout=30)
if __name__=='__main__':main()
