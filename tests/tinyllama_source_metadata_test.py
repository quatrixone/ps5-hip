"""Bounded source metadata tests; default uses generated sidecars, never model weights."""
import argparse,json,pathlib,subprocess,tempfile
from tinyllama_source_metadata_fixture import generate,run as mutations
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-source',type=pathlib.Path,default=ROOT/'.deps/sources/llama-client')
    p.add_argument('--cpu-build',type=pathlib.Path,required=True,help='Existing authenticated CPU library build directory')
    p.add_argument('--original-metadata',type=pathlib.Path,help='Optional original HF metadata folder; weights are never read')
    p.add_argument('--vocab-fixture',type=pathlib.Path,help='Optional zero-tensor GGUF used only for comparison')
    p.add_argument('--sanitize',action='store_true')
    a=p.parse_args()
    if bool(a.original_metadata)!=bool(a.vocab_fixture):p.error('original metadata and vocab fixture must be supplied together')
    lib=a.cpu_build.resolve()/'cmake/bin'
    with tempfile.TemporaryDirectory(prefix='tinyllama-source-metadata-') as tmp:
        tmp=pathlib.Path(tmp);binary=tmp/'test'
        cmd=['clang++-18','-std=c++17','-O1','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-I'+str(ROOT/'ports/ggml'),*['-I'+str(a.llama_source.resolve()/n) for n in ('include','ggml/include','vendor')],str(ROOT/'tests/tinyllama_source_metadata_test.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-o',str(binary)]
        if a.sanitize:cmd[1:1]=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
        subprocess.run(cmd,check=True)
        def run(argv):
            result=subprocess.run([str(binary),*map(str,argv)],capture_output=True,text=True,timeout=30)
            if result.returncode or any(m in result.stderr for m in ('AddressSanitizer','UndefinedBehaviorSanitizer','LeakSanitizer','runtime error:')):raise RuntimeError(result.stderr)
            return result.stdout
        print(run([]),end='');source=tmp/'source';generate(binary,source)
        mutations(binary,source);assert run(['--vocab-only',source])=='VOCAB_ONLY_PASS\n';print('PASS synthetic original-metadata public vocab_only API')
        if a.original_metadata:
            print(run([a.original_metadata.resolve(),a.vocab_fixture.resolve()]),end='')
            mutations(binary,a.original_metadata.resolve())
            cmd[cmd.index(str(ROOT/'tests/tinyllama_source_metadata_test.cpp'))]=str(ROOT/'tests/tinyllama_source_vocab_fixture.cpp');subprocess.run(cmd,check=True)
            actual=run([a.original_metadata.resolve()])
            cmd[cmd.index(str(ROOT/'tests/tinyllama_source_vocab_fixture.cpp'))]=str(ROOT/'tests/llm_vocab_tokenize.cpp');subprocess.run(cmd,check=True)
            assert actual==run([a.vocab_fixture.resolve()])
            rows=[json.loads(line) for line in actual.splitlines()];assert len(rows)==32 and sum(len(row['tokens']) for row in rows)==236
            print('PASS original fixture:32 cases,236 tokens, byte-identical')
    print('Note: prebuilt llama/ggml libraries are not sanitizer-instrumented.')

if __name__=='__main__':main()
