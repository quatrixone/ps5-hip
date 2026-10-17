"""Focused actual-model layer/prefix integration with mocked HIP, no native package."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess
import source_llm_layer_log as layer_log
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def build(cpu, model, output):
    cpu, model, output = (Path(p).resolve() for p in (cpu, model, output))
    if output.exists() or not output.is_relative_to(ROOT / 'results'):
        raise ValueError('fresh workspace results directory required')
    output.mkdir(parents=True)
    (output/'raw').mkdir()
    client = ROOT / '.deps/sources/llama-client'
    lib = cpu / 'cmake/bin'
    flags = ['-std=c++17', '-O1', '-gline-tables-only', '-Wall', '-Wextra', '-Werror',
             '-Wno-misleading-indentation', '-fsanitize=address,undefined', '-D__HIP_PLATFORM_AMD__']
    flags += ['-I' + str(p) for p in (ROOT/'ports/ggml', ROOT/'ports/hip', ROOT/'tests',
              ROOT/'.deps/sources/hip-headers/include', ROOT/'.deps/sources/clr/hipamd/include',
              client/'include', client/'ggml/include')]
    flags += ['-isystem', str(client/'ggml/src')]
    scrub = ('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH',
             'OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX',
             'ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','ASAN_SYMBOLIZER_PATH')
    env = {k:v for k,v in os.environ.items() if k not in scrub}
    env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    commands = []
    def run(label, argv):
        args = list(map(str, argv))
        process = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=600)
        (output/(label+'.stdout')).write_bytes(process.stdout)
        (output/(label+'.stderr')).write_bytes(process.stderr)
        commands.append(dict(label=label, argv=args, exit_code=process.returncode))
        (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if process.returncode:
            raise RuntimeError(label+' failed; inspect retained stderr')
    units = {'backend':'ports/ggml/ps5_backend.cpp', 'layer':'tests/llm_layer_backend_test.cpp',
             'prefix':'tests/llm_prefix_backend_test.cpp', 'dispatch':'tests/llm_prefix_dispatch_test.cpp', 'helper':'tests/llm_native_layer_test.cpp'}
    sources = {str(ROOT/n):sha(ROOT/n) for n in (*units.values(), 'ports/ggml/model_layer.h',
               'ports/ggml/full_dispatch.h','ports/ggml/full_graph.h','ports/ggml/full_qualification.h',
               'ports/ggml/layer_dispatch.h','ports/ggml/layer_qualification.h','ports/ggml/prefix_dispatch.h',
               'ports/ggml/native_layer.h','tests/llm_layer_hip_mock.h','tools/build_llm_layer_host_test.py')}
    for name, source in units.items():
        run(name+'-compile', ['/usr/bin/clang++-18', *flags, '-c', ROOT/source, '-o', output/(name+'.o')])
    for name in ('dispatch','prefix','layer','helper'):
        objects=[output/(name+'.o')]+([output/'backend.o'] if name!='dispatch' else [])
        run(name+'-link', ['/usr/bin/clang++-18','-fsanitize=address,undefined',*objects,
            '-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lllama','-lggml','-lggml-cpu','-lggml-base','-o',output/name])
        run(name,[output/name]+([model] if name!='dispatch' else [])+([output/'raw'] if name=='helper' else []))
    for path,digest in sources.items():
        if sha(Path(path))!=digest: raise RuntimeError('source changed during focused test: '+path)
    fragment=layer_log.check_fragment((output/'helper.stdout').read_bytes())
    raw={p.name:p.stat().st_size for p in (output/'raw').iterdir()}
    expected={}
    sizes=[8192,8192,8192,8192,8192,1024,1024,1024,None,None,8192,8192,8192,8192,8192,8192,22528,22528,22528,8192,8192]
    for run,positions in ((0,(0,1,255,256,511)),(1,(0,1))):
        for pos in positions:
            extent=256 if pos<256 else 512
            for i,size in enumerate(sizes):expected[f'run-{run}-pos-{pos}-out-{i:02d}.f32le']=size or 128*extent
        for cache in ('k','v'):expected[f'run-{run}-cache-{cache}.f16le']=262144
    if raw!=expected:raise RuntimeError('layer raw inventory differs')
    for name in raw:
        if any((output/'raw'/name).read_bytes()):raise RuntimeError('mock unexpectedly produced tensor arithmetic')
    receipt=dict(status='PASS' ,classification='focused-actual-model-layer-mocked-HIP',sources=sources,
        commands=commands,cpu_build=str(cpu),model_path=str(model),sanitizers=['address','undefined'],
        helper_fragment=fragment,raw_files=151,raw_bytes=sum(raw.values()),requests=514,sequence=[512,2],prefix_regression=True,gpu_executed=False,numerical_verified=False,
        limits=['Mocked public HIP; no GPU arithmetic','Existing CPU libraries unchanged and not sanitizer instrumented',
                'Focused integration only; full prerequisite provenance requires milestone builder before deployment'])
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS focused layer514 requests, reset, preflight negatives and unchanged prefix regression')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-build','model','output'): parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args();build(args.cpu_build,args.model,args.output)
