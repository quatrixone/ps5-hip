"""Focused real llama scheduler forwarding to mocked whole-model HIP backend."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build(cpu,model,output):
 cpu,model,output=(Path(p).resolve() for p in (cpu,model,output))
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh workspace results directory required')
 client=ROOT/'.deps/sources/llama-client';lib=cpu/'cmake/bin'
 sources=('tools/build_llm_full_backend_test.py','tests/llm_full_backend_test.cpp','tests/llm_layer_hip_mock.h','ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/full_qualification.h','ports/ggml/full_dispatch.h','ports/ggml/full_graph.h','ports/ggml/layer_dispatch.h','ports/ggml/model_layer.h','ports/ggml/layer_qualification.h','ports/ggml/prefix_dispatch.h')
 inputs={str(ROOT/n):sha(ROOT/n) for n in sources}
 for p in (model,Path('/usr/bin/clang++-18'),*lib.glob('lib*.so*')):
  if p.is_file():inputs[str(p.resolve())]=sha(p)
 output.mkdir(parents=True);commands=[]
 flags=['-std=c++17','-O1','-gline-tables-only','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-fsanitize=address,undefined','-D__HIP_PLATFORM_AMD__']
 flags+=['-I'+str(p) for p in (ROOT/'ports/ggml',ROOT/'ports/hip',ROOT/'tests',ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',client/'include',client/'ggml/include')]
 flags+=['-isystem',str(client/'ggml/src')]
 scrub=('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','ASAN_SYMBOLIZER_PATH')
 env={k:v for k,v in os.environ.items() if k not in scrub};env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 def run(label,argv):
  argv=list(map(str,argv));p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=600)
  for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:(output/(label+'.'+suffix)).write_bytes(data)
  commands.append(dict(label=label,argv=argv,exit_code=p.returncode));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if p.returncode:raise RuntimeError(label+' failed; inspect retained stderr')
  if any(s in p.stderr for s in (b'ERROR: AddressSanitizer',b'LeakSanitizer:',b'runtime error:')):raise RuntimeError('sanitizer diagnostics')
 run('compile',['/usr/bin/clang++-18',*flags,ROOT/'tests/llm_full_backend_test.cpp',ROOT/'ports/ggml/ps5_backend.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lllama','-lggml','-lggml-cpu','-lggml-base','-ldl','-o',output/'full-backend'])
 for mode in ('whole','split'):run(mode,[output/'full-backend',model,mode])
 for n,d in inputs.items():
  if sha(n)!=d:raise RuntimeError('input changed during focused test: '+n)
 report=dict(status='PASS',classification='focused-actual-full-model-scheduler-mocked-HIP',inputs=inputs,commands=commands,modes=['whole','split'],requests_per_mode=3,sequence=[0,1,'reset',0],gpu_executed=False,numerical_verified=False,limits=['Transport mocks perform no tensor arithmetic; zero logits prove only output path','Existing CPU libraries are not sanitizer-instrumented','Focused test input identities; not complete deployment prerequisite provenance'])
 (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS full-model scheduler whole/split modes, committed KV, reset, no CPU tensor arithmetic')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cpu-build','model','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();build(a.cpu_build,a.model,a.output)
