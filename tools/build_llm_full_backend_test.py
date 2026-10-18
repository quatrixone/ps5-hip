"""Focused real llama scheduler forwarding to mocked whole-model HIP backend."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build(cpu,model,output,helper=False):
 cpu,model,output=(Path(p).resolve() for p in (cpu,model,output))
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh workspace results directory required')
 client=ROOT/'.deps/sources/llama-client';lib=cpu/'cmake/bin'
 sources=('tools/build_llm_full_backend_test.py','tests/llm_full_backend_test.cpp','tests/llm_layer_hip_mock.h','ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/full_qualification.h','ports/ggml/full_dispatch.h','ports/ggml/full_graph.h','ports/ggml/layer_dispatch.h','ports/ggml/model_layer.h','ports/ggml/layer_qualification.h','ports/ggml/prefix_dispatch.h')
 if helper:sources+=('ports/ggml/generation.h','ports/ggml/generation_cases.def')
 if helper:sources+=('ports/ggml/native_full.h','tests/llm_native_full_test.cpp')
 inputs={str(ROOT/n):sha(ROOT/n) for n in sources}
 for p in (model,Path('/usr/bin/clang++-18'),*lib.glob('lib*.so*')):
  if p.is_file():inputs[str(p.resolve())]=sha(p)
 output.mkdir(parents=True);commands=[]
 flags=['-std=c++17','-O1','-gline-tables-only','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-fsanitize=address,undefined','-D__HIP_PLATFORM_AMD__']
 flags+=['-I'+str(p) for p in (ROOT/'ports/ggml',ROOT/'ports/hip',ROOT/'tests',ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',client/'include',client/'ggml/include')]
 flags+=['-isystem',str(client/'ggml/src')]
 scrub=('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','ASAN_SYMBOLIZER_PATH')
 env={k:v for k,v in os.environ.items() if k not in scrub};env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 def run(label,argv,expected=0):
  argv=list(map(str,argv));p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=600)
  for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:(output/(label+'.'+suffix)).write_bytes(data)
  commands.append(dict(label=label,argv=argv,exit_code=p.returncode,expected_exit_code=expected));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if p.returncode!=expected:raise RuntimeError(label+' failed; inspect retained stderr')
  if any(s in p.stderr for s in (b'ERROR: AddressSanitizer',b'LeakSanitizer:',b'runtime error:')):raise RuntimeError('sanitizer diagnostics')
 run('compile',['/usr/bin/clang++-18',*flags,ROOT/('tests/llm_native_full_test.cpp' if helper else 'tests/llm_full_backend_test.cpp'),ROOT/'ports/ggml/ps5_backend.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lllama','-lggml','-lggml-cpu','-lggml-base','-ldl','-o',output/'full-backend'])
 for mode in (('sequence','smoke','generation','generation-eos','corpus') if helper else ('whole','split')):run(mode,[output/'full-backend',model,*([] if mode=='sequence' else [mode])])
 if helper:run('corpus-eos',[output/'full-backend',model,'corpus-eos'],expected=-6)
 if helper and b'TEST_TERMINAL backend-invariant' not in (output/'corpus-eos.stderr').read_bytes():raise RuntimeError('wrong corpus EOS rejection')
 if helper:run('clock-regression',[output/'full-backend',model,'clock-regress'],expected=-6)
 if helper and b'TEST_TERMINAL backend-invariant' not in (output/'clock-regression.stderr').read_bytes():raise RuntimeError('wrong clock rejection')
 for n,d in inputs.items():
  if sha(n)!=d:raise RuntimeError('input changed during focused test: '+n)
 corpus_requests=0
 if helper:
  matches=re.findall(r'^PASS native corpus helper mocked: cases=5 resets=10 requests=([0-9]+) files=([0-9]+); GPU/numerics unproven$',(output/'corpus.stdout').read_text(),re.M)
  if len(matches)!=1 or int(matches[0][0])<160 or int(matches[0][1])!=int(matches[0][0])*24+11:raise RuntimeError('corpus completion/inventory')
  corpus_requests=int(matches[0][0])
 report=dict(status='PASS',classification='focused-native-full-helper-mocked-HIP' if helper else 'focused-actual-full-model-scheduler-mocked-HIP',inputs=inputs,commands=commands,modes=['sequence','smoke','generation','generation-eos','corpus','corpus-eos'] if helper else ['whole','split'],requests_per_mode={'sequence':514,'smoke':3,'generation':16,'generation-eos':10,'corpus':corpus_requests} if helper else 3,sequence={'sequence':[512,'reset',2],'smoke':[0,1,'reset',0],'generation':['prompt5','greedy4','reset','repeat'],'generation-eos':['prompt5','eos','reset','repeat'],'corpus':['five prompts','greedy16','reset/repeat each'],'corpus-eos':['prompt5','eos','required rejection']} if helper else [0,1,'reset',0],gpu_executed=False,numerical_verified=False,limits=['Transport mocks perform no tensor arithmetic; controlled zero/EOS and input-dependent corpus logits prove only transport, feedback and reset','Existing CPU libraries are not sanitizer-instrumented','Focused test input identities; not complete deployment prerequisite provenance'])
 (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS native helper sequence514, smoke3, generation16, EOS10, five-case corpus, corpus EOS rejection and clock rejection' if helper else 'PASS full-model scheduler whole/split modes, committed KV, reset, no CPU tensor arithmetic')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cpu-build','model','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--native-helper',action='store_true')
 a=p.parse_args();build(a.cpu_build,a.model,a.output,a.native_helper)
