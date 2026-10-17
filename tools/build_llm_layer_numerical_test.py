"""Focused real-backend numerical HIP test double; never GPU qualification."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(cpu,model,output,steps=512):
 cpu,model,output=(Path(p).resolve() for p in (cpu,model,output))
 if steps not in (4,512) or output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results directory and 4 or512 steps required')
 client=ROOT/'.deps/sources/llama-client';lib=cpu/'cmake/bin'
 names=('tools/build_llm_layer_numerical_test.py','tests/llm_layer_numeric_test.cpp','tests/llm_layer_numeric_mock.h','kernels/llm_layer.hip','ports/ggml/ps5_backend.cpp','ports/ggml/model_layer.h','ports/ggml/layer_dispatch.h','ports/ggml/layer_qualification.h','ports/ggml/prefix_dispatch.h')
 inputs={str(ROOT/n):sha(ROOT/n) for n in names}
 for p in [model,Path('/usr/bin/clang++-18'),*lib.glob('lib*.so*')]:
  if p.is_file():inputs[str(p.resolve())]=sha(p)
 output.mkdir(parents=True);(output/'outputs').mkdir()
 flags=['-std=c++17','-O2','-ffp-contract=off','-D__HIP_PLATFORM_AMD__']+['-I'+str(p) for p in (ROOT/'ports/ggml',ROOT/'ports/hip',ROOT/'tests',ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',client/'include',client/'ggml/include',client/'ggml/src')]
 env=dict(os.environ)
 for n in ('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH'):env.pop(n,None)
 commands=[]
 def run(label,a):
  a=list(map(str,a));p=subprocess.run(a,cwd=ROOT,env=env,capture_output=True,timeout=900)
  for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:(output/(label+'.'+suffix)).write_bytes(data)
  commands.append(dict(label=label,argv=a,exit_code=p.returncode));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');p.check_returncode()
 run('build',['/usr/bin/clang++-18',*flags,ROOT/'tests/llm_layer_numeric_test.cpp',ROOT/'ports/ggml/ps5_backend.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lllama','-lggml','-lggml-cpu','-lggml-base','-o',output/'numeric'])
 run('run',[output/'numeric',model,output/'outputs',str(steps)])
 for n,d in inputs.items():
  if sha(n)!=d:raise ValueError('input changed: '+n)
 receipt=dict(classification='focused-actual-layer-numerical-HIP-mock',status='PASS',requests=steps+2,inputs=inputs,commands=commands,gpu_executed=False,native_executed=False,outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (output/'outputs').iterdir()},limits=['Focused numerical mock; no complete compiler/header provenance.','Host libm replaces GPU OCML.','Independent CPU/readback comparison is a separate mandatory acceptance step.'])
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--steps',type=int,choices=(4,512),default=512);a=p.parse_args();r=build(a.cpu_build,a.model,a.output,a.steps);print('PASS numerical mock execution requests='+str(r['requests'])+'; independent comparison required')
