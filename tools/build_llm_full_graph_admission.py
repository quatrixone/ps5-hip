"""Test full-model metadata admission against authenticated successful CPU traces."""
import argparse,json,os,subprocess
from pathlib import Path
import build_llm_executed_graph as provenance
from build_llm_full_graph_trace import authenticate_full_graph_trace, CHECKPOINTS
ROOT=Path(__file__).resolve().parents[1]

def build(trace,output):
 trace,output=Path(trace).resolve(),Path(output).resolve()
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results output required')
 data,inputs=authenticate_full_graph_trace(trace)
 for name in ('tools/build_llm_full_graph_admission.py','tests/llm_full_graph_admission.cpp','ports/ggml/full_graph.h','ports/ggml/model_layer.h'):
  path=ROOT/name;provenance.merge(inputs,path,provenance.sha(path))
 compiler=Path('/usr/bin/clang++-18');provenance.merge(inputs,compiler,provenance.sha(compiler))
 output.mkdir(parents=True);client=ROOT/'.deps/sources/llama-client';lib=Path(data['cpu_build'])/'cmake/bin'
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 commands=[]
 def run(label,args):
  args=list(map(str,args));result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,timeout=240)
  for suffix,raw in [('stdout',result.stdout),('stderr',result.stderr)]:(output/(label+'.'+suffix)).write_bytes(raw)
  commands.append({'label':label,'argv':args,'exit_code':result.returncode});(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');result.check_returncode()
 includes=['-I'+str(p) for p in (ROOT/'ports/ggml',client/'ggml/include',client/'ggml/src',client/'vendor')]
 flags=['-std=c++17','-O1','-g','-fsanitize=address,undefined',*includes]
 source=ROOT/'tests/llm_full_graph_admission.cpp'
 run('headers',[compiler,*flags,'-M','-MF',output/'headers.d',source])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',[compiler,*flags,'-MD','-MF',output/'compile.d',source,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml-base','-o',output/'admission'])
 if provenance.dependencies(output/'headers.d')!=provenance.dependencies(output/'compile.d'):raise ValueError('header closure differs')
 run('admission',[output/'admission',*[trace/'raw'/f'run-{r}-pos-{p}.graph.json' for r,p in CHECKPOINTS]])
 provenance.recheck(inputs)
 report={'schema':1,'status':'PASS','classification':'host-full-model-metadata-admission','trace_receipt_sha256':provenance.sha(trace/'receipt.json'),'graphs':7,'nodes_per_graph':801,'active_nodes_per_graph':798,'mutation_classes':10,'inputs':inputs,'commands':commands,'artifacts':{p.name:provenance.sha(p) for p in output.iterdir() if p.is_file()},'gpu_executed':False,'limits':['Metadata and topology only: no GPU ownership, live controls or scheduler split authorization.','Does not establish full-model GPU execution or inference.']}
 (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 build(a.trace,a.output);print('PASS full-model graph metadata admission; GPU execution unproven')
