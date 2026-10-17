"""Capture full-model CPU boundaries before activation reuse using existing libraries."""
import argparse,json,os,subprocess,struct,math
from pathlib import Path
import build_llm_executed_graph as provenance
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=[(0,p) for p in (0,1,255,256,511)]+[(1,p) for p in (0,1)]
LINE='FULL_MODEL_REFERENCE_PASS requests=514 checkpoints=7 boundaries=24 raw_files=168 original_cpu_decode=true GPU_execution=false\n'

def raw_files():
 return {f'run-{run}-pos-{pos}-{name}.f32le':128000 if name=='logits' else 8192 for run,pos in CHECKPOINTS for name in [*(f'layer-{i}' for i in range(22)),'norm','logits']}
FILES=raw_files()

def validate(folder):
 folder=Path(folder)
 if {p.name for p in folder.iterdir()}!=set(FILES)|{'steps.jsonl'} or any(p.is_symlink() for p in folder.iterdir()):raise ValueError('reference inventory')
 steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
 expected=[(r,p) for r,n in ((0,512),(1,2)) for p in range(n)]
 if len(steps)!=514:raise ValueError('reference step count')
 for row,(run,pos) in zip(steps,expected):
  if row!={'run':run,'position':pos,'token':(0,1,2,31999)[pos%4],'status':0,'committed_position':pos}:raise ValueError('reference committed progression')
 summary=[]
 for name,size in FILES.items():
  raw=(folder/name).read_bytes()
  if len(raw)!=size:raise ValueError('reference size')
  values=struct.unpack('<'+str(size//4)+'f',raw)
  if not all(map(math.isfinite,values)):raise ValueError('nonfinite reference')
  summary.append({'file':name,'bytes':size,'minimum':min(values),'maximum':max(values)})
 for pos in (0,1):
  for name in [*(f'layer-{i}' for i in range(22)),'norm','logits']:
   if (folder/f'run-0-pos-{pos}-{name}.f32le').read_bytes()!=(folder/f'run-1-pos-{pos}-{name}.f32le').read_bytes():raise ValueError('reset boundary differs')
 return summary

def authenticate_full_model_reference(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=json.loads(receipt.read_text())
 if data.get('status')!='PASS' or data.get('classification')!='successful-full-model-cpu-boundary-reference' or data.get('gpu_executed') is not False or data.get('requests')!=514 or data.get('raw_files')!=168:raise ValueError('trace receipt contract')
 inputs=data['inputs']
 required=[Path(__file__).resolve(),ROOT/'tests/llm_full_model_reference.cpp',ROOT/'tests/llm_full_model_reference_test.py',Path(provenance.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json']
 if any(inputs.get(str(p))!=provenance.sha(p) for p in required):raise ValueError('trace source binding')
 artifacts={str(p.relative_to(folder)):provenance.sha(p) for p in folder.rglob('*') if p.is_file() and p!=receipt}
 if any(p.is_symlink() for p in folder.rglob('*')) or artifacts!=data['artifacts']:raise ValueError('trace artifacts changed')
 model=data['model'];path=Path(model['path'])
 if not path.is_absolute() or path.stat().st_size!=model['bytes'] or inputs.get(str(path))!=model['sha256']:raise ValueError('trace model identity')
 for p,h in provenance.cpu_inputs(Path(data['cpu_build']),path).items():
  if inputs.get(p)!=h:raise ValueError('trace CPU closure')
 headers=provenance.dependencies(folder/'headers.d')
 if headers!=provenance.dependencies(folder/'compile.d') or any(inputs.get(p)!=provenance.sha(p) for p in headers):raise ValueError('trace compiler closure')
 for p,h in provenance.runtime_inputs((folder/'ldd.stdout').read_text(),Path(data['cpu_build'])/'cmake/bin',inputs).items():
  if inputs.get(p)!=h:raise ValueError('trace runtime closure')
 if (folder/'capture.stdout').read_text()!=LINE or validate(folder/'raw')!=json.loads((folder/'summary.json').read_text()):raise ValueError('trace summary/completion')
 provenance.recheck(inputs)
 if provenance.sha(receipt)!=digest:raise ValueError('trace receipt changed')
 closure=dict(inputs)
 for p,h in artifacts.items():provenance.merge(closure,folder/p,h)
 provenance.merge(closure,receipt,digest)
 return {**data,'receipt':str(receipt),'receipt_sha256':digest},closure

def build(cpu,model,output):
 cpu,model,output=map(lambda p:Path(p).resolve(),(cpu,model,output))
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results directory required')
 inputs=provenance.cpu_inputs(cpu,model);harness=ROOT/'tests/llm_full_model_reference.cpp';compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 for p in (Path(__file__).resolve(),harness,ROOT/'tests/llm_full_model_reference_test.py',Path(provenance.__file__).resolve(),compiler,Path('/usr/bin/ldd'),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json'):provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True);raw=output/'raw';raw.mkdir();commands=[]
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 def run(label,args):
  result=subprocess.run(list(map(str,args)),cwd=ROOT,env=env,capture_output=True,timeout=1800);(output/(label+'.stdout')).write_bytes(result.stdout);(output/(label+'.stderr')).write_bytes(result.stderr);commands.append({'label':label,'argv':list(map(str,args)),'exit_code':result.returncode});(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if result.returncode:raise ValueError(label+' failed')
  return result.stdout.decode()
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(client/'include'),'-I'+str(client/'ggml/include'),'-isystem',str(client/'ggml/src')]
 run('headers',[compiler,*flags,'-M','-MF',output/'headers.d',harness])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',[compiler,*flags,'-MD','-MF',output/'compile.d',harness,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml-cpu','-lggml-base','-o',output/'trace'])
 if provenance.dependencies(output/'headers.d')!=provenance.dependencies(output/'compile.d'):raise ValueError('compiler dependency drift')
 for p,h in provenance.runtime_inputs(run('ldd',['/usr/bin/ldd',output/'trace']),lib,inputs).items():provenance.merge(inputs,p,h)
 binary=provenance.sha(output/'trace');line=run('capture',[output/'trace',model,raw])
 if line!=LINE or binary!=provenance.sha(output/'trace'):raise ValueError('capture completion/binary')
 summary=validate(raw);(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');provenance.recheck(inputs)
 artifacts={str(p.relative_to(output)):provenance.sha(p) for p in output.rglob('*') if p.is_file()}
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024*1024:raise ValueError('trace bound')
 (output/'receipt.json').write_text(json.dumps({'status':'PASS','classification':'successful-full-model-cpu-boundary-reference','cpu_build':str(cpu),'model':{'path':str(model),'bytes':model.stat().st_size,'sha256':inputs[str(model)]},'inputs':inputs,'artifacts':artifacts,'commands':commands,'requests':514,'raw_files':168,'gpu_executed':False,'limits':['CPU diagnostic, not GPU admission','Single run with reset equality, not independent repeated-run determinism']},indent=2)+'\n');print('PASS full-model CPU reference,514 committed requests,168 boundary files')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();build(args.cpu_build,args.model,args.output)
