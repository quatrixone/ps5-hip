"""Build/authenticate a CPU-only actual-model layer reference using existing CPU libraries."""
from pathlib import Path
import argparse,json,os,re,subprocess
import build_llm_executed_graph as provenance
import source_llm_layer_raw as contract
ROOT=Path(__file__).resolve().parents[1]
HARNESS=ROOT/'tests/llm_actual_layer_reference.cpp'
CLASSIFICATION='source-only-actual-model-layer-cpu-reference'
LINE='ACTUAL_LAYER_REFERENCE_PASS requests=514 original_cpu_validation=4 nodes=37 outputs=21 raw_files=151 cache_reset=true GPU_execution=false\n'

def sources():
 return [Path(__file__).resolve(),HARNESS,ROOT/'ports/ggml/model_layer.h',ROOT/'tests/llm_layer_reference_test.py',Path(provenance.__file__).resolve(),Path(contract.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json']

def inventory(folder):
 result={}
 for p in Path(folder).rglob('*'):
  if p.is_symlink():raise ValueError('reference symlink')
  if p.is_file():result[p.relative_to(folder).as_posix()]=provenance.sha(p)
 return result

def validate_reference(folder,stdout):
 if stdout!=LINE:raise ValueError('layer oracle completion differs')
 data=contract.read_raw(folder)
 for pos in (0,1):
  for i in range(21):
   if data[f'run-0-pos-{pos}-out-{i:02d}.f32le']!=data[f'run-1-pos-{pos}-out-{i:02d}.f32le']:raise ValueError('reference reset differs')
 return {'files':len(data),'bytes':sum(map(len,data.values())),'original_cpu_validation':4,'requests':514,'nodes':37,'reset_bitwise':True}

def authenticate_layer_reference(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=json.loads(receipt.read_text())
 if data.get('schema')!=1 or data.get('status')!='PASS' or data.get('classification')!=CLASSIFICATION or data.get('native_executed') is not False or data.get('gpu_executed') is not False or data.get('repeat_bitwise') is not True:raise ValueError('layer reference contract')
 inputs=data.get('inputs');artifacts=data.get('artifacts')
 if not isinstance(inputs,dict) or not inputs or not isinstance(artifacts,dict) or not artifacts or any(not Path(p).is_absolute() for p in inputs):raise ValueError('layer reference identities')
 for p in sources():
  if inputs.get(str(p))!=provenance.sha(p):raise ValueError('layer reference source stale/missing')
 actual=inventory(folder);actual.pop('receipt.json',None)
 if actual!=artifacts:raise ValueError('layer reference artifact inventory differs')
 model=data.get('model')
 if not isinstance(model,dict) or set(model)!={'path','bytes','sha256'} or not Path(model['path']).is_absolute() or Path(model['path']).stat().st_size!=model['bytes'] or provenance.sha(model['path'])!=model['sha256']:raise ValueError('layer reference model differs')
 for p,h in provenance.cpu_inputs(Path(data['cpu_build']),Path(model['path'])).items():
  if inputs.get(p)!=h:raise ValueError('layer CPU input closure differs')
 headers=provenance.dependencies(folder/'headers.d')
 if headers!=provenance.dependencies(folder/'compile.d'):raise ValueError('layer compiler dependency inventory differs')
 for p in headers:
  if inputs.get(p)!=provenance.sha(p):raise ValueError('layer compiler header differs')
 for p,h in provenance.runtime_inputs((folder/'ldd.stdout').read_text(),Path(data['cpu_build'])/'cmake/bin',inputs).items():
  if inputs.get(p)!=h:raise ValueError('layer CPU runtime differs')
 provenance.recheck(inputs)
 if inventory(folder/'run-1')!=inventory(folder/'run-2'):raise ValueError('layer reference repeats differ')
 metrics=validate_reference(folder/'run-1',(folder/'run-1.stdout').read_text())
 if metrics!=data.get('metrics') or validate_reference(folder/'run-2',(folder/'run-2.stdout').read_text())!=metrics:raise ValueError('layer reference metrics differ')
 if provenance.sha(receipt)!=digest:raise ValueError('layer reference receipt changed')
 closure=dict(inputs)
 for p,h in artifacts.items():provenance.merge(closure,folder/p,h)
 provenance.merge(closure,receipt,digest)
 return {**data,'reference':str(folder/'run-1'),'receipt':str(receipt),'receipt_sha256':digest},closure

def build(cpu,model,output):
 cpu=Path(cpu).resolve();model=Path(model).resolve();output=Path(output).resolve()
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results output required')
 compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 inputs=provenance.cpu_inputs(cpu,model)
 for p in [*sources(),compiler,Path('/usr/bin/ldd')]:provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True);commands=[]
 forbidden={'LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS'}
 env={k:v for k,v in os.environ.items() if k not in forbidden and not k.startswith(('GGML_','OMP_'))}
 def run(label,args):
  r=subprocess.run(list(map(str,args)),cwd=ROOT,env=env,capture_output=True,timeout=900)
  (output/(label+'.stdout')).write_bytes(r.stdout);(output/(label+'.stderr')).write_bytes(r.stderr);commands.append({'label':label,'argv':list(map(str,args)),'exit_code':r.returncode})
  if r.returncode or re.search(rb'AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer|runtime error:',r.stderr):raise ValueError(label+' failed')
  return r.stdout.decode()
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'ports/ggml'),'-I'+str(client/'include'),'-I'+str(client/'ggml/include')]
 run('headers',[compiler,*flags,'-M','-MF',output/'headers.d',HARNESS])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',[compiler,*flags,'-MD','-MF',output/'compile.d',HARNESS,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml-cpu','-lggml-base','-o',output/'oracle'])
 if provenance.dependencies(output/'headers.d')!=provenance.dependencies(output/'compile.d'):raise ValueError('layer compiler inputs changed')
 for p,h in provenance.runtime_inputs(run('ldd',['/usr/bin/ldd',output/'oracle']),lib,inputs).items():provenance.merge(inputs,p,h)
 binary_sha=provenance.sha(output/'oracle')
 for number in (1,2):
  folder=output/f'run-{number}';folder.mkdir();run(f'run-{number}',[output/'oracle',model,folder])
 if binary_sha!=provenance.sha(output/'oracle'):raise ValueError('layer oracle binary changed')
 if inventory(output/'run-1')!=inventory(output/'run-2'):raise ValueError('layer oracle repeats differ')
 metrics=validate_reference(output/'run-1',(output/'run-1.stdout').read_text())
 if validate_reference(output/'run-2',(output/'run-2.stdout').read_text())!=metrics:raise ValueError('layer oracle metrics differ')
 provenance.recheck(inputs);artifacts=inventory(output)
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>=12*1024*1024:raise ValueError('layer reference size bound')
 data={'schema':1,'status':'PASS','classification':CLASSIFICATION,'cpu_build':str(cpu),'model':{'path':str(model),'bytes':model.stat().st_size,'sha256':inputs[str(model)]},'repeat_bitwise':True,'metrics':metrics,'inputs':inputs,'artifacts':artifacts,'commands':commands,'native_executed':False,'gpu_executed':False,'limits':['CPU derived-layer sequence, validated against four successful original CPU requests first','CPU Q4 uses Q8_0; CPU F16 matmul uses upstream CPU rounding; GPU numerical acceptance separate','No full-model inference or original full-model cache commits asserted for derived514-step sequence']}
 (output/'receipt.json').write_text(json.dumps(data,indent=2)+'\n');authenticate_layer_reference(output)
 print('PASS authenticated actual-model CPU layer reference, 512 steps and reset2, two bitwise repeats')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();build(args.cpu_build,args.model,args.output)
