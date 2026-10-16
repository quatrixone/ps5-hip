"""Authenticated CPU-only actual-model prefix reference; no GPU claim."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys,types
BASE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_llm_executed_graph as provenance
import verify_llm_prefix_readbacks as numerical
NUMERICAL=ROOT/'tools/verify_llm_prefix_readbacks.py'
HARNESS=ROOT/'tests/llm_actual_prefix_reference.cpp'
SELECTOR=ROOT/'ports/ggml/model_prefix.h'
TOKENS=(0,1,2,31999)
SIZES={**numerical.FILES,'norm.f32le':8192,'q.weights.q4_0':2359296,'k.weights.q4_0':294912,'v.weights.q4_0':294912,**{f'token-{t}-embedding.q4_0':1152 for t in TOKENS}}
CLASSIFICATION='source-only-actual-model-prefix-cpu-reference'

def sources():
 # Follow only actual dependencies; unrelated consumer imports must not alter admission.
 pending=[provenance,numerical];seen=set();imported=set()
 while pending:
  module=pending.pop()
  if id(module) in seen:continue
  seen.add(id(module));name=getattr(module,'__file__',None)
  if not name:continue
  path=Path(name).resolve()
  if not path.is_relative_to(ROOT/'tools'):continue
  imported.add(path)
  for value in vars(module).values():
   if isinstance(value,types.ModuleType):pending.append(value)
   elif isinstance(value,(types.FunctionType,type)):
    owner=sys.modules.get(getattr(value,'__module__',''))
    if owner is not None:pending.append(owner)
 return sorted({ROOT/'tests/llm_prefix_reference_test.py',*imported,Path('/usr/bin/clang++-18'),Path('/usr/bin/ldd'),Path(__file__).resolve(),HARNESS,SELECTOR,NUMERICAL,Path(provenance.__file__).resolve(),Path(numerical.norm.__file__).resolve(),Path(numerical.weighted.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json'})


def inventory(folder):
 files={}
 for p in folder.rglob('*'):
  if p.is_symlink():raise ValueError('symlink in reference output')
  if p.is_file():files[str(p.relative_to(folder))]=provenance.sha(p)
 return files

def validate_reference(folder,stdout):
 expected=''.join(f'ACTUAL_PREFIX_PASS token={t} nodes=6 source_bytes=39821316 Q=2048 K=256 V=256 GPU_execution=false\n' for t in TOKENS)
 if stdout!=expected:raise ValueError('actual four-token/six-node capture differs')
 if set(inventory(folder))!=set(SIZES)|{'source-tensors.json'}:raise ValueError('reference inventory differs')
 for name,size in SIZES.items():
  if (folder/name).stat().st_size!=size:raise ValueError('reference byte size differs')
 metadata=json.loads((folder/'source-tensors.json').read_text())
 tensors=[]
 for name,dtype,ne,nb,size in [
  ('token_embd.weight','q4_0',[2048,32000,1,1],[18,1152,36864000,36864000],36864000),
  ('inp_tokens','i32',[1,1,1,1],[4,4,4,4],4),
  ('blk.0.attn_norm.weight','f32',[2048,1,1,1],[4,8192,8192,8192],8192),
  ('blk.0.attn_q.weight','q4_0',[2048,2048,1,1],[18,1152,2359296,2359296],2359296),
  ('blk.0.attn_k.weight','q4_0',[2048,256,1,1],[18,1152,294912,294912],294912),
  ('blk.0.attn_v.weight','q4_0',[2048,256,1,1],[18,1152,294912,294912],294912)]:
  tensors.append(dict(name=name,dtype=dtype,ne=ne,nb=nb,bytes=size))
 if metadata!={'classification':'actual-selected-model-leaves','tensors':tensors,'embedding_row_bytes':1152,'embedding_row_offset_rule':'token_id * 1152 within token_embd.weight'}:raise ValueError('selected source metadata differs')
 weights=(folder/'norm.f32le').read_bytes();numerical.values(weights,2048);metrics={}
 for token in TOKENS:
  raw={name:(folder/f'token-{token}-{name}.f32le').read_bytes() for name in numerical.WIDTHS}
  for name,count in numerical.WIDTHS.items():numerical.values(raw[name],count)
  embedding=numerical.embedding_reference((folder/f'token-{token}-embedding.q4_0').read_bytes())
  if raw['embedding']!=embedding:raise ValueError('independent embedding decode differs')
  norm=numerical.norm.compare(numerical.values(raw['normalized'],2048),numerical.norm.fp64_reference(numerical.values(embedding,2048)))
  target,budget=numerical.weighted.weighted_reference(embedding,weights)
  weighted=numerical.weighted.compare_weighted(raw['weighted'],target,budget)
  if max(norm['maximum_budget_fraction'],weighted['maximum_budget_fraction'])>1/32:raise ValueError('CPU baseline consumes more than 1/32 of budget')
  metrics[str(token)]={'normalized':norm,'weighted':weighted}
 return metrics

def authenticate_prefix_reference(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=json.loads(receipt.read_text())
 if data.get('schema')!=1 or data.get('status')!='PASS' or data.get('classification')!=CLASSIFICATION or data.get('native_executed') is not False or data.get('gpu_executed') is not False or data.get('tokens')!=list(TOKENS) or data.get('nodes')!=6 or data.get('repeat_bitwise') is not True:raise ValueError('reference receipt contract differs')
 inputs=data.get('inputs');artifacts=data.get('artifacts')
 if not isinstance(inputs,dict) or not inputs or not isinstance(artifacts,dict) or not artifacts:raise ValueError('missing identities')
 for p in inputs:
  if not Path(p).is_absolute():raise ValueError('nonabsolute input')
 for p in sources():
  if inputs.get(str(p))!=provenance.sha(p):raise ValueError('required source missing or stale')
 actual=inventory(folder);actual.pop('receipt.json',None)
 if actual!=artifacts:raise ValueError('reference artifact identity or inventory differs')
 model=data.get('model')
 if not isinstance(model,dict) or set(model)!={'path','bytes','sha256'} or not Path(model['path']).is_absolute() or Path(model['path']).stat().st_size!=model['bytes'] or provenance.sha(model['path'])!=model['sha256']:raise ValueError('model identity differs')
 authenticated=provenance.cpu_inputs(Path(data['cpu_build']),Path(data['model']['path']))
 for p,h in authenticated.items():
  if inputs.get(p)!=h:raise ValueError('CPU/model closure differs')
 for p in provenance.dependencies(folder/'compile.d') | provenance.dependencies(folder/'headers.d'):
  if inputs.get(p)!=provenance.sha(p):raise ValueError('reference compiler header missing or changed')
 for p,h in provenance.runtime_inputs((folder/'ldd.stdout').read_text(),Path(data['cpu_build'])/'cmake/bin',inputs).items():
  if inputs.get(p)!=h:raise ValueError('reference runtime identity missing')
 provenance.recheck(inputs)
 one,two=folder/'run-1',folder/'run-2'
 if inventory(one)!=inventory(two):raise ValueError('reference repetitions differ')
 metrics=validate_reference(one,(folder/'run-1.stdout').read_text())
 if validate_reference(two,(folder/'run-2.stdout').read_text())!=metrics or data.get('metrics')!=metrics:raise ValueError('reference metrics differ')
 if provenance.sha(receipt)!=digest:raise ValueError('reference receipt changed')
 result={**data,'receipt':str(receipt),'receipt_sha256':digest,'reference':str(one)}
 closure=dict(inputs)
 for name,h in artifacts.items():provenance.merge(closure,folder/name,h)
 provenance.merge(closure,receipt,digest)
 return result,closure

def build(cpu,model,output):
 cpu=Path(cpu).resolve();model=Path(model).resolve();output=Path(output).resolve()
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results directory required')
 compiler=Path('/usr/bin/clang++-18');ldd=Path('/usr/bin/ldd');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 inputs=provenance.cpu_inputs(cpu,model)
 for p in [*sources(),compiler,ldd]:provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True)
 env={k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS')}
 commands=[]
 def run(label,argv):
  r=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,capture_output=True,timeout=900)
  (output/(label+'.stdout')).write_bytes(r.stdout);(output/(label+'.stderr')).write_bytes(r.stderr)
  commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':r.returncode})
  if r.returncode or re.search(rb'AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer|runtime error:',r.stderr):raise ValueError(label+' failed')
  return r.stdout.decode()
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'ports/ggml'),'-I'+str(client/'include'),'-I'+str(client/'ggml/include'),'-I'+str(client/'ggml/src')]
 run('headers',[compiler,*flags,'-M','-MF',output/'headers.d',HARNESS])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',[compiler,*flags,'-MD','-MF',output/'compile.d',HARNESS,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml-cpu','-lggml-base','-o',output/'oracle'])
 if provenance.dependencies(output/'compile.d')!=provenance.dependencies(output/'headers.d'):raise ValueError('compiler dependencies changed')
 for p,h in provenance.runtime_inputs(run('ldd',[ldd,output/'oracle']),lib,inputs).items():provenance.merge(inputs,p,h)
 binary_sha=provenance.sha(output/'oracle')
 for number in (1,2):
  directory=output/f'run-{number}';directory.mkdir();run(f'run-{number}',[output/'oracle',model,directory])
 if provenance.sha(output/'oracle')!=binary_sha:raise ValueError('executed oracle changed')
 if inventory(output/'run-1')!=inventory(output/'run-2'):raise ValueError('CPU reference not bitwise reproducible')
 metrics=validate_reference(output/'run-1',(output/'run-1.stdout').read_text())
 if validate_reference(output/'run-2',(output/'run-2.stdout').read_text())!=metrics:raise ValueError('second run metrics differ')
 provenance.recheck(inputs)
 artifacts=inventory(output)
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>=10*1024*1024:raise ValueError('reference exceeds 10MiB')
 data={'schema':1,'status':'PASS','classification':CLASSIFICATION,'cpu_build':str(cpu),'model':{'path':str(model),'bytes':model.stat().st_size,'sha256':inputs[str(model)]},'tokens':list(TOKENS),'nodes':6,'repeat_bitwise':True,'metrics':metrics,'inputs':inputs,'artifacts':artifacts,'commands':commands,'native_executed':False,'gpu_executed':False,'limits':['CPU oracle only; GPU transport and numerical qualification are separate','CPU projection outputs use upstream Q8_0 dot products; no Q8_1 equivalence asserted']}
 (output/'receipt.json').write_text(json.dumps(data,indent=2)+'\n')
 authenticate_prefix_reference(output)
 print('PASS authenticated repeatable four-token CPU prefix reference')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();build(a.cpu_build,a.model,a.output)
