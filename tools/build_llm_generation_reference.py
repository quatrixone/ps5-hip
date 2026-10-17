"""Source-bound short CPU greedy generation; no CPU library rebuild or GPU claim."""
import argparse,json,math,os,re,struct,subprocess
from pathlib import Path
import build_llm_executed_graph as provenance
ROOT=Path(__file__).resolve().parents[1]
LINE='GENERATION_REFERENCE_PASS runs=2 max_new_tokens=4 original_cpu_decode=true GPU_execution=false\n'
SOURCES=[Path(__file__).resolve(),ROOT/'tests/llm_generation_reference.cpp',ROOT/'tests/llm_generation_reference_test.py',ROOT/'ports/ggml/generation.h',ROOT/'tests/llm_generation_test.cpp',ROOT/'tests/llm_generation_test.py',Path(provenance.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json']

def loads(text):
 def pairs(rows):
  out={}
  for k,v in rows:
   if k in out:raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 def constant(value):raise ValueError('nonfinite JSON constant')
 return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)

def validate(folder):
 folder=Path(folder)
 if folder.is_symlink() or not folder.is_dir():raise ValueError('raw directory')
 if any(not p.is_file() or p.is_symlink() for p in folder.iterdir()) or not all((folder/n).is_file() for n in ['generation.json','steps.jsonl']):raise ValueError('generation inventory')
 meta=loads((folder/'generation.json').read_text());steps=[loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
 fixed=dict(schema=1,prompt='Once upon a time',add_special=True,parse_special=False,max_new_tokens=4,vocab_size=32000)
 if not isinstance(meta,dict) or set(meta)!=set(fixed)|{'prompt_tokens','eog_tokens','runs'} or any(type(meta.get(k))is not type(v) or meta.get(k)!=v for k,v in fixed.items()):raise ValueError('generation metadata')
 prompt=meta['prompt_tokens'];eog=meta['eog_tokens']
 valid=lambda v:type(v)is int and 0<=v<32000
 if not isinstance(prompt,list) or not 1<=len(prompt)<=32 or not all(map(valid,prompt)):raise ValueError('prompt tokens')
 if not isinstance(eog,list) or not eog or not all(map(valid,eog)) or eog!=sorted(set(eog)):raise ValueError('EOS vocabulary')
 if not isinstance(meta['runs'],list) or len(meta['runs'])!=2:raise ValueError('reset runs')
 inventory={'generation.json','steps.jsonl'};expected_steps=[];raws={};runs=[]
 for run,row in enumerate(meta['runs']):
  if not isinstance(row,dict) or set(row)!={'run','processed','generated','stop','pieces_hex','text_hex'} or type(row['run'])is not int or row['run']!=run:raise ValueError('run metadata')
  processed=row['processed'];generated=row['generated']
  if not isinstance(processed,list) or not all(map(valid,processed)) or not isinstance(generated,list) or not all(map(valid,generated)) or not 1<=len(generated)<=4:raise ValueError('token bounds')
  if processed!=prompt+generated[:-1]:raise ValueError('generated feedback')
  if row['stop']!=('eos' if generated[-1] in eog else 'step_limit') or any(t in eog for t in generated[:-1]) or (row['stop']=='step_limit' and len(generated)!=4):raise ValueError('EOS stopping')
  if not isinstance(row['pieces_hex'],list) or len(row['pieces_hex'])!=len(generated):raise ValueError('decoded pieces')
  if any(not isinstance(s,str) or re.fullmatch(r'(?:[0-9a-f]{2})*',s)is None for s in [*row['pieces_hex'],row['text_hex']]):raise ValueError('decoded hex')
  try:
   pieces=[bytes.fromhex(s) for s in row['pieces_hex']];text=bytes.fromhex(row['text_hex'])
  except (ValueError,TypeError):raise ValueError('decoded hex')
  if any(len(p)>4096 for p in pieces) or b''.join(pieces)!=text:raise ValueError('decoded text')
  for pos,token in enumerate(processed):
   name=f'run-{run}-pos-{pos}-logits.f32le';inventory.add(name)
   if not (folder/name).is_file():raise ValueError('generation inventory')
   raw=(folder/name).read_bytes()
   if len(raw)!=128000:raise ValueError('logits size')
   values=struct.unpack('<32000f',raw)
   if not all(map(math.isfinite,values)):raise ValueError('nonfinite logits')
   if pos>=len(prompt)-1:
    # Independent Python argmax: first maximum wins, no shared C++ implementation.
    best=max(range(32000),key=values.__getitem__)
    if generated[pos-len(prompt)+1]!=best:raise ValueError('greedy argmax')
   raws[(run,pos)]=raw;expected_steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
  runs.append({k:v for k,v in row.items() if k!='run'})
 if any(not isinstance(s,dict) or any(type(v)is not int for v in s.values()) for s in steps) or steps!=expected_steps:raise ValueError('committed progression')
 if runs[0]!=runs[1] or any(raws[(0,p)]!=raws[(1,p)] for p in range(len(runs[0]['processed']))):raise ValueError('reset equality')
 if {p.name for p in folder.iterdir()}!=inventory or any(not p.is_file() or p.is_symlink() for p in folder.iterdir()):raise ValueError('generation inventory')
 return dict(requests=len(steps),raw_files=len(raws),raw_bytes=len(raws)*128000,prompt_tokens=prompt,generated=runs[0]['generated'],stop=runs[0]['stop'],text_hex=runs[0]['text_hex'])

def commands_valid(data,folder,cpu,model):
 rows=loads((folder/'commands.json').read_text())
 if not isinstance(rows,list) or any(not isinstance(x,dict) or set(x)!={'label','argv','exit_code'} or type(x['exit_code'])is not int or not isinstance(x['argv'],list) or not all(isinstance(v,str) for v in x['argv']) for x in rows):raise ValueError('command schema')
 if rows!=data['commands'] or [x.get('label') for x in rows]!=['headers','compile','ldd','capture'] or any(x.get('exit_code')!=0 for x in rows):raise ValueError('command completion')
 expected=command_plan(folder,cpu,model)
 if any(row['argv']!=expected[row['label']] for row in rows):raise ValueError('exact command identity')

def command_plan(folder,cpu,model):
 client=ROOT/'.deps/sources/llama-client';lib=cpu/'cmake/bin';harness=ROOT/'tests/llm_generation_reference.cpp'
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(client/'include'),'-I'+str(client/'ggml/include'),'-isystem',str(client/'ggml/src')]
 return {k:list(map(str,v)) for k,v in {
  'headers':['/usr/bin/clang++-18',*flags,'-M','-MF',folder/'headers.d',harness],
  'compile':['/usr/bin/clang++-18',*flags,'-MD','-MF',folder/'compile.d',harness,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml-cpu','-lggml-base','-o',folder/'trace'],
  'ldd':['/usr/bin/ldd',folder/'trace'],'capture':[folder/'trace',model,folder/'raw']}.items()}

def artifact_inventory(folder,receipt):
 if any(p.is_symlink() for p in folder.rglob('*')):raise ValueError('generation artifact symlink')
 return {str(p.relative_to(folder)):provenance.sha(p) for p in folder.rglob('*') if p.is_file() and p!=receipt}

def final_artifact_check(folder,receipt,artifacts,digest):
 if artifact_inventory(folder,receipt)!=artifacts or provenance.sha(receipt)!=digest:raise ValueError('generation artifacts or receipt changed during authentication')

def authenticate_generation_reference(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=loads(receipt.read_text())
 if data.get('status')!='PASS' or data.get('classification')!='successful-greedy-generation-cpu-reference' or data.get('gpu_executed') is not False or data.get('max_new_tokens')!=4:raise ValueError('generation receipt')
 inputs=data['inputs']
 if any(inputs.get(str(p))!=provenance.sha(p) for p in SOURCES):raise ValueError('generation source binding')
 artifacts=artifact_inventory(folder,receipt)
 if artifacts!=data['artifacts']:raise ValueError('generation artifacts')
 model=data['model'];path=Path(model['path']);cpu=Path(data['cpu_build'])
 if not path.is_absolute() or path.stat().st_size!=model['bytes'] or inputs.get(str(path))!=model['sha256']:raise ValueError('model identity')
 for p,h in provenance.cpu_inputs(cpu,path).items():
  if inputs.get(p)!=h:raise ValueError('CPU closure')
 headers=provenance.dependencies(folder/'headers.d')
 if headers!=provenance.dependencies(folder/'compile.d') or any(inputs.get(p)!=provenance.sha(p) for p in headers):raise ValueError('compiler closure')
 for p,h in provenance.runtime_inputs((folder/'ldd.stdout').read_text(),cpu/'cmake/bin',inputs).items():
  if inputs.get(p)!=h:raise ValueError('runtime closure')
 commands_valid(data,folder,cpu,path);summary=validate(folder/'raw')
 if summary!=loads((folder/'summary.json').read_text()) or summary['requests']!=data['requests'] or summary['raw_files']!=data['raw_files'] or (folder/'capture.stdout').read_text()!=LINE:raise ValueError('summary/completion')
 provenance.recheck(inputs)
 final_artifact_check(folder,receipt,artifacts,digest)
 closure=dict(inputs)
 for p,h in artifacts.items():provenance.merge(closure,folder/p,h)
 provenance.merge(closure,receipt,digest)
 return {**data,'receipt':str(receipt),'receipt_sha256':digest,'reference_data':str(folder/'raw')},closure

def build(cpu,model,output):
 cpu,model,output=map(lambda p:Path(p).resolve(),(cpu,model,output))
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results directory required')
 inputs=provenance.cpu_inputs(cpu,model);compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client';harness=ROOT/'tests/llm_generation_reference.cpp'
 for p in [*SOURCES,compiler,Path('/usr/bin/ldd')]:provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True);raw=output/'raw';raw.mkdir();commands=[]
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 def run(label,args):
  args=list(map(str,args));result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,timeout=1800)
  (output/(label+'.stdout')).write_bytes(result.stdout);(output/(label+'.stderr')).write_bytes(result.stderr);commands.append(dict(label=label,argv=args,exit_code=result.returncode));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
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
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>32*1024*1024:raise ValueError('reference bound')
 data=dict(status='PASS',classification='successful-greedy-generation-cpu-reference',cpu_build=str(cpu),model=dict(path=str(model),bytes=model.stat().st_size,sha256=inputs[str(model)]),inputs=inputs,artifacts=artifacts,commands=commands,requests=summary['requests'],raw_files=summary['raw_files'],max_new_tokens=4,gpu_executed=False,limits=['CPU reference only; no GPU execution','Reset/repeat in one process, not independent-process determinism','Text pieces are source-bound vocabulary API output; Python independently validates greedy IDs and feedback'])
 commands_valid(data,output,cpu,model);(output/'receipt.json').write_text(json.dumps(data,indent=2)+'\n');print('PASS CPU greedy generation reference')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();build(args.cpu_build,args.model,args.output)
