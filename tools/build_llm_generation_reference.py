"""Source-bound short CPU greedy generation; no CPU library rebuild or GPU claim."""
import argparse,json,math,os,re,struct,subprocess
from pathlib import Path
import build_llm_executed_graph as provenance
ROOT=Path(__file__).resolve().parents[1]
LINE='GENERATION_REFERENCE_PASS runs=2 max_new_tokens=4 original_cpu_decode=true GPU_execution=false\n'
STAGES=[*(f'layer-{i}' for i in range(22)),'norm','logits']
CASES=ROOT/'ports/ggml/generation_cases.def'
CORPUS_LINE='GENERATION_REFERENCE_PASS profile=corpus cases=5 runs=2 max_new_tokens=16 original_cpu_decode=true GPU_execution=false\n'
SOURCES=[CASES,Path(__file__).resolve(),ROOT/'tests/llm_generation_reference.cpp',ROOT/'tests/llm_generation_reference_test.py',ROOT/'ports/ggml/generation.h',ROOT/'tests/llm_generation_test.cpp',ROOT/'tests/llm_generation_test.py',Path(provenance.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json']

def loads(text):
 def pairs(rows):
  out={}
  for k,v in rows:
   if k in out:raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 def constant(value):raise ValueError('nonfinite JSON constant')
 return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)

def corpus_cases():
 rows=[]
 for line in CASES.read_text(encoding='utf-8').splitlines():
  if line.startswith('//') or not line:continue
  match=re.fullmatch(r'PS5HIP_GENERATION_CASE\(("[a-z]+"), ("(?:[^"\\]|\\.)*")\)',line)
  if not match:raise ValueError('fixed corpus definition')
  rows.append(tuple(loads(v) for v in match.groups()))
 if len(rows)!=5 or len({r[0] for r in rows})!=5:raise ValueError('fixed corpus cases')
 return rows

def profile_policy(profile):
 if profile not in ('short','corpus'):raise ValueError('generation profile')
 return (4,LINE,32*1024*1024) if profile=='short' else (16,CORPUS_LINE,128*1024*1024)

def validate(folder,profile='short'):
 profile_policy(profile);folder=Path(folder)
 if profile=='short':return validate_case(folder,'Once upon a time',4)
 if folder.is_symlink() or not folder.is_dir():raise ValueError('corpus directory')
 cases=corpus_cases();expected=dict(schema=1,profile='corpus',max_new_tokens=16,cases=[k for k,_ in cases])
 if {p.name for p in folder.iterdir()}!={'corpus.json',*[k for k,_ in cases]} or (folder/'corpus.json').is_symlink():raise ValueError('corpus inventory')
 meta=loads((folder/'corpus.json').read_text())
 if meta!=expected or type(meta.get('schema'))is not int or type(meta.get('max_new_tokens'))is not int:raise ValueError('corpus metadata')
 summaries={key:validate_case(folder/key,prompt,16) for key,prompt in cases}
 return dict(profile='corpus',cases=summaries,**{key:sum(v[key] for v in summaries.values()) for key in ('requests','raw_files','raw_bytes')})

def validate_case(folder,prompt_text,budget):
 folder=Path(folder)
 if folder.is_symlink() or not folder.is_dir():raise ValueError('raw directory')
 if any(not p.is_file() or p.is_symlink() for p in folder.iterdir()) or not all((folder/n).is_file() for n in ['generation.json','steps.jsonl']):raise ValueError('generation inventory')
 meta=loads((folder/'generation.json').read_text());steps=[loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
 fixed=dict(schema=1,prompt=prompt_text,add_special=True,parse_special=False,max_new_tokens=budget,vocab_size=32000)
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
  if not isinstance(processed,list) or not all(map(valid,processed)) or not isinstance(generated,list) or not all(map(valid,generated)) or not 1<=len(generated)<=budget:raise ValueError('token bounds')
  if budget==16 and (len(generated)!=16 or row['stop']!='step_limit' or any(t in eog for t in generated)):raise ValueError('corpus requires sixteen tokens without EOS')
  if processed!=prompt+generated[:-1]:raise ValueError('generated feedback')
  if row['stop']!=('eos' if generated[-1] in eog else 'step_limit') or any(t in eog for t in generated[:-1]) or (row['stop']=='step_limit' and len(generated)!=budget):raise ValueError('EOS stopping')
  if not isinstance(row['pieces_hex'],list) or len(row['pieces_hex'])!=len(generated):raise ValueError('decoded pieces')
  if any(not isinstance(s,str) or re.fullmatch(r'(?:[0-9a-f]{2})*',s)is None for s in [*row['pieces_hex'],row['text_hex']]):raise ValueError('decoded hex')
  try:
   pieces=[bytes.fromhex(s) for s in row['pieces_hex']];text=bytes.fromhex(row['text_hex'])
  except (ValueError,TypeError):raise ValueError('decoded hex')
  if any(len(p)>4096 for p in pieces) or b''.join(pieces)!=text:raise ValueError('decoded text')
  for pos,token in enumerate(processed):
   for stage in STAGES:
    name=f'run-{run}-pos-{pos}-{stage}.f32le';inventory.add(name)
    path=folder/name
    if not path.is_file():raise ValueError('generation inventory')
    raw=path.read_bytes();count=32000 if stage=='logits' else 2048
    if len(raw)!=count*4:raise ValueError('boundary size')
    if not all(map(math.isfinite,struct.unpack('<'+str(count)+'f',raw))):raise ValueError('nonfinite boundary')
    raws[(run,pos,stage)]=raw
   name=f'run-{run}-pos-{pos}-logits.f32le'
   if not (folder/name).is_file():raise ValueError('generation inventory')
   raw=(folder/name).read_bytes()
   if len(raw)!=128000:raise ValueError('logits size')
   values=struct.unpack('<32000f',raw)
   if not all(map(math.isfinite,values)):raise ValueError('nonfinite logits')
   if pos>=len(prompt)-1:
    # Independent Python argmax: first maximum wins, no shared C++ implementation.
    best=max(range(32000),key=values.__getitem__)
    if generated[pos-len(prompt)+1]!=best:raise ValueError('greedy argmax')
   expected_steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
  runs.append({k:v for k,v in row.items() if k!='run'})
 if any(not isinstance(s,dict) or any(type(v)is not int for v in s.values()) for s in steps) or steps!=expected_steps:raise ValueError('committed progression')
 if runs[0]!=runs[1] or any(raws[(0,p,stage)]!=raws[(1,p,stage)] for p in range(len(runs[0]['processed'])) for stage in STAGES):raise ValueError('reset equality')
 if {p.name for p in folder.iterdir()}!=inventory or any(not p.is_file() or p.is_symlink() for p in folder.iterdir()):raise ValueError('generation inventory')
 return dict(requests=len(steps),raw_files=len(raws),raw_bytes=sum(map(len,raws.values())),prompt_tokens=prompt,generated=runs[0]['generated'],stop=runs[0]['stop'],text_hex=runs[0]['text_hex'])

def commands_valid(data,folder,cpu,model):
 rows=loads((folder/'commands.json').read_text())
 if not isinstance(rows,list) or any(not isinstance(x,dict) or set(x)!={'label','argv','exit_code'} or type(x['exit_code'])is not int or not isinstance(x['argv'],list) or not all(isinstance(v,str) for v in x['argv']) for x in rows):raise ValueError('command schema')
 if rows!=data['commands'] or [x.get('label') for x in rows]!=['headers','compile','ldd','capture'] or any(x.get('exit_code')!=0 for x in rows):raise ValueError('command completion')
 producer=Path(data['producer_output'])
 if not producer.is_absolute() or not producer.is_relative_to(ROOT/'results') or producer!=producer.resolve():raise ValueError('producer output identity')
 expected=command_plan(producer,cpu,model,data.get('profile','short'),Path(data['original_source']) if 'original_source' in data else None)
 if any(row['argv']!=expected[row['label']] for row in rows):raise ValueError('exact command identity')

def command_plan(folder,cpu,model,profile='short',original_source=None):
 profile_policy(profile)
 client=ROOT/'.deps/sources/llama-client';lib=cpu/'cmake/bin';harness=ROOT/'tests/llm_generation_reference.cpp'
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(client/'include'),'-I'+str(client/'ggml/include'),'-isystem',str(client/'ggml/src')]
 if original_source is not None:
  if profile!='short':raise ValueError('original source requires short profile')
  flags += ['-DPS5HIP_ORIGINAL_MODEL=1','-I'+str(client/'vendor')]
 return {k:list(map(str,v)) for k,v in {
  'headers':['/usr/bin/clang++-18',*flags,'-M','-MF',folder/'headers.d',harness],
  'compile':['/usr/bin/clang++-18',*flags,'-MD','-MF',folder/'compile.d',harness,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml-cpu','-lggml-base','-o',folder/'trace'],
  'ldd':['/usr/bin/ldd',folder/'trace'],'capture':[folder/'trace',original_source if original_source is not None else model,folder/'raw',*(['--original-source'] if original_source is not None else ['corpus'] if profile=='corpus' else [])]}.items()}

def artifact_inventory(folder,receipt):
 if any(p.is_symlink() for p in folder.rglob('*')):raise ValueError('generation artifact symlink')
 return {str(p.relative_to(folder)):provenance.sha(p) for p in folder.rglob('*') if p.is_file() and p!=receipt}

def final_artifact_check(folder,receipt,artifacts,digest):
 if artifact_inventory(folder,receipt)!=artifacts or provenance.sha(receipt)!=digest:raise ValueError('generation artifacts or receipt changed during authentication')

def authenticate_generation_reference(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=loads(receipt.read_text())
 if any(k in data for k in ('original_source','reference','complete201')):raise ValueError('original-source comparison is not an independent GGUF oracle')
 profile=data.get('profile','short');budget,line,limit=profile_policy(profile)
 if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>limit:raise ValueError('reference bound')
 if data.get('status')!='PASS' or data.get('classification')!='successful-greedy-generation-cpu-reference' or data.get('gpu_executed') is not False or data.get('max_new_tokens')!=budget:raise ValueError('generation receipt')
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
 commands_valid(data,folder,cpu,path);summary=validate(folder/'raw',profile)
 if summary!=loads((folder/'summary.json').read_text()) or summary['requests']!=data['requests'] or summary['raw_files']!=data['raw_files'] or (folder/'capture.stdout').read_text()!=line:raise ValueError('summary/completion')
 provenance.recheck(inputs)
 final_artifact_check(folder,receipt,artifacts,digest)
 closure=dict(inputs)
 for p,h in artifacts.items():provenance.merge(closure,folder/p,h)
 provenance.merge(closure,receipt,digest)
 return {**data,'receipt':str(receipt),'receipt_sha256':digest,'reference_data':str(folder/'raw')},closure

def build(cpu,model,output,profile='short',original_source=None,reference=None):
 budget,completion,limit=profile_policy(profile)
 cpu,model,output=map(lambda p:Path(p).resolve(),(cpu,model,output))
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results directory required')
 inputs=provenance.cpu_inputs(cpu,model);compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client';harness=ROOT/'tests/llm_generation_reference.cpp'
 for p in [*SOURCES,compiler,Path('/usr/bin/ldd')]:provenance.merge(inputs,p,provenance.sha(p))
 if (original_source is None)!=(reference is None):raise ValueError('original source and reference must be supplied together')
 if original_source is not None:
  if profile!='short':raise ValueError('original source requires short profile')
  original_source=Path(original_source).resolve();reference=Path(reference).resolve()
  from prepare_llm_native_loader import render_user
  for name in ('tools/prepare_llm_native_loader.py','tools/llm_user_loader.cmake'):
   if loads((cpu/'receipt.json').read_text())['identity']['inputs'].get(name)!=provenance.sha(ROOT/name):raise ValueError('patched CPU loader prerequisite')
  generated=cpu/'llama-model-loader-user.cpp'
  if generated.is_symlink() or generated.read_text()!=render_user((client/'src/llama-model-loader.cpp').read_text()):raise ValueError('patched CPU loader source')
  oracle,closure=authenticate_generation_reference(reference)
  if oracle.get('profile','short')!='short' or oracle['cpu_build']!=str(cpu) or oracle['model']['path']!=str(model):raise ValueError('short oracle CPU/model differs')
  for p,h in closure.items():provenance.merge(inputs,p,h)
  lock=ROOT/'models/tinyllama-1.1b-chat-v1.0.json';spec=loads(lock.read_text());provenance.merge(inputs,lock,provenance.sha(lock))
  originals=dict(spec['metadata']);originals['model.safetensors']=dict(bytes=spec['weights']['file_bytes'],sha256=spec['weights']['expected_file_sha256'])
  for name,item in originals.items():
   p=original_source/name
   if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes'] or provenance.sha(p)!=item['sha256']:raise ValueError('original source identity')
   provenance.merge(inputs,p,item['sha256'])
  for name in ('tools/prepare_llm_native_loader.py','tools/llm_user_loader.cmake','tools/build_llm_generation_numeric_test.py','tests/llm_original_generation_test.py'):provenance.merge(inputs,ROOT/name,provenance.sha(ROOT/name))
  provenance.merge(inputs,generated,provenance.sha(generated))

 output.mkdir(parents=True);raw=output/'raw';raw.mkdir();commands=[]
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 def run(label,args):
  args=list(map(str,args));result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,timeout=1800)
  (output/(label+'.stdout')).write_bytes(result.stdout);(output/(label+'.stderr')).write_bytes(result.stderr);commands.append(dict(label=label,argv=args,exit_code=result.returncode));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if result.returncode:raise ValueError(label+' failed')
  return result.stdout.decode()
 plan=command_plan(output,cpu,model,profile,original_source)
 run('headers',plan['headers'])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',plan['compile'])
 if provenance.dependencies(output/'headers.d')!=provenance.dependencies(output/'compile.d'):raise ValueError('compiler dependency drift')
 for p,h in provenance.runtime_inputs(run('ldd',['/usr/bin/ldd',output/'trace']),lib,inputs).items():provenance.merge(inputs,p,h)
 binary=provenance.sha(output/'trace');line=run('capture',plan['capture'])
 if line!=completion or binary!=provenance.sha(output/'trace'):raise ValueError('capture completion/binary')
 summary=validate(raw,profile);(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');provenance.recheck(inputs)
 if original_source is not None:
  from build_llm_generation_numeric_test import compare_generation
  rows,files,requests=compare_generation(raw,reference/'raw',host_backend=False)
  if requests!=16 or len(files)!=384 or not all(r['pass_budget'] for r in rows):raise ValueError('original-source numerical comparison')
  provenance.recheck(inputs)
  (output/'comparison.json').write_text(json.dumps(dict(rows=rows,requests=requests,raw_files=len(files)),indent=2)+'\n')
 artifacts={str(p.relative_to(output)):provenance.sha(p) for p in output.rglob('*') if p.is_file()}
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>limit:raise ValueError('reference bound')
 data=dict(producer_output=str(output),status='PASS',classification='successful-greedy-generation-cpu-reference',cpu_build=str(cpu),model=dict(path=str(model),bytes=model.stat().st_size,sha256=inputs[str(model)]),inputs=inputs,artifacts=artifacts,commands=commands,requests=summary['requests'],raw_files=summary['raw_files'],max_new_tokens=budget,gpu_executed=False,limits=['CPU reference only; no GPU execution','Reset/repeat in one process, not independent-process determinism','Text pieces are source-bound vocabulary API output; Python independently validates greedy IDs and feedback'])
 if original_source is not None:
  data.update(classification='original-format-host-short-generation-compared',original_source=str(original_source),reference=str(reference),complete201=True,native_executed=False)
 if profile=='corpus':data['profile']=profile
 commands_valid(data,output,cpu,model);(output/'receipt.json').write_text(json.dumps(data,indent=2)+'\n');print('PASS original-format host short generation compared' if original_source is not None else 'PASS CPU greedy generation reference')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--profile',choices=('short','corpus'),default='short')
 parser.add_argument('--original-source',type=Path)
 parser.add_argument('--reference',type=Path)
 args=parser.parse_args();build(args.cpu_build,args.model,args.output,args.profile,args.original_source,args.reference)
