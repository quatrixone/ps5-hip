"""Capture complete successful CPU scheduler graphs using existing CPU libraries."""
import argparse,json,os,subprocess,struct,math
from pathlib import Path
import build_llm_executed_graph as provenance
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=[(0,p) for p in (0,1,255,256,511)]+[(1,p) for p in (0,1)]
LINE='FULL_GRAPH_TRACE_PASS requests=514 graphs=7 original_cpu_scheduler=true GPU_execution=false\n'

def validate(folder):
 folder=Path(folder);steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
 expected=[(r,p) for r,n in ((0,512),(1,2)) for p in range(n)]
 if len(steps)!=514:raise ValueError('trace step count')
 for row,(run,pos) in zip(steps,expected):
  if row!={'run':run,'position':pos,'token':(0,1,2,31999)[pos%4],'status':0,'scheduler_calls':1,'committed_position':pos}:raise ValueError('trace committed progression')
 files={'steps.jsonl'};summary=[]
 for run,pos in CHECKPOINTS:
  stem=f'run-{run}-pos-{pos}';graph=json.loads((folder/(stem+'.graph.json')).read_text());files.add(stem+'.graph.json')
  tensors=graph['tensors'];nodes=graph['nodes'];ids=set(range(len(tensors)))
  if graph['run']!=run or graph['position']!=pos or not 0<len(nodes)<4096 or len(tensors)>=8192 or len(set(nodes))!=len(nodes) or not set(nodes)<=ids or not set(graph['leaves'])<=ids:raise ValueError('graph inventory')
  controls={};ops={};leaf_bytes=0;owner_bytes=0
  if [i for i in nodes if not tensors[i]['compute']]!=[1,2,3]:raise ValueError('inactive branch differs')
  for index,t in enumerate(tensors):
   if t['id']!=index or any(s!=-1 and s not in ids for s in t['sources']) or t['view_source']!=-1 and t['view_source'] not in ids or len(bytes.fromhex(t['op_params_hex']))!=64 or len(t['ne'])!=4 or len(t['nb'])!=4 or t['view_offset']<0 or t['bytes']<=0:raise ValueError('tensor identity/layout')
   if t['view_source']==-1:owner_bytes+=t['bytes']
   if t['op']=='NONE':leaf_bytes+=t['bytes']
   if index in nodes:ops[t['description']]=ops.get(t['description'],0)+1
   name=t['control_file']
   if name:
    if Path(name).name!=name or name in files:raise ValueError('control file name')
    raw=(folder/name).read_bytes();files.add(name)
    if len(raw)!=t['bytes']:raise ValueError('control size')
    controls[t['name']]={'id':index,'dtype':t['type'],'bytes':len(raw),'hex':raw.hex()}
  for name,value,fmt in [('inp_tokens',(0,1,2,31999)[pos%4],'i'),('inp_pos',pos,'i'),('attn_inp_k_idxs',pos,'q'),('out_ids',0,'i')]:
   if bytes.fromhex(controls[name]['hex'])!=struct.pack('<'+fmt,value):raise ValueError('token/position/index')
  if bytes.fromhex(controls['attn_inp_v_idxs']['hex'])!=struct.pack('<256q',*(i*512+pos for i in range(256))):raise ValueError('V indices')
  mask=bytes.fromhex(controls['attn_inp_kq_mask']['hex']);n=256 if pos<256 else 512
  if mask!=struct.pack('<'+str(n)+'f',*(0. if i<=pos else -math.inf for i in range(n))):raise ValueError('causal mask')
  logits=folder/(stem+'.logits.f32le');files.add(logits.name);values=struct.unpack('<32000f',logits.read_bytes())
  if not all(map(math.isfinite,values)):raise ValueError('finite logits')
  gathers=[{'id':t['id'],'name':t['name'],'type':t['type'],'source_type':tensors[t['sources'][0]]['type'],'shape':t['ne'],'indices':t['sources'][1]} for t in tensors if t['op']=='GET_ROWS']
  summary.append({'run':run,'position':pos,'nodes':len(nodes),'tensors':len(tensors),'ops':ops,'leaf_span_bytes':leaf_bytes,'sum_nonview_tensor_bytes':owner_bytes,'gathers':gathers,'controls':controls,'inactive_nodes':[i for i in nodes if not tensors[i]['compute']]})
 if {p.name for p in folder.iterdir()}!=files or any(p.is_symlink() for p in folder.iterdir()):raise ValueError('trace output inventory')
 for p in (0,1):
  if (folder/f'run-0-pos-{p}.logits.f32le').read_bytes()!=(folder/f'run-1-pos-{p}.logits.f32le').read_bytes():raise ValueError('reset logits differ')
 return summary

def authenticate_full_graph_trace(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=provenance.sha(receipt);data=json.loads(receipt.read_text())
 if data.get('status')!='PASS' or data.get('classification')!='successful-full-model-cpu-scheduler-trace' or data.get('gpu_executed') is not False or data.get('requests')!=514 or data.get('graphs')!=7:raise ValueError('trace receipt contract')
 inputs=data['inputs']
 required=[Path(__file__).resolve(),ROOT/'tests/llm_full_graph_trace.cpp',ROOT/'tests/llm_full_graph_trace_test.py',Path(provenance.__file__).resolve(),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json']
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
 inputs=provenance.cpu_inputs(cpu,model);harness=ROOT/'tests/llm_full_graph_trace.cpp';compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 for p in (Path(__file__).resolve(),harness,ROOT/'tests/llm_full_graph_trace_test.py',Path(provenance.__file__).resolve(),compiler,Path('/usr/bin/ldd'),ROOT/'llm-inputs.json',ROOT/'evidence/offline-llm-model-conversion.json'):provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True);raw=output/'raw';raw.mkdir();commands=[]
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 def run(label,args):
  result=subprocess.run(list(map(str,args)),cwd=ROOT,env=env,capture_output=True,timeout=1800);(output/(label+'.stdout')).write_bytes(result.stdout);(output/(label+'.stderr')).write_bytes(result.stderr);commands.append({'label':label,'argv':list(map(str,args)),'exit_code':result.returncode});(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if result.returncode:raise ValueError(label+' failed')
  return result.stdout.decode()
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(client/'include'),'-I'+str(client/'ggml/include'),'-isystem',str(client/'ggml/src')]
 run('headers',[compiler,*flags,'-M','-MF',output/'headers.d',harness])
 for p in provenance.dependencies(output/'headers.d'):provenance.merge(inputs,p,provenance.sha(p))
 run('compile',[compiler,*flags,'-MD','-MF',output/'compile.d',harness,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--export-dynamic','-lllama','-lggml-cpu','-lggml-base','-ldl','-o',output/'trace'])
 if provenance.dependencies(output/'headers.d')!=provenance.dependencies(output/'compile.d'):raise ValueError('compiler dependency drift')
 for p,h in provenance.runtime_inputs(run('ldd',['/usr/bin/ldd',output/'trace']),lib,inputs).items():provenance.merge(inputs,p,h)
 binary=provenance.sha(output/'trace');line=run('capture',[output/'trace',model,raw])
 if line!=LINE or binary!=provenance.sha(output/'trace'):raise ValueError('capture completion/binary')
 summary=validate(raw);(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');provenance.recheck(inputs)
 artifacts={str(p.relative_to(output)):provenance.sha(p) for p in output.rglob('*') if p.is_file()}
 if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>16*1024*1024:raise ValueError('trace bound')
 (output/'receipt.json').write_text(json.dumps({'status':'PASS','classification':'successful-full-model-cpu-scheduler-trace','cpu_build':str(cpu),'model':{'path':str(model),'bytes':model.stat().st_size,'sha256':inputs[str(model)]},'inputs':inputs,'artifacts':artifacts,'commands':commands,'requests':514,'graphs':7,'gpu_executed':False,'limits':['CPU diagnostic, not GPU admission','Tensor byte sums are logical spans, not actual peak allocator residency','Single run with reset equality, not independent repeated-run determinism']},indent=2)+'\n');print('PASS complete CPU scheduler graphs,514 committed requests,7 checkpoints')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();build(args.cpu_build,args.model,args.output)
