"""Offline weighted RMSNorm -> shared Q8 -> full Q/K/V readback verification.

File comparisons do not establish hardware origin or absence of identical-byte reuploads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import build_llm_weighted_qkv_reference as reference

ROOT=Path(__file__).resolve().parents[1]
STAGES=('weighted','q8','q','k','v')
CAPACITIES={'input':8192,'norm-weight':8192,'weighted':8192,'q8':2304,'projection-weight':2359296,'output':8192}
DUMP_NAMES={f'weighted-qkv-{s}.{b}.raw' for s in STAGES for b in CAPACITIES}

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def exact(path,size):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=size:raise ValueError('missing or wrong-size asset: '+str(p))
 data=p.read_bytes()
 if len(data)!=size:raise ValueError('asset changed size')
 return data

def seeded(name,payload=b''):
 capacity=CAPACITIES[name]
 if len(payload)>capacity:raise ValueError('payload exceeds capacity')
 fill=b'\xa5' if name in ('input','norm-weight','projection-weight') else b'\x5a'
 return fill*16+payload+fill*(capacity-len(payload)+16)

def authenticate(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';r=json.loads(receipt.read_text())
 if r.get('classification')!='offline-weighted-rmsnorm-full-qkv-reference' or r.get('status')!='PASS' or r.get('gpu_executed') is not False or r.get('native_executed') is not False or r.get('profile')!=reference.PROFILE:raise ValueError('wrong reference contract')
 inputs=r.get('inputs');artifacts=r.get('artifacts')
 if not isinstance(inputs,dict) or not inputs or not isinstance(artifacts,dict) or not artifacts:raise ValueError('empty reference closure')
 identities={str(receipt):sha(receipt)}
 for path,digest in inputs.items():
  p=Path(path)
  if not p.is_absolute() or sha(p)!=digest:raise ValueError('stale reference input: '+path)
  identities[path]=digest
 required=('tools/build_llm_weighted_qkv_reference.py','tests/llm_weighted_qkv_reference.cpp','tests/llm_weighted_qkv_reference_test.py','tools/build_llm_q4_reference.py','tools/build_llm_rmsnorm_reference.py','tools/llm_q4_reference.py','tools/build_public_hip.py')
 if any(str(ROOT/p) not in inputs for p in required):raise ValueError('missing producer identity')
 parents=[Path(p).parent for p in inputs if Path(p).name=='receipt.json' and json.loads(Path(p).read_text()).get('classification')=='host-cpu-rmsnorm-reference']
 if len(parents)!=1:raise ValueError('one authenticated CPU prerequisite required')
 from build_public_hip import rmsnorm_reference_inputs
 parent=rmsnorm_reference_inputs(parents[0])
 if any(inputs.get(p)!=h for p,h in parent['inputs'].items()):raise ValueError('missing CPU prerequisite closure')
 for name,row in artifacts.items():
  p=folder/name
  if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(folder):raise ValueError('unsafe artifact path')
  exact(p,row['bytes'])
  if sha(p)!=row['sha256']:raise ValueError('stale reference artifact')
  identities[str(p)]=row['sha256']
 required_assets={'input.f32le':8192,'run1/norm.f32le':8192,'weighted.fp64le':16384,'weighted.budget.f64le':16384,**{'run1/'+n+'.weights.q4_0':rows*1152 for n,rows in reference.PROFILE['projections'].items()}}
 for n,size in required_assets.items():
  if n not in artifacts or artifacts[n]['bytes']!=size:raise ValueError('missing reference asset')
 data={n:exact(folder/n,size) for n,size in required_assets.items()}
 models=[Path(p) for p in inputs if Path(p).suffix=='.gguf']
 if len(models)!=1:raise ValueError('one locked full model identity required')
 reference.verify_model(models[0])
 with models[0].open('rb') as model:
  for name,(size,offset) in reference.TENSORS.items():
   model.seek(offset)
   if model.read(size)!=data['run1/'+name]:raise ValueError('model tensor identity differs')
 if data['input.f32le']!=exact(parent['assets']['rmsnorm/embedding-1.input.f32le'],8192):raise ValueError('activation identity differs')
 semantic,budgets=reference.weighted_reference(data['input.f32le'],data['run1/norm.f32le'])
 if reference.pack(semantic,'d')!=data['weighted.fp64le'] or reference.pack(budgets,'d')!=data['weighted.budget.f64le']:raise ValueError('reference budget differs from frozen formula')
 if not r.get('cpu_repeats_bitwise_equal') or not r.get('commands') or any(c.get('exit_code')!=0 for c in r['commands']):raise ValueError('reference commands incomplete')
 return data,identities

def verify_buffers(dumps,assets):
 """assets must be authenticated by authenticate before use outside unit tests."""
 dumps=Path(dumps)
 if {p.name for p in dumps.iterdir()}!=DUMP_NAMES:raise ValueError('exact 30 raw files required')
 raw={};identities={}
 for stage in STAGES:
  for name,capacity in CAPACITIES.items():
   p=dumps/f'weighted-qkv-{stage}.{name}.raw';data=exact(p,capacity+32);raw[stage,name]=data;identities[str(p.resolve())]=hashlib.sha256(data).hexdigest()
 weighted=raw['weighted','weighted'][16:-16]
 semantic=struct.unpack('<2048d',assets['weighted.fp64le']);budgets=struct.unpack('<2048d',assets['weighted.budget.f64le'])
 weighted_metrics=reference.compare_weighted(weighted,semantic,budgets)
 weights={n:assets['run1/'+n+'.weights.q4_0'] for n in ('q','k','v')}
 chain=reference.chained_reference(weighted,weights)
 records=[]
 for stage in STAGES:
  expected={'input':seeded('input',assets['input.f32le']),'norm-weight':seeded('norm-weight',assets['run1/norm.f32le']),
   'weighted':seeded('weighted',weighted),'q8':seeded('q8',b'\xcc'*2304 if stage=='weighted' else chain['q8']),
   'projection-weight':seeded('projection-weight',weights[stage] if stage in ('k','v') else weights['q'])}
  for name,data in expected.items():
   if raw[stage,name]!=data:raise ValueError('immutable/shared/guard/inactive mismatch: '+stage+'/'+name)
  output=raw[stage,'output']
  if stage in ('weighted','q8'):
   if output!=seeded('output',struct.pack('<I',0x7fc00000)*2048):raise ValueError('unwritten output seed changed')
   continue
  rows=reference.PROFILE['projections'][stage];end=16+rows*4
  seed=seeded('output')
  if output[:16]!=seed[:16] or output[end:]!=seed[end:]:raise ValueError('output guard/inactive mismatch')
  values=struct.unpack('<'+'f'*rows,output[16:end]);oracle=chain['projections'][stage];errors=[];fractions=[]
  for i,(value,target,budget) in enumerate(zip(values,oracle['fp64'],oracle['budget'])):
   if not all(math.isfinite(v) for v in (value,target,budget)) or budget<0 or abs(value-target)>budget:raise ValueError('projection budget exceeded: '+stage+'/'+str(i))
   errors.append(abs(value-target));fractions.append(abs(value-target)/budget if budget else 0)
  records.append({'projection':stage,'values':rows,'maximum_absolute_error':max(errors),'maximum_budget_fraction':max(fractions)})
 for p,h in identities.items():
  if sha(p)!=h:raise ValueError('dump changed during comparison')
 return {'status':'PASS','classification':'offline-weighted-qkv-readback-comparison','gpu_origin_established':False,'raw_files':30,'raw_bytes':11972800,'weighted_values':2048,'q8_bytes':2304,'projection_values':2560,'weighted_metrics':weighted_metrics,'projections':records,'dump_sha256':identities,'limits':['Byte continuity cannot distinguish identical-byte reupload; native title audit and execution provenance required separately.']}

def verify(reference_folder,dumps,output):
 output=Path(output)
 if output.exists():raise ValueError('output already exists')
 identities={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tools/build_llm_weighted_qkv_reference.py',ROOT/'tools/llm_q4_reference.py')}
 assets,authenticated=authenticate(reference_folder)
 for p,h in authenticated.items():
  if p in identities and identities[p]!=h:raise ValueError('conflicting authenticated identity')
  identities[p]=h
 result=verify_buffers(dumps,assets)
 for p,h in identities.items():
  if sha(p)!=h:raise ValueError('authenticated input changed')
 result['authenticated_inputs']=identities
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('reference','dumps','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();verify(args.reference,args.dumps,args.output)
