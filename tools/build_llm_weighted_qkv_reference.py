"""Authenticated layer-zero weighted RMSNorm and full frozen Q/K/V CPU oracles."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
from build_llm_q4_reference import bind, exact, pack, sha, unchanged, verify_model
from build_public_hip import rmsnorm_reference_inputs
from build_llm_rmsnorm_reference import fp64_reference, EPSILON
import llm_q4_reference as q4

ROOT=Path(__file__).resolve().parents[1]
WIDTH=2048
PROFILE={
 'schema':1,'width':WIDTH,'rows':1,'epsilon_f32':EPSILON,
 'input':'authenticated original RMSNorm embedding-1 input, token 1',
 'weighted_budget':{'norm_absolute':2e-5,'norm_relative':2e-4,'unit_roundoff':2**-24,
  'rule':'Bnorm=2e-5+2e-4*abs(norm64); Bweighted=abs(w)*Bnorm + u/(1-u)*abs(w)*(abs(norm64)+Bnorm) + 2^-126',
  'derivation':'Original frozen norm envelope covers F32 reduction/rsqrt and scale*x. Multiplication by exact F32 learned weight scales that error by abs(w); one further rounded multiply contributes u/(1-u) times its maximum magnitude. Absolute 2^-126 covers gradual or flushed subnormal multiplication. FP64 semantic-reference error is negligible relative to the norm envelope; CPU baseline must use at most 1/32 of this budget.',
  'status':'Declared before any weighted GPU observation; acceptance envelope, not a measured accuracy claim'},
 'q4_oracle':q4.PROFILE,'projections':{'q':2048,'k':256,'v':256},
 'chain':'Validate actual weighted GPU readback first; derive chained Q8 bytes and Q4 oracles from that readback. Never demand equality to quantized frozen CPU activation.',
 'native_executed':False,'gpu_executed':False}
TENSORS={'norm.f32le':(8192,74768640),'q.weights.q4_0':(2359296,77136128),
         'k.weights.q4_0':(294912,74473728),'v.weights.q4_0':(294912,79495424)}


def floats(data):
 if len(data)!=WIDTH*4:raise ValueError('weighted tensor must be 2048 F32 values')
 values=struct.unpack('<2048f',data)
 if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite weighted tensor')
 return values


def weighted_reference(input_bytes,weight_bytes):
 source,weights=floats(input_bytes),floats(weight_bytes)
 norm=fp64_reference(source);values=[];budgets=[];u=2**-24
 for n,w in zip(norm,weights):
  value=n*w;b=2e-5+2e-4*abs(n)
  budget=abs(w)*b+u/(1-u)*abs(w)*(abs(n)+b)+2**-126
  if not math.isfinite(value) or not math.isfinite(budget) or abs(value)+budget>float.fromhex('0x1.fffffep127'):
   raise ValueError('weighted result outside finite F32 domain')
  values.append(value);budgets.append(budget)
 return values,budgets


def compare_weighted(actual_bytes,reference,budgets):
 actual=floats(actual_bytes)
 if len(reference)!=WIDTH or len(budgets)!=WIDTH:raise ValueError('weighted oracle shape')
 fractions=[];errors=[]
 for i,(a,b,limit) in enumerate(zip(actual,reference,budgets)):
  if not math.isfinite(b) or not math.isfinite(limit) or limit<=0:raise ValueError('invalid weighted budget')
  error=abs(a-b)
  if error>limit:raise ValueError('weighted error budget exceeded at '+str(i))
  errors.append(error);fractions.append(error/limit)
 return {'elements':WIDTH,'maximum_absolute_error':max(errors),'maximum_budget_fraction':max(fractions)}


def chained_reference(weighted_readback,projection_weights):
 """Pure numerical stage oracle; caller separately authenticates native origin/guards."""
 floats(weighted_readback)
 if set(projection_weights)!=set(PROFILE['projections']):raise ValueError('exact Q/K/V weights required')
 quantized=q4.quantize_q8_1(weighted_readback)
 return {'q8':quantized,'projections':{name:q4.reference_q4(projection_weights[name],quantized,WIDTH,rows)
                                     for name,rows in PROFILE['projections'].items()}}


def build(reference,model,output):
 reference,model,output=(Path(p).resolve() for p in (reference,model,output))
 if output.exists() or output==ROOT/'results' or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh output under results required')
 verify_model(model);bound=rmsnorm_reference_inputs(reference);inputs=dict(bound['inputs'])
 for name in ('tools/build_llm_weighted_qkv_reference.py','tests/llm_weighted_qkv_reference.cpp','tests/llm_weighted_qkv_reference_test.py',
              'tools/build_llm_q4_reference.py','tools/build_llm_rmsnorm_reference.py','tools/llm_q4_reference.py','tests/llm_q4_reference_test.py','tools/build_public_hip.py'):
  bind(inputs,ROOT/name)
 bind(inputs,model);bind(inputs,sys.executable)
 cpus=[]
 for path in inputs:
  if Path(path).name=='receipt.json' and json.loads(Path(path).read_text()).get('classification')=='host-cpu-reference-tools-built':cpus.append(Path(path))
 if len(cpus)!=1:raise ValueError('one authenticated CPU build required')
 libraries=cpus[0].parent/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 compiler=Path('/usr/bin/clang++-18');bind(inputs,compiler)
 if 'version 18.1.8' not in subprocess.check_output([compiler,'--version'],text=True):raise ValueError('LLVM18.1.8 required')
 activation=exact(bound['assets']['rmsnorm/embedding-1.input.f32le'],8192);floats(activation)
 output.mkdir(parents=True);(output/'input.f32le').write_bytes(activation)
 (output/'declared-profile.json').write_text(json.dumps(PROFILE,indent=2)+'\n');profile_hash=sha(output/'declared-profile.json')
 env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None);commands=[]
 def run(label,argv):
  row={'label':label,'argv':list(map(str,argv))};commands.append(row)
  try:
   p=subprocess.run(row['argv'],cwd=ROOT,capture_output=True,text=True,timeout=120,env=env);row['exit_code']=p.returncode
   (output/(label+'.stdout')).write_text(p.stdout);(output/(label+'.stderr')).write_text(p.stderr)
   if p.returncode:raise RuntimeError(label+': '+p.stderr[-2000:])
   return p.stdout
  except subprocess.TimeoutExpired as error:
   row['timed_out']=True
   for suffix,value in [('stdout',error.stdout),('stderr',error.stderr)]:(output/(label+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
   raise
  finally:(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 args=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',ROOT/'tests/llm_weighted_qkv_reference.cpp','-I'+str(client/'ggml/include')]
 run('dependencies',args+['-M','-MF',output/'fixture.d'])
 for name in shlex.split((output/'fixture.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):bind(inputs,name)
 binary=output/'fixture';run('compile',args+['-L'+str(libraries),'-Wl,-rpath,'+str(libraries),'-lggml','-lggml-cpu','-lggml-base','-o',binary]);bind(inputs,binary)
 for line in run('runtime-libraries',['ldd',binary]).splitlines():
  parts=line.split();path=parts[2] if len(parts)>2 and parts[1]=='=>' else parts[0] if parts else ''
  if path.startswith('/'):bind(inputs,path)
 expected=[f'{name} {size} {offset}' for name,(size,offset) in TENSORS.items()]+['WEIGHTED_QKV_CPU_PASS width=2048 q=2048 k=256 v=256 threads=1 gpu_execution=false']
 for repeat in ('run1','run2'):
  (output/repeat).mkdir()
  if run(repeat,[binary,model,output/'input.f32le',output/repeat]).splitlines()!=expected:raise ValueError('tensor identity or CPU transcript differs')
 for name,size in [(n,v[0]) for n,v in TENSORS.items()]+[('weighted.cpu.f32le',8192)]:
  if exact(output/'run1'/name,size)!=exact(output/'run2'/name,size):raise ValueError('CPU extraction/repeats differ')
 # Independently bind exact locked model offsets, not merely extractor self-reports.
 with model.open('rb') as f:
  for name,(size,offset) in TENSORS.items():
   f.seek(offset)
   if f.read(size)!=exact(output/'run1'/name,size):raise ValueError('model tensor byte mismatch')
 weights=exact(output/'run1/norm.f32le',8192);semantic,budgets=weighted_reference(activation,weights)
 cpu=exact(output/'run1/weighted.cpu.f32le',8192);metrics=compare_weighted(cpu,semantic,budgets)
 if metrics['maximum_budget_fraction']>1/32:raise ValueError('CPU baseline consumes weighted budget')
 (output/'weighted.fp64le').write_bytes(pack(semantic,'d'));(output/'weighted.budget.f64le').write_bytes(pack(budgets,'d'))
 quantized=q4.quantize_q8_1(cpu);(output/'frozen.q8.bin').write_bytes(quantized);projections={}
 for name,rows in PROFILE['projections'].items():
  weight=exact(output/'run1'/(name+'.weights.q4_0'),rows*1152);result=q4.reference_q4(weight,quantized,WIDTH,rows)
  for field,code in [('fp64','d'),('lane_f32','f'),('budget','d')]:(output/(name+'.'+field)).write_bytes(pack(result[field],code))
  projections[name]={'rows':rows,'width':WIDTH,'maximum_lane_error':max(abs(a-b) for a,b in zip(result['lane_f32'],result['fp64']))}
  print('Frozen projection '+name+' rows='+str(rows),flush=True)
 if sha(output/'declared-profile.json')!=profile_hash or exact(output/'input.f32le',8192)!=activation:raise ValueError('frozen profile/input changed')
 if rmsnorm_reference_inputs(reference)!=bound:raise ValueError('prerequisite changed')
 verify_model(model);unchanged(inputs)
 receipt={'schema':1,'status':'PASS','classification':'offline-weighted-rmsnorm-full-qkv-reference','inputs':inputs,'profile':PROFILE,
          'weighted_metrics':metrics,'projections':projections,'cpu_repeats_bitwise_equal':True,'commands':commands,
          'gpu_executed':False,'native_executed':False,'artifacts':{str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.rglob('*') if p.is_file()},
          'limits':['One layer0 token1 input; not attention, RoPE, a complete layer or inference.',
                    'Frozen CPU activation Q8 is a separate control. Native chained Q8 must be derived from validated actual GPU weighted output.',
                    'Pure chained_reference does not authenticate hardware origin; controller/readback provenance is a future gate.']}
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS '+str(output/'receipt.json'),flush=True)
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rmsnorm-reference',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();build(a.rmsnorm_reference,a.model,a.output)
