"""Predeclared whole-layer CPU comparison; native origin requires controller evidence."""
import hashlib,json,math,struct
from pathlib import Path
from source_llm_layer_raw import FILES,RAW_BYTES,read_raw
# CPU-compatible Q4 packing and F32 accumulation: tight float tolerance plus
# two half-precision relative rounding steps for persistent KV storage.
F32_ABS=2e-5
F32_REL=2e-4
F16_ABS=2e-5
F16_REL=2e-3
RMS_REL=2e-4

def score_oracle(name,actual):
 # Cache slots are append-only; the final cache contains each earlier checkpoint.
 # Future slots were still zero when that checkpoint was computed.
 parts=name.split('-');run=int(parts[1]);position=int(parts[3]);extent=256 if position<256 else 512
 cache=struct.unpack('<131072e',actual[f'run-{run}-cache-k.f16le'])
 query=struct.unpack('<2048f',actual[f'run-{run}-pos-{position}-out-04.f32le'])
 query=struct.unpack('<2048e',struct.pack('<2048e',*query))
 targets=[];budgets=[];epsilon=2**-24;gamma=64*epsilon/(1-64*epsilon)
 for head in range(32):
  for slot in range(extent):
   products=[query[head*64+d]*cache[slot*256+(head//8)*64+d] for d in range(64)] if slot<=position else [0.]*64
   target=math.fsum(products)
   targets.append(target);budgets.append(gamma*math.fsum(abs(v) for v in products)+2**-126)
 return targets,budgets

def compare(actual,reference):
 if set(actual)!=set(FILES) or set(reference)!=set(FILES):raise ValueError('layer comparison inventory')
 metrics={}
 for name,size in sorted(FILES.items(),key=lambda item:not item[0].endswith(".f16le")):
  a,b=actual[name],reference[name]
  if len(a)!=size or len(b)!=size:raise ValueError('layer comparison size')
  half=name.endswith('.f16le');fmt='e' if half else 'f';width=2 if half else 4
  av=struct.unpack('<'+str(size//width)+fmt,a);bv=struct.unpack('<'+str(size//width)+fmt,b)
  absolute,relative=(F16_ABS,F16_REL) if half else (F32_ABS,F32_REL)
  if not all(math.isfinite(x) for x in (*av,*bv)):raise ValueError('layer comparison nonfinite')
  errors=[abs(x-y) for x,y in zip(av,bv)]
  if name.endswith('out-08.f32le'):
   targets,budgets=score_oracle(name,actual)
   if any(abs(x-y)>budget for x,y,budget in zip(av,targets,budgets)):raise ValueError('layer chained score budget: '+name)
  elif any(e>absolute+relative*abs(y) for e,y in zip(errors,bv)):raise ValueError('layer element budget: '+name)
  if name.endswith('out-00.f32le') and a!=b:raise ValueError('layer embedding must be exact')
  if half and any(x!=0 for x,y in zip(av,bv) if y==0):raise ValueError('layer zero cache differs')
  if name.endswith('out-09.f32le') and any(x!=0 for x,y in zip(av,bv) if y==0):raise ValueError('layer masked probability differs')
  energy=sum(y*y for y in bv);error_energy=sum(e*e for e in errors)
  rms=math.sqrt(error_energy/energy) if energy else (0 if not error_energy else math.inf)
  if rms>(F16_REL if half else RMS_REL):raise ValueError('layer RMS budget: '+name)
  metrics[name]={'maximum_absolute_error':max(errors),'relative_rms_error':rms,'elements':len(av),'element_oracle':'validated-device-Q-and-K-F64-dot' if name.endswith('out-08.f32le') else 'CPU'}
 for position in (0,1):
  for index in range(21):
   if actual[f'run-0-pos-{position}-out-{index:02d}.f32le']!=actual[f'run-1-pos-{position}-out-{index:02d}.f32le']:raise ValueError('layer reset is not bitwise repeatable')
 return metrics

def verify(reference,raw_folder,output):
 from build_llm_layer_reference import authenticate_layer_reference
 data,inputs=authenticate_layer_reference(reference)
 source=Path(data['reference']);expected=read_raw(source)
 for name,blob in expected.items():
  if inputs.get(str(source/name))!=hashlib.sha256(blob).hexdigest():raise ValueError('reference consumed bytes differ')
 actual=read_raw(raw_folder);metrics=compare(actual,expected)
 if read_raw(raw_folder)!=actual:raise ValueError('layer raw changed during comparison')
 for name,digest in inputs.items():
  with Path(name).open('rb') as stream:
   if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('layer reference changed')
 result=dict(status='PASS',classification='actual-model-layer-numerical-comparison',reference_receipt_sha256=data['receipt_sha256'],model=data['model'],raw_files=len(FILES),raw_bytes=RAW_BYTES,raw_sha256={n:hashlib.sha256(b).hexdigest() for n,b in actual.items()},metrics=metrics,budgets=dict(f32_absolute=F32_ABS,f32_relative=F32_REL,f32_rms_relative=RMS_REL,f16_absolute=F16_ABS,f16_relative=F16_REL,score_accumulation='gamma64 times sum absolute F16 products; CPU RMS gate retained'),gpu_origin_established=False,inference=False,limits=['Native origin and cleanup require the separate controller receipt','Attention score element bounds chain from independently CPU-validated GPU Q/K; CPU RMS comparison remains required','One selected real-model layer; full llama decoder progression is not established'])
 Path(output).write_text(json.dumps(result,indent=2)+'\n')
 return result
