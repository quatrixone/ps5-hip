"""Strict synthetic dense packed readbacks; numerical agreement is not native origin."""
import argparse,array,hashlib,json,math,re,struct,sys
from pathlib import Path
import dense_layer_reference as reference
ROOT=reference.ROOT
SOURCES=('ports/ggml/dense_layer_fixture.h','ports/ggml/dense_layer_cases.def','tools/dense_layer_reference.py','tests/llm_dense_layer_default_test.cpp','tests/llm_dense_layer_test.cpp','tests/llm_dense_layer_failure_test.cpp','tests/llm_dense_layer_checks_test.cpp','tests/llm_dense_layer_checks.h','tests/llm_layer_numeric_mock.h','kernels/llm_layer.hip','kernels/llm_cpu_math.h','ports/ggml/model_layer.h','ports/ggml/layer_dispatch.h','ports/ggml/prefix_dispatch.h','ports/ggml/model_prefix.h','ports/ggml/prefix_calls.h','tools/build_llm_dense_layer_test.py')
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):
 require(path.is_file() and not path.is_symlink(),'file kind')
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read_json(path,limit=8*1024*1024):
 require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'JSON kind/size')
 blob=path.read_bytes()
 def unique(pairs):
  result={}
  for k,v in pairs:
   require(k not in result,'duplicate JSON key');result[k]=v
  return result
 return json.loads(blob,object_pairs_hook=unique),hashlib.sha256(blob).hexdigest()
def sweep(inputs):
 for name,digest in inputs.items():require(sha(Path(name))==digest,'bound file changed: '+name)
def authenticate_reference(folder):
 own=Path(__file__).resolve();own_test=own.parents[1]/'tests/llm_dense_readbacks_test.py'
 implementation={str(p):sha(p) for p in (own,own_test)}
 folder=Path(folder).resolve();receipt=folder/'receipt.json';data,receipt_hash=read_json(receipt)
 require(data.get('status')=='PASS' and data.get('classification')=='synthetic-dense-layer-host-arithmetic' and data.get('gpu_executed') is False and data.get('native_executed') is False,'reference receipt scope')
 inputs=data.get('inputs');require(isinstance(inputs,dict),'reference inputs')
 source_paths={str(ROOT/n) for n in SOURCES};require(source_paths<=set(inputs),'reference source inventory')
 extra=set(inputs)-source_paths
 caches=[Path(x) for x in extra if x.endswith('/cmake/CMakeCache.txt')];require(len(caches)==1,'CPU cache identity');cpu=caches[0].parents[1];lib=cpu/'cmake/bin'
 expected_extra={str(caches[0]),str(Path('/usr/bin/clang++-18').resolve()),str((lib/'libggml-cpu.so').resolve()),str((lib/'libggml-base.so').resolve())}
 require(extra==expected_extra,'reference CPU/compiler inventory')
 require(all(Path(x).is_absolute() and isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for x,h in inputs.items()),'reference identity syntax')
 sweep(inputs)
 flags=['-mavx2','-mfma','-mf16c','-std=c++17','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-D__HIP_PLATFORM_AMD__']
 flags+=['-I'+str(ROOT/n) for n in ('ports/ggml','ports/hip','.deps/sources/hip-headers/include','.deps/sources/clr/hipamd/include','.deps/sources/llama-client/ggml/include')]
 expected=[]
 for name,source in [('default','tests/llm_dense_layer_default_test.cpp'),('checks','tests/llm_dense_layer_checks_test.cpp'),('failures','tests/llm_dense_layer_failure_test.cpp'),('admission','tests/llm_dense_layer_test.cpp'),('numeric','tests/llm_dense_layer_test.cpp')]:
  binary=folder/name;options=['-O2'] if name=='numeric' else ['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
  expected += [('compile-'+name,['/usr/bin/clang++-18',*flags,*options,str(ROOT/source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml-cpu','-lggml-base','-o',str(binary)]),('linkage-'+name,['/usr/bin/ldd',str(binary)]),(name,[str(binary),*(['admission'] if name=='admission' else ['numeric',str(folder/'reference.bin')] if name=='numeric' else [])])]
 commands=data.get('commands');require(isinstance(commands,list) and len(commands)==len(expected),'reference command inventory')
 closure={**inputs,**implementation}
 for command,(label,argv) in zip(commands,expected):
  require(command.get('label')==label and command.get('argv')==argv and command.get('exit_code')==0,'reference command profile')
  for suffix in ('stdout','stderr'):
   path=folder/(label+'.'+suffix);require(path.stat().st_size<=1024*1024,'command output bound');closure[str(path)]=sha(path)
  if label.startswith('linkage-'):
   text=(folder/(label+'.stdout')).read_text()
   require(all(str(lib/n) in text for n in ('libggml-cpu.so','libggml-base.so')),'reference linked libraries')
 require('DENSE_LAYER_NUMERIC_PASS requests=526 checked=11084 resident_f16_bytes=88080384 max_abs=0 GPU_execution=false' in (folder/'numeric.stdout').read_text(),'reference numeric result')
 raw=folder/'reference.bin';manifest=folder/'reference.json';validated=reference.validate(raw,manifest)
 binding=dict(raw=str(raw),raw_sha256=validated['sha256'],manifest=str(manifest),manifest_sha256=validated['manifest_sha256'],records=11084,bytes=reference.BYTES)
 require(data.get('reference_export')==binding,'reference exported identity')
 closure.update({str(raw):binding['raw_sha256'],str(manifest):binding['manifest_sha256'],str(receipt):receipt_hash})
 sweep(closure)
 return dict(inputs=closure,reference=validated,receipt=dict(path=str(receipt),sha256=receipt_hash),limits=['Prebuilt CPU libraries hash-bound; full CPU producer provenance is not reauthenticated'])
def control_inventory():
 return dict(schema=1,classification='synthetic-dense-layer-readback',raw='readback.bin',requests=526,bytes=reference.BYTES,records=reference.inventory())
def floats(blob,half=False):
 require(len(blob)%(2 if half else 4)==0,'float byte alignment')
 if half:return struct.unpack('<'+str(len(blob)//2)+'e',blob)
 out=array.array('f');out.frombytes(blob)
 if sys.byteorder!='little':out.byteswap()
 return out
def compare_values(actual,expected,half=False,exact=False,probabilities=False):
 require(len(actual)==len(expected),'record size')
 require(not exact or actual==expected,'exact embedding bits')
 a=floats(actual,half);b=floats(expected,half);error=energy=maximum=0.;relative=2e-3 if half else 2e-4
 for x,y in zip(a,b):
  require(math.isfinite(x) and math.isfinite(y),'nonfinite record')
  e=abs(x-y);require(e<=2e-5+relative*abs(y),'element budget')
  require(not ((half or probabilities) and y==0 and x!=0),'exact cache/masked zero')
  error+=e*e;energy+=y*y;maximum=max(maximum,e)
 rms=math.sqrt(error/energy) if energy else (math.inf if error else 0.)
 require(rms<=relative,'relative RMS budget')
 return a,maximum
class Attention:
 def __init__(self):self.reset()
 def reset(self):self.k=bytearray(262144);self.v=bytearray(262144);self.values=[0.]*131072;self.q=None
 def query(self,values):
  try:self.q=struct.unpack('<2048e',struct.pack('<2048e',*values))
  except (OverflowError,struct.error):raise ValueError('query half conversion')
 def update(self,values,position,key):
  require(len(values)==256,'KV width')
  try:blob=struct.pack('<256e',*values)
  except (OverflowError,struct.error):raise ValueError('KV half conversion')
  if key:self.k[position*512:(position+1)*512]=blob;self.values[position*256:(position+1)*256]=struct.unpack('<256e',blob)
  else:
   for i in range(256):self.v[(i*512+position)*2:(i*512+position)*2+2]=blob[i*2:i*2+2]
 def scores(self,actual,extent):
  require(self.q is not None and len(actual)==32*extent,'attention inventory')
  gamma=64*2**-24/(1-64*2**-24);floor=2**-126
  for head in range(32):
   q=self.q[head*64:(head+1)*64];group=(head//8)*64
   for slot in range(extent):
    start=slot*256+group;products=[x*y for x,y in zip(q,self.values[start:start+64])]
    target=math.fsum(products);budget=gamma*math.fsum(map(abs,products))+floor
    require(math.isfinite(actual[head*extent+slot]) and abs(actual[head*extent+slot]-target)<=budget,'device Q/K gamma64 score bound')
 def checkpoint(self,blob,key):require(blob==bytes(self.k if key else self.v),'cache differs from device KV writes')
def compare_stream(rows,candidate,oracle):
 state=Attention();reset={};current_case=None;maximum=0.;actual_hash=hashlib.sha256();expected_hash=hashlib.sha256();records=0
 for row in rows:
  if row['case']!=current_case:state.reset();current_case=row['case']
  a=candidate.read(row['bytes']);b=oracle.read(row['bytes']);require(len(a)==len(b)==row['bytes'],'truncated readback')
  actual_hash.update(a);expected_hash.update(b);require(hashlib.sha256(b).hexdigest()==row['sha256'],'consumed reference changed')
  stage=row['stage'];position=row['position'];values,error=compare_values(a,b,row['dtype']=='f16le',stage=='out-00',stage=='out-09');maximum=max(maximum,error)
  if stage.startswith('out-'):
   if current_case=='sequence' and position<2:reset[position,stage]=a
   if current_case=='reset':require(a==reset[position,stage],'all21 reset bits')
  if stage=='out-04':state.query(values)
  elif stage=='out-05':state.update(values,position,False)
  elif stage=='out-07':state.update(values,position,True)
  elif stage=='out-08':state.scores(values,256 if position<256 else 512)
  elif stage in ('cache-k','cache-v'):state.checkpoint(a,stage=='cache-k')
  records+=1
 require(not candidate.read(1) and not oracle.read(1),'trailing readback bytes')
 return records,maximum,actual_hash.hexdigest(),expected_hash.hexdigest()

def compare(reference_folder,raw,controls,output):
 bound=authenticate_reference(reference_folder);raw=Path(raw);controls=Path(controls);output=Path(output)
 require(not raw.is_symlink() and not controls.is_symlink(),'readback symlink');raw=raw.resolve();controls=controls.resolve()
 require(not output.exists(),'comparison output must be fresh')
 observed,control_hash=read_json(controls)
 require(json.dumps(observed,sort_keys=True)==json.dumps(control_inventory(),sort_keys=True),'readback control inventory')
 require(raw.name=='readback.bin' and raw.is_file() and not raw.is_symlink() and raw.stat().st_size==reference.BYTES,'readback kind/size')
 rows=bound['reference']['records']
 with raw.open('rb') as candidate,(Path(reference_folder)/'reference.bin').open('rb') as oracle:
  records,maximum,actual_digest,expected_digest=compare_stream(rows,candidate,oracle)
 require(records==11084 and expected_digest==bound['reference']['sha256'],'reference stream identity')
 identities={**bound['inputs'],str(raw):actual_digest,str(controls):control_hash};sweep(identities)
 result=dict(status='PASS',classification='synthetic-dense-layer-numerical-comparison',requests=526,records=records,bytes=reference.BYTES,maximum_absolute_error=maximum,inputs=identities,native_origin_established=False,gpu_origin_established=False,limits=bound['limits']+['Native title execution, control provenance, cleanup and hardware origin require separate controller evidence'])
 with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();compare(a.reference,a.raw,a.controls,a.output)
