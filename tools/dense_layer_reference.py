"""Exact packed CPU reference for the synthetic 526-request dense fixture."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BYTES=154392576
RECORDS=11084

def inventory():
 rows=(ROOT/'ports/ggml/dense_layer_cases.def').read_text().splitlines()
 cases=[]
 for row in rows:
  m=re.fullmatch(r'PS5HIP_DENSE_CASE\(([a-z_]+),([0-9]+),([-0-9.e]+f)\)',row)
  if not m:raise ValueError('dense case syntax')
  cases.append((m[1],int(m[2]),m[3]))
 if len(cases)!=8 or len({x[0] for x in cases})!=8 or [x[1] for x in cases]!=[512]+[2]*7:raise ValueError('dense case inventory')
 result=[];offset=0;request=0
 for case,count,scale in cases:
  for position in range(count):
   extent=256 if position<256 else 512
   sizes=[8192]*5+[1024]*3+[128*extent]*2+[8192]*6+[22528]*3+[8192]*2
   fields=[(f'out-{i:02d}','f32le',size) for i,size in enumerate(sizes)]
   if position in (0,1,255,256,511):fields += [('cache-k','f16le',262144),('cache-v','f16le',262144)]
   for stage,dtype,size in fields:
    result.append(dict(request=request,case=case,position=position,scale=scale,stage=stage,dtype=dtype,offset=offset,bytes=size));offset+=size
   request+=1
 if request!=526 or len(result)!=RECORDS or offset!=BYTES:raise ValueError('dense packed geometry')
 return result

def _scan(raw,rows):
 if raw.is_symlink() or not raw.is_file() or raw.stat().st_size!=BYTES:raise ValueError('dense raw kind/size')
 digest=hashlib.sha256();hashed=[]
 with raw.open('rb') as stream:
  for row in rows:
   blob=stream.read(row['bytes'])
   if len(blob)!=row['bytes']:raise ValueError('dense raw truncated')
   digest.update(blob);hashed.append({**row,'sha256':hashlib.sha256(blob).hexdigest()})
  if stream.read(1):raise ValueError('dense raw trailing bytes')
 return hashed,digest.hexdigest()

def write_manifest(raw,manifest):
 if manifest.exists() or manifest.is_symlink():raise ValueError('dense manifest must be fresh')
 rows,digest=_scan(raw,inventory())
 data=dict(schema=1,classification='synthetic-dense-layer-cpu-reference',raw='reference.bin',bytes=BYTES,requests=526,records=rows,sha256=digest,gpu_executed=False)
 with manifest.open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
 return data

def validate(raw,manifest):
 if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size>8*1024*1024:raise ValueError('dense manifest kind/size')
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('duplicate manifest key')
   out[k]=v
  return out
 manifest_bytes=manifest.read_bytes()
 data=json.loads(manifest_bytes,object_pairs_hook=unique)
 rows=inventory();records=data.get('records')
 if not isinstance(records,list) or len(records)!=len(rows):raise ValueError('dense record inventory')
 for actual,expected in zip(records,rows):
  if not isinstance(actual,dict) or set(actual)!=set(expected)|{'sha256'} or any(actual[k]!=v for k,v in expected.items()):raise ValueError('dense record identity/offset')
 if set(data)!={'schema','classification','raw','bytes','requests','records','sha256','gpu_executed'} or data['schema']!=1 or data['classification']!='synthetic-dense-layer-cpu-reference' or data['raw']!='reference.bin' or data['bytes']!=BYTES or data['requests']!=526 or data['gpu_executed'] is not False:raise ValueError('dense manifest identity')
 observed,digest=_scan(raw,rows)
 if observed!=records or digest!=data['sha256']:raise ValueError('dense reference hash mismatch')
 return {**data,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest()}
