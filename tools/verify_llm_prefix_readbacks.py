"""Numerical comparison only; caller authenticates model/reference/native origin."""
import math
import struct
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_llm_rmsnorm_reference as norm
import build_llm_weighted_qkv_reference as weighted

TOKENS=(0,1,2,31999)
WIDTHS={'embedding':2048,'normalized':2048,'weighted':2048,'q':2048,'k':256,'v':256}
FILES={f'token-{t}-{name}.f32le':width*4 for t in TOKENS for name,width in WIDTHS.items()}

def embedding_reference(row):
    if len(row)!=1152:raise ValueError('wrong Q4 embedding row size')
    result=[]
    for offset in range(0,len(row),18):
        scale=struct.unpack_from('<e',row,offset)[0]
        if not math.isfinite(scale):raise ValueError('nonfinite embedding scale')
        quants=row[offset+2:offset+18]
        result.extend(scale*((q&15)-8) for q in quants)
        result.extend(scale*((q>>4)-8) for q in quants)
    return struct.pack('<2048f',*result)

def values(data,count):
    if len(data)!=count*4:raise ValueError('wrong tensor byte count')
    result=struct.unpack('<'+'f'*count,data)
    if not all(math.isfinite(v) for v in result):raise ValueError('nonfinite tensor')
    return result

def compare_projection(data,oracle,rows):
    actual=values(data,rows)
    if any(len(oracle[k])!=rows for k in ('fp64','budget')):raise ValueError('wrong projection oracle shape')
    errors=[];fractions=[]
    for value,target,budget in zip(actual,oracle['fp64'],oracle['budget']):
        if not math.isfinite(target) or not math.isfinite(budget) or budget<0:raise ValueError('invalid projection oracle')
        error=abs(value-target)
        if error>budget:raise ValueError('projection budget exceeded')
        errors.append(error);fractions.append(error/budget if budget else 0)
    return {'elements':rows,'maximum_absolute_error':max(errors),'maximum_budget_fraction':max(fractions)}

def compare_token(raw,embedding,norm_weight,projection_weights):
    if set(raw)!=set(WIDTHS):raise ValueError('exact six outputs required')
    for name,count in WIDTHS.items():values(raw[name],count)
    # Q4 dequantization is exact in F32 for these finite model half scales.
    if raw['embedding']!=embedding:raise ValueError('embedding row mismatch')
    semantic=norm.fp64_reference(values(embedding,2048))
    result={'normalized':norm.compare(values(raw['normalized'],2048),semantic)}
    target,budget=weighted.weighted_reference(embedding,norm_weight)
    result['weighted']=weighted.compare_weighted(raw['weighted'],target,budget)
    # Chain from validated device activation; never upload this oracle to the app.
    chain=weighted.chained_reference(raw['weighted'],projection_weights)
    for name,rows in weighted.PROFILE['projections'].items():
        result[name]=compare_projection(raw[name],chain['projections'][name],rows)
    return result

def compare_buffers(dumps,embeddings,norm_weight,projection_weights):
    if set(dumps)!=set(FILES) or set(embeddings)!=set(TOKENS):raise ValueError('exact four-token inventory required')
    for name,size in FILES.items():
        if len(dumps[name])!=size:raise ValueError('raw output size mismatch')
    return {str(t):compare_token({n:dumps[f'token-{t}-{n}.f32le'] for n in WIDTHS},embeddings[t],norm_weight,projection_weights) for t in TOKENS}


def verify(reference, raw_folder, output):
    """Authenticate CPU/model closure and compare captured bytes; origin is separate."""
    import hashlib
    import json
    from build_llm_prefix_reference import authenticate_prefix_reference
    from source_llm_prefix_log import read_raw
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    data, inputs = authenticate_prefix_reference(reference)
    source = Path(data['reference'])
    def read_source(name):
        path=source/name
        blob=path.read_bytes()
        if path.is_symlink() or inputs.get(str(path))!=hashlib.sha256(blob).hexdigest():
            raise ValueError('consumed reference bytes differ from authenticated identity: '+name)
        return blob
    raw = read_raw(raw_folder)
    raw_hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in raw.items()}
    embeddings = {}
    for token in TOKENS:
        embedding = embedding_reference(read_source(f'token-{token}-embedding.q4_0'))
        if embedding != read_source(f'token-{token}-embedding.f32le'):
            raise ValueError('CPU embedding disagrees with independent Q4 decode')
        embeddings[token] = embedding
    weights = {name:read_source(f'{name}.weights.q4_0') for name in ('q','k','v')}
    metrics = compare_buffers(raw, embeddings, read_source('norm.f32le'), weights)
    if read_raw(raw_folder) != raw:
        raise ValueError('raw readbacks changed during comparison')
    for path, expected in inputs.items():
        if digest(path) != expected:raise ValueError('authenticated reference changed')
    result = dict(status='PASS', classification='actual-model-prefix-numerical-comparison',
        reference_receipt_sha256=data['receipt_sha256'], model=data['model'],
        raw_files=24, raw_bytes=139264, raw_sha256=raw_hashes, tokens=list(TOKENS),
        metrics=metrics, gpu_origin_established=False, inference=False,
        limits=['GPU origin and finalization require the separately reviewed native controller receipt',
                'QKV oracle is chained from validated device weighted activation; no new Q8 byte proof'])
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result
