"""Offline raw-buffer oracle. Dump provenance must be established separately."""
import hashlib
import math
from pathlib import Path
import struct

PREFIX = 16
CAPACITIES = dict(input=5632*4, weights=256*64*18, q8=176*36, output=256*4)
MATRIX_CASES = tuple((f'attn-k-{n}',2048,n) for n in (1,2,17,256)) + tuple(
    (f'ffn-down-{n}',5632,n) for n in (1,17)) + (
    ('synthetic-zero',2048,1),('synthetic-pattern',2048,1))
Q8_CASES = tuple((name,width,256) for name,width in (
    ('q8-zero',256),('q8-constant',256),('q8-signed-ties',256),('q8-padding',255)))
MATRIX_FILES = ('input.f32le','weights.q4_0','q8.bin','fp64le','lane.f32le','budget.f64le')
ASSET_NAMES = {f'q4/{name}/{suffix}' for name,_,_ in MATRIX_CASES for suffix in MATRIX_FILES} | {
    f'q4/{name}/{suffix}' for name,_,_ in Q8_CASES for suffix in ('input.f32le','q8.bin')}
STAGES = tuple((name,'q8',width,0,padded) for name,width,padded in Q8_CASES) + tuple(
    (name,stage,width,rows,width) for name,width,rows in MATRIX_CASES for stage in ('frozen','q8','chained'))
DUMP_NAMES = {f'q4-{name}.{stage}.{buffer}.raw' for name,stage,_,_,_ in STAGES for buffer in CAPACITIES}


def exact(path,size):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError('missing, symlinked or wrong-size buffer: '+str(path))
    data=path.read_bytes()
    if len(data)!=size:raise ValueError('buffer changed size')
    return data


def seeded(buffer,payload=b''):
    capacity=CAPACITIES[buffer]
    if len(payload)>capacity:raise ValueError('payload exceeds buffer capacity')
    fill=b'\xa5' if buffer in ('input','weights') else b'\x5a'
    return fill*PREFIX+payload+fill*(capacity-len(payload)+16)


def verify_buffers(dumps,assets):
    """Require an authenticated exact asset map; check every byte outside Q4 output."""
    if set(assets)!=ASSET_NAMES:raise ValueError('exact Q4 reference asset map required')
    dumps=Path(dumps)
    if {p.name for p in dumps.glob('q4-*.raw')}!=DUMP_NAMES:raise ValueError('exact112 raw dumps required')
    identities={};records=[];total_values=0;total_q8=0
    def read(path,size):
        data=exact(path,size);key=str(Path(path).resolve());digest=hashlib.sha256(data).hexdigest()
        if key in identities and identities[key]!=digest:raise ValueError('file changed during verification')
        identities[key]=digest;return data
    for name,stage,width,rows,padded in STAGES:
        def asset(suffix,size):return read(assets[f'q4/{name}/{suffix}'],size)
        activation=asset('input.f32le',width*4)
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',activation)):
            raise ValueError('nonfinite activation reference')
        quantized=asset('q8.bin',padded//32*36)
        weights=asset('weights.q4_0',rows*(width//32)*18) if rows else b''
        reference=budget=()
        if rows:
            reference=struct.unpack('<'+'d'*rows,asset('fp64le',rows*8))
            budget=struct.unpack('<'+'d'*rows,asset('budget.f64le',rows*8))
            lane=struct.unpack('<'+'f'*rows,asset('lane.f32le',rows*4))
            if not all(math.isfinite(r) and math.isfinite(b) and b>=0 and math.isfinite(l) and abs(l-r)<=b
                       for r,b,l in zip(reference,budget,lane)):
                raise ValueError('invalid numerical reference')
        output_seed=struct.pack('<I',0x7fc00000)*rows
        expected=dict(input=seeded('input',activation),weights=seeded('weights',weights),
                      q8=seeded('q8',quantized),output=seeded('output',output_seed))
        maximum_error=maximum_fraction=0.0
        for buffer,capacity in CAPACITIES.items():
            raw=read(dumps/f'q4-{name}.{stage}.{buffer}.raw',capacity+32)
            if buffer!='output' or not rows or stage=='q8':
                if raw!=expected[buffer]:raise ValueError(f'input/quantization/guard/inactive mismatch: {name}/{stage}/{buffer}')
                continue
            end=PREFIX+rows*4
            if raw[:PREFIX]!=expected[buffer][:PREFIX] or raw[end:]!=expected[buffer][end:]:
                raise ValueError('output guard/inactive mismatch: '+name+'/'+stage)
            values=struct.unpack('<'+'f'*rows,raw[PREFIX:end])
            for index,(value,r,b) in enumerate(zip(values,reference,budget)):
                error=abs(value-r)
                if not math.isfinite(value) or error>b:
                    raise ValueError(f'Q4 numerical mismatch: {name}/{stage}/{index}')
                maximum_error=max(maximum_error,error)
                maximum_fraction=max(maximum_fraction,error/b if b else 0.0)
            total_values+=rows
        total_q8+=len(quantized)
        records.append(dict(name=name,stage=stage,rows=rows,maximum_absolute_error=maximum_error,
                            maximum_budget_fraction=maximum_fraction))
    for name,digest in identities.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('file changed during verification')
    if total_values!=592 or total_q8!=80640:raise ValueError('unexpected verification totals')
    return dict(classification='offline-q4-raw-buffer-comparison',gpu_origin_established=False,
                stages=len(records),raw_files=len(DUMP_NAMES),q4_values=total_values,q8_bytes=total_q8,
                raw_bytes=sum(c+32 for c in CAPACITIES.values())*len(STAGES),records=records,sha256=identities)


def verify(fixture, reference, dumps, output):
    """Authenticate the oracle before comparing raw bytes; lifecycle proof is separate."""
    from public_hip_q4_assets import assets as authenticate
    import json
    implementation=Path(__file__).resolve()
    implementation_sha=hashlib.sha256(implementation.read_bytes()).hexdigest()
    output=Path(output)
    if output.exists():raise ValueError('verification output already exists')
    authenticated=authenticate(fixture,reference)
    oracle_assets={n:v for n,v in authenticated['assets'].items() if n in ASSET_NAMES}
    report=verify_buffers(dumps,oracle_assets)
    report.update(status='PASS',reference=authenticated['reference'],
                  authenticated_inputs=authenticated['inputs'])
    report['implementation_sha256']=implementation_sha
    for name,digest in authenticated['inputs'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('authenticated input changed during raw verification')
    if hashlib.sha256(implementation.read_bytes()).hexdigest()!=implementation_sha:
        raise ValueError('verifier implementation changed during verification')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','reference','dumps','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    verify(args.fixture,args.reference,args.dumps,args.output)
