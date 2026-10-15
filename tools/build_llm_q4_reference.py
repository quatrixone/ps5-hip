"""Build frozen offline GPU-format Q8_1/Q4_0 references from authenticated inputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
from build_public_hip import rmsnorm_reference_inputs
from qualify_public_hip_q4 import fixture_inputs
import llm_q4_reference as oracle

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def exact(path,size):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=size:raise ValueError('wrong reference asset size')
    data=path.read_bytes()
    if len(data)!=size:raise ValueError('reference asset changed size')
    return data


def verify_model(model):
    entry=json.loads((ROOT/'evidence/offline-llm-model-conversion.json').read_text())['outputs']['tinyllama-q4_0.gguf']
    if model.is_symlink() or model.stat().st_size!=entry['bytes'] or sha(model)!=entry['sha256']:raise ValueError('wrong locked model')


def bind(inputs,path):
    path=str(Path(path).resolve());digest=sha(path)
    if path in inputs and inputs[path]!=digest:raise ValueError('input identity changed: '+path)
    inputs[path]=digest


def unchanged(inputs):
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('input changed before publication: '+path)


def pack(values,code):
    if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite reference output')
    return struct.pack('<'+code*len(values),*values)


def build(reference,fixture,model,output):
    reference,fixture,model,output=(Path(p).resolve() for p in (reference,fixture,model,output))
    if output.exists() or output==ROOT/'results' or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh output under results required')
    verify_model(model)
    bound=rmsnorm_reference_inputs(reference);module_inputs=fixture_inputs(fixture)
    inputs=dict(bound['inputs'])
    for path,digest in module_inputs.items():
        if path in inputs and inputs[path]!=digest:raise ValueError('conflicting prerequisite identity')
        inputs[path]=digest
    for name in ['tools/build_llm_q4_reference.py','tests/llm_q4_reference_build_test.py','tests/llm_q4_weights.cpp','tools/llm_q4_reference.py','tests/llm_q4_reference_test.py','tools/build_public_hip.py','tools/qualify_public_hip_q4.py']:
        bind(inputs,ROOT/name)
    bind(inputs,model);bind(inputs,sys.executable)
    receipts=[]
    for path in inputs:
        if Path(path).name=='receipt.json':
            j=json.loads(Path(path).read_text())
            if j.get('classification')=='host-cpu-reference-tools-built':receipts.append(Path(path))
    if len(receipts)!=1:raise ValueError('one authenticated CPU toolchain required')
    libraries=receipts[0].parent/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
    compiler=Path('/usr/bin/clang++-18');bind(inputs,compiler)
    if 'version 18.1.8' not in subprocess.check_output([compiler,'--version'],text=True):raise ValueError('LLVM18.1.8 required')
    output.mkdir(parents=True);commands=[]
    profile=dict(schema=1,oracle=oracle.PROFILE,cases=[dict(name='attn-k-'+str(n),width=2048,rows=n,input='authenticated RMSNorm embedding output row0 token1') for n in (1,2,17,256)]+[dict(name='ffn-down-'+str(n),width=5632,rows=n,input='deterministic synthetic; not a model activation trace') for n in (1,17)],native_executed=False)
    profile['synthetic_cases']=['synthetic-zero','synthetic-pattern']
    profile['q8_cases']=[dict(name=n,input_width=w,padded_width=256) for n,w in [('q8-zero',256),('q8-constant',256),('q8-signed-ties',256),('q8-padding',255)]]
    (output/'declared-profile.json').write_text(json.dumps(profile,indent=2)+'\n');profile_sha=sha(output/'declared-profile.json')
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    def run(label,argv):
        argv=list(map(str,argv));row=dict(label=label,argv=argv);commands.append(row)
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=120,env=env);row['exit_code']=p.returncode
            (output/(label+'.stdout')).write_text(p.stdout);(output/(label+'.stderr')).write_text(p.stderr)
            if p.returncode:raise RuntimeError(label+': '+p.stderr[-1000:])
            return p.stdout
        finally:(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    args=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',ROOT/'tests/llm_q4_weights.cpp','-I'+str(client/'ggml/include')]
    run('dependencies',args+['-M','-MF',output/'extractor.d'])
    for path in shlex.split((output/'extractor.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):bind(inputs,path)
    binary=output/'extractor';run('compile',args+['-L'+str(libraries),'-Wl,-rpath,'+str(libraries),'-lggml','-lggml-base','-o',binary]);bind(inputs,binary)
    loaded=run('runtime-libraries',['ldd',binary])
    for line in loaded.splitlines():
        parts=line.split();path=parts[2] if len(parts)>2 and parts[1]=='=>' else parts[0] if parts else ''
        if path.startswith('/'):bind(inputs,path)
    for repeat in ('weights1','weights2'):
        (output/repeat).mkdir();run(repeat,[binary,model,output/repeat])
    weights={}
    for name,width,rows in [('attn-k',2048,256),('ffn-down',5632,17)]:
        size=width//32*18*rows;data=exact(output/'weights1'/(name+'.weights.q4_0'),size)
        if exact(output/'weights2'/(name+'.weights.q4_0'),size)!=data:raise ValueError('fresh weight extraction differs')
        weights[name]=data
    embedding=exact(bound['assets']['rmsnorm/embedding-1.cpu.f32le'],2048*4)
    synthetic=pack([((i*73)%257-128)/64 for i in range(5632)],'f')
    cases=[];negative_controls=[]
    def case(name,width,rows,activation,weight,origin):
        folder=output/name;folder.mkdir();q8=oracle.quantize_q8_1(activation)
        result=oracle.reference_q4(weight,q8,width,rows)
        first=weight[:width//32*18]
        for label,value in [('nonfinite',float('nan')),('corruption',result['fp64'][0]+4*result['budget'][0]+1)]:
            try:oracle.verify_output(value,first,q8)
            except ValueError:negative_controls.append(name+':'+label)
            else:raise ValueError('negative numerical control accepted')
        assets={'input.f32le':activation,'weights.q4_0':weight,'q8.bin':q8,'fp64le':pack(result['fp64'],'d'),'lane.f32le':pack(result['lane_f32'],'f'),'budget.f64le':pack(result['budget'],'d')}
        for filename,data in assets.items():(folder/filename).write_bytes(data)
        cases.append(dict(name=name,width=width,rows=rows,origin=origin,artifacts={n:dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest()) for n,d in assets.items()}))
        print('Reference '+name,flush=True)
    for entry in profile['cases']:
        name=entry['name'];family='attn-k' if name.startswith('attn-k') else 'ffn-down';width=entry['width'];rows=entry['rows']
        case(name,width,rows,embedding if width==2048 else synthetic,weights[family][:rows*width//32*18],entry['input'])
    for name,values in [('synthetic-zero',[0.0]*2048),('synthetic-pattern',[((i%32)-16)/8 for i in range(2048)])]:
        # Hand-encoded Q4 blocks: halfscale1/8, all16nibble values in low/high planes.
        block=struct.pack('<e',0.125)+bytes(i|((15-i)<<4) for i in range(16))
        case(name,2048,1,pack(values,'f'),block*64,'handencoded synthetic Q4 and activation')
    q8cases=[]
    for name,values,padded in [('q8-zero',[0.0]*256,256),('q8-constant',[1.0]*256,256),('q8-signed-ties',[127.0,-127.0,0.5,-0.5]*64,256),('q8-padding',[((i%31)-15)/8 for i in range(255)],256)]:
        folder=output/name;folder.mkdir();activation=pack(values,'f');q8=oracle.quantize_q8(values,padded_width=padded)
        (folder/'input.f32le').write_bytes(activation);(folder/'q8.bin').write_bytes(q8)
        q8cases.append(dict(name=name,input_width=len(values),padded_width=padded,input_sha256=sha(folder/'input.f32le'),q8_sha256=sha(folder/'q8.bin')))
    if sha(output/'declared-profile.json')!=profile_sha:raise ValueError('declared profile changed')
    if rmsnorm_reference_inputs(reference)!=bound or fixture_inputs(fixture)!=module_inputs:raise ValueError('prerequisite changed')
    verify_model(model);unchanged(inputs)
    receipt=dict(schema=1,status='PASS',classification='offline-gpu-format-q4-reference',inputs=inputs,profile=profile,cases=cases,q8_cases=q8cases,negative_controls=negative_controls,commands=commands,weights_repeated_bitwise_equal=True,native_executed=False,gpu_executed=False,artifacts={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.rglob('*') if p.is_file()},limits=['CPU numerical oracle for GPU-specific quantized formats; no GPU execution or full-model inference.','Synthetic5632activations are not traced model hidden states.'])
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS '+str(output/'receipt.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rmsnorm-reference',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.rmsnorm_reference,a.fixture,a.model,a.output)
