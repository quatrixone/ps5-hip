"""Build genuine CPU RMSNorm references for the locked TinyLlama model.

Run after the documented CPU build/model conversion prerequisites:
  python3 tools/build_llm_rmsnorm_reference.py --cpu-build build/llm-cpu-reference-v2 \
    --model models/tinyllama-gguf/tinyllama-q4_0.gguf --output results/llm-rmsnorm-reference-v1
No GPU execution, model inference, conversion or dependency installation occurs.
"""
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
ROOT=Path(__file__).resolve().parents[1]
CLIENT=ROOT/'.deps/sources/llama-client'
WIDTH=2048
EPSILON=struct.unpack('<f',struct.pack('<f',1e-5))[0]
TOLERANCE={'absolute':2e-5,'relative':2e-4,'rule':'abs(actual-reference) <= absolute + relative*abs(reference)',
           'status':'proposed frozen operator acceptance budget; not a GPU accuracy claim',
           'rationale':'Allows float32 tree/FMA/rsqrt differences from CPU wider accumulation/sqrt; CPU-vs-FP64 baseline must consume at most1/32 of this budget. No GPU result was used to tune it.'}


def sha(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def read_f32(path,count):
    data=Path(path).read_bytes()
    if len(data)!=count*4:raise ValueError('wrong F32 byte count')
    values=struct.unpack('<'+'f'*count,data)
    if not all(math.isfinite(x) for x in values):raise ValueError('non-finite F32 tensor')
    return values


def fp64_reference(values):
    if not values or len(values)%WIDTH:raise ValueError('invalid RMSNorm shape')
    result=[]
    for start in range(0,len(values),WIDTH):
        row=values[start:start+WIDTH]
        if not all(math.isfinite(x) for x in row):raise ValueError('non-finite input')
        scale=1/math.sqrt(math.fsum(float(x)*x for x in row)/WIDTH+EPSILON)
        result.extend(x*scale for x in row)
    return result


def compare(actual,reference,require_pass=True):
    if not actual or len(actual)!=len(reference):raise ValueError('oracle shape mismatch')
    absolute=[];relative=[];fractions=[]
    for i,(a,b) in enumerate(zip(actual,reference)):
        if not math.isfinite(a) or not math.isfinite(b):raise ValueError('non-finite oracle value')
        error=abs(a-b);budget=TOLERANCE['absolute']+TOLERANCE['relative']*abs(b)
        if require_pass and error>budget:raise ValueError('numerical oracle mismatch at '+str(i))
        absolute.append(error);relative.append(error/max(abs(b),1e-30));fractions.append(error/budget)
    return dict(elements=len(actual),maximum_absolute_error=max(absolute),maximum_relative_error=max(relative),
                rms_error=math.sqrt(math.fsum(x*x for x in absolute)/len(absolute)),maximum_budget_fraction=max(fractions))


def build(output,cpu_build,model):
    output,cpu_build,model=(Path(p).resolve() for p in (output,cpu_build,model))
    if output.exists():raise ValueError('output must be a fresh directory')
    lock=json.loads((ROOT/'llm-inputs.json').read_text())
    conversion=json.loads((ROOT/'evidence/offline-llm-model-conversion.json').read_text())
    expected=conversion['outputs']['tinyllama-q4_0.gguf']
    if model.stat().st_size!=expected['bytes'] or sha(model)!=expected['sha256']:raise ValueError('model differs from qualified Q4_0 conversion')
    cpu=json.loads((cpu_build/'receipt.json').read_text())
    if cpu['classification']!='host-cpu-reference-tools-built':raise ValueError('wrong CPU build classification')
    flags=cpu['identity']['flags']
    for required in ['-DGGML_BACKEND_DL=OFF','-DGGML_CPU=ON','-DGGML_HIP=OFF','-DGGML_CUDA=OFF','-DGGML_BLAS=OFF']:
        if required not in flags:raise ValueError('unsupported CPU backend configuration')
    compiler=Path('/usr/bin/clang++-18')
    version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    if 'version 18.1.8' not in version:raise ValueError('LLVM18.1.8 required')
    source_names=['tools/build_llm_rmsnorm_reference.py','tests/llm_rmsnorm_reference.cpp','tests/llm_rmsnorm_reference_test.py',
                  'llm-inputs.json','evidence/offline-llm-model-conversion.json','tests/fixtures/llm-reference-prompts.json']
    identities={str(ROOT/name):sha(ROOT/name) for name in source_names}
    for p in [cpu_build/'receipt.json',compiler,Path(sys.executable)]:identities[str(p)]=sha(p)
    def check_source():
        revision=subprocess.check_output(['git','-C',CLIENT,'rev-parse','HEAD'],text=True).strip()
        if revision!=lock['client']['revision'] or revision!=cpu['source_revision'] or subprocess.check_output(['git','-C',CLIENT,'status','--porcelain'],text=True).strip():
            raise ValueError('requires clean locked upstream source and matching CPU build')
        for name,digest in cpu['artifacts'].items():
            path=cpu_build/name
            if sha(path)!=digest:raise ValueError('CPU artifact changed: '+name)
            identities.setdefault(str(path),digest)
    check_source()
    profile=json.loads((ROOT/'tests/fixtures/llm-reference-prompts.json').read_text())
    if profile['model_revision']!=conversion['model_revision']:raise ValueError('token profile model mismatch')
    tokens=list(dict.fromkeys(t for c in profile['cases'] for t in c['tokens']))[:17]
    if len(tokens)!=17 or any(type(t)!=int or not 0<=t<32000 for t in tokens):raise ValueError('need17 valid fixed tokens')
    output.mkdir(parents=True);commands=[]
    manifest={'width':WIDTH,'epsilon_f32':EPSILON,'row_counts':[1,2,17],'embedding_tokens':tokens,'threads':1,
              'operator':'GGML_OP_RMS_NORM unfused F32','tolerance':TOLERANCE}
    manifest_path=output/'declared-profile.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');manifest_sha=sha(manifest_path)
    token_file=output/'tokens.txt';token_file.write_text(' '.join(map(str,tokens))+'\n');token_sha=sha(token_file)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    def run(label,argv):
        record=dict(label=label,argv=list(map(str,argv)));commands.append(record)
        try:
            p=subprocess.run(record['argv'],capture_output=True,text=True,timeout=120,env=env)
            record['exit_code']=p.returncode
            (output/(label+'.stdout')).write_text(p.stdout);(output/(label+'.stderr')).write_text(p.stderr)
            if p.returncode:raise RuntimeError(label+': '+p.stderr[-3000:])
            return p.stdout
        except subprocess.TimeoutExpired as error:
            record['timed_out']=True
            for suffix,value in [('stdout',error.stdout),('stderr',error.stderr)]:
                (output/(label+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
            raise
        finally:(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    binary=output/'fixture';libraries=cpu_build/'cmake/bin'
    args=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror',ROOT/'tests/llm_rmsnorm_reference.cpp',
          '-I'+str(CLIENT/'ggml/include')]
    dep=output/'fixture.d';run('dependencies',args+['-M','-MF',dep])
    for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
        p=Path(name).resolve();identities[str(p)]=sha(p)
    run('compile',args+['-L'+str(libraries),'-Wl,-rpath,'+str(libraries),'-lggml','-lggml-cpu','-lggml-base','-o',binary])
    binary_sha=sha(binary)
    loaded=run('runtime-libraries',['ldd',binary])
    for line in loaded.splitlines():
        parts=line.strip().split();path=parts[2] if len(parts)>2 and parts[1]=='=>' else parts[0] if parts else ''
        if path.startswith('/') and Path(path).is_file():identities[path]=sha(path)
    for repeat in [1,2]:
        directory=output/('run'+str(repeat));directory.mkdir()
        text=run('run'+str(repeat),[binary,model,token_file,directory])
        if text.strip()!='GGML_RMS_NORM_CPU_PASS cases=6 width=2048 threads=1 gpu_execution=false':raise ValueError('unexpected CPU fixture result')
    cases=[]
    for family in ['stress','embedding']:
        for rows in [1,2,17]:
            name=family+'-'+str(rows);inputs=output/'run1'/(name+'.input.f32le');cpu_path=output/'run1'/(name+'.cpu.f32le')
            for suffix in ['input.f32le','cpu.f32le']:
                if (output/'run1'/(name+'.'+suffix)).read_bytes()!=(output/'run2'/(name+'.'+suffix)).read_bytes():raise ValueError('fresh CPU repeats differ')
            values=read_f32(inputs,rows*WIDTH);actual=read_f32(cpu_path,rows*WIDTH);reference=fp64_reference(values)
            metrics=compare(actual,reference)
            if metrics['maximum_budget_fraction']>1/32:raise ValueError('CPU baseline consumes too much proposed budget')
            fp64=output/(name+'.fp64le');fp64.write_bytes(struct.pack('<'+'d'*len(reference),*reference))
            cases.append(dict(name=name,rows=rows,input_sha256=sha(inputs),cpu_sha256=sha(cpu_path),fp64_sha256=sha(fp64),metrics=metrics))
    actual=read_f32(output/'run1/stress-17.cpu.f32le',17*WIDTH)
    controls={'noop':read_f32(output/'run1/stress-17.input.f32le',17*WIDTH),'corrupt':list(actual)};controls['corrupt'][0]+=1
    for label,values in controls.items():
        try:compare(values,actual)
        except ValueError as error:(output/(label+'-rejection.txt')).write_text(str(error)+'\n')
        else:raise ValueError('negative oracle unexpectedly accepted '+label)
    check_source()
    if sha(model)!=expected['sha256'] or sha(binary)!=binary_sha or sha(manifest_path)!=manifest_sha or sha(token_file)!=token_sha:raise ValueError('model, executable or frozen profile changed')
    for name,digest in identities.items():
        if sha(name)!=digest:raise ValueError('input changed: '+name)
    result=dict(schema=1,classification='host-cpu-rmsnorm-reference',source_revision=cpu['source_revision'],model_sha256=expected['sha256'],
                compiler_version=version,identities=identities,profile=manifest,cases=cases,
                artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                fresh_repeats_bitwise_equal=True,negative_controls=['noop rejected','single-element corruption rejected'],gpu_executed=False,
                limits=['CPU operator reference only; no native numerical accuracy, tolerance qualification or model inference claim.',
                        'Proposed tolerance is frozen before GPU evaluation; future rejection must be investigated, not silently relaxed.',
                        'Contiguous rows only; strided input, fused norm weights and nonfinite handling remain separate cases.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path,required=True);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.output,args.cpu_build,args.model)
