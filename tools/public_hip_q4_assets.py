"""Validate the fixed first Q4 operator corpus before packaging; never runs GPU code.

Heavy authenticated dependency helpers are imported inside assets(), so the
native builder can call this module without an import cycle.
"""
import hashlib
import json
from pathlib import Path
import shlex
import struct

ROOT = Path(__file__).resolve().parents[1]
Q4_CASES = [("attn-k-"+str(n),2048,n) for n in (1,2,17,256)] + [("ffn-down-"+str(n),5632,n) for n in (1,17)] + [("synthetic-zero",2048,1),("synthetic-pattern",2048,1)]
Q8_CASES = [("q8-zero",256),("q8-constant",256),("q8-signed-ties",256),("q8-padding",255)]
FIELDS = ("input.f32le","weights.q4_0","q8.bin","fp64le","lane.f32le","budget.f64le")
# Offsets are specific to the authenticated tinyllama-q4_0.gguf identity checked
# by verify_model and its original CPU/model source closure, NOT arbitrary GGUFs.
WEIGHTS = {"attn-k":(74473728,2048//32*18*256),"ffn-down":(79790336,5632//32*18*17)}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(value, message):
    if not value:raise ValueError(message)


def pack(values,code):
    return struct.pack('<'+code*len(values),*values)


def file_bytes(folder,name,size=None):
    path=folder/name
    require(not Path(name).is_absolute() and path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(), 'unsafe asset path')
    value=path.read_bytes()
    require(size is None or len(value)==size,'asset byte count differs: '+name)
    return value


def declared_profile(oracle):
    return dict(schema=1,oracle=oracle.PROFILE,cases=[dict(name=n,width=w,rows=r,input='authenticated RMSNorm embedding output row0 token1' if w==2048 else 'deterministic synthetic; not a model activation trace') for n,w,r in Q4_CASES[:6]],native_executed=False,
                synthetic_cases=['synthetic-zero','synthetic-pattern'],q8_cases=[dict(name=n,input_width=w,padded_width=256) for n,w in Q8_CASES])


def validate_corpus(folder,report,embedding,weights,oracle):
    """Independently regenerate every packaged tensor; also used by unit tests."""
    require(report.get('profile')==declared_profile(oracle),'declared numerical profile differs')
    require(json.loads(file_bytes(folder,'declared-profile.json'))==report['profile'],'profile artifact differs')
    require([(c.get('name'),c.get('width'),c.get('rows')) for c in report.get('cases',[])]==Q4_CASES,'Q4 case inventory differs')
    require([(c.get('name'),c.get('input_width'),c.get('padded_width')) for c in report.get('q8_cases',[])]==[(n,w,256) for n,w in Q8_CASES],'Q8 case inventory differs')
    output={}
    for row,(name,width,count) in zip(report['cases'],Q4_CASES):
        if name.startswith('attn-k'):
            activation=embedding;weight=weights['attn-k'][:count*width//32*18]
        elif name.startswith('ffn-down'):
            activation=pack([((i*73)%257-128)/64 for i in range(width)],'f');weight=weights['ffn-down'][:count*width//32*18]
        else:
            activation=pack([0.0]*width if name=='synthetic-zero' else [((i%32)-16)/8 for i in range(width)],'f')
            weight=(struct.pack('<e',0.125)+bytes(i|((15-i)<<4) for i in range(16)))*(width//32)
        q8=oracle.quantize_q8_1(activation)
        numerical=oracle.reference_q4(weight,q8,width,count)
        expected=dict(zip(FIELDS,[activation,weight,q8,pack(numerical['fp64'],'d'),pack(numerical['lane_f32'],'f'),pack(numerical['budget'],'d')]))
        require(set(row.get('artifacts',{}))==set(FIELDS),'case asset inventory differs')
        for field,data in expected.items():
            relative=name+'/'+field;actual=file_bytes(folder,relative,len(data));require(actual==data,'recomputed asset differs: '+relative)
            require(row['artifacts'][field]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),'case identity differs')
            output['q4/'+relative]=str((folder/relative).resolve())
    q8values={'q8-zero':[0.0]*256,'q8-constant':[1.0]*256,'q8-signed-ties':[127.0,-127.0,0.5,-0.5]*64,'q8-padding':[((i%31)-15)/8 for i in range(255)]}
    for row,(name,width) in zip(report['q8_cases'],Q8_CASES):
        raw=pack(q8values[name],'f');q8=oracle.quantize_q8(q8values[name],padded_width=256)
        for field,data in [('input.f32le',raw),('q8.bin',q8)]:
            relative=name+'/'+field;require(file_bytes(folder,relative,len(data))==data,'Q8 stress asset differs')
            key='input_sha256' if field=='input.f32le' else 'q8_sha256';require(row.get(key)==hashlib.sha256(data).hexdigest(),'Q8 stress identity differs')
            output['q4/'+relative]=str((folder/relative).resolve())
    return output


def assets(fixture,reference):
    self_inputs={str(ROOT/n):sha(ROOT/n) for n in ('tools/public_hip_q4_assets.py','tests/public_hip_q4_assets_test.py')}
    from qualify_public_hip_q4 import fixture_inputs
    from build_public_hip import rmsnorm_reference_inputs
    from build_llm_q4_reference import verify_model
    import llm_q4_reference as oracle
    fixture,reference=Path(fixture).resolve(),Path(reference).resolve()
    modules=fixture_inputs(fixture)
    receipt=reference/'receipt.json';report=json.loads(receipt.read_text());receipt_sha=sha(receipt)
    require(report.get('status')=='PASS' and report.get('classification')=='offline-gpu-format-q4-reference' and report.get('native_executed') is False and report.get('gpu_executed') is False and report.get('weights_repeated_bitwise_equal') is True,'unqualified numerical reference')
    inputs=report.get('inputs');require(isinstance(inputs,dict) and inputs,'missing numerical input closure')
    required=['tools/build_llm_q4_reference.py','tests/llm_q4_reference_build_test.py','tests/llm_q4_weights.cpp','tools/llm_q4_reference.py','tests/llm_q4_reference_test.py','tools/build_public_hip.py','tools/qualify_public_hip_q4.py']
    require(all(str(ROOT/n) in inputs for n in required),'required numerical source identity missing')
    for n,d in inputs.items():require(Path(n).is_absolute() and sha(n)==d,'stale numerical input: '+n)
    require(all(inputs.get(n)==d for n,d in modules.items()),'selected module closure missing')
    rms=[]
    for n in inputs:
        if Path(n).name=='receipt.json' and json.loads(Path(n).read_text()).get('classification')=='host-cpu-rmsnorm-reference':rms.append(Path(n).parent)
    require(len(rms)==1,'one original RMSNorm reference required')
    authenticated=rmsnorm_reference_inputs(rms[0])
    require(all(inputs.get(n)==d for n,d in authenticated['inputs'].items()),'original model/CPU closure missing')
    models=[Path(n) for n in inputs if n.endswith('.gguf')];require(len(models)==1,'one authenticated Q4 model required');model=models[0];verify_model(model)
    artifacts=report.get('artifacts');require(isinstance(artifacts,dict) and artifacts,'missing numerical artifacts')
    expected_names={name+'/'+field for name,_,_ in Q4_CASES for field in FIELDS}|{name+'/'+field for name,_ in Q8_CASES for field in ('input.f32le','q8.bin')}
    expected_names|={'commands.json','extractor','extractor.d','declared-profile.json'}
    expected_names|={n+s for n in ('compile','dependencies','runtime-libraries','weights1','weights2') for s in ('.stdout','.stderr')}
    expected_names|={repeat+'/'+name+'.weights.q4_0' for repeat in ('weights1','weights2') for name in WEIGHTS}
    require(set(artifacts)==expected_names,'reference artifact inventory differs')
    for n,row in artifacts.items():
        data=file_bytes(reference,n,row.get('bytes'));require(hashlib.sha256(data).hexdigest()==row.get('sha256'),'reference artifact changed')
    require(set(p.relative_to(reference).as_posix() for p in reference.rglob('*') if p.is_file())==expected_names|{'receipt.json'},'unlisted reference file')
    for n in shlex.split((reference/'extractor.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):require(str(Path(n).resolve()) in inputs,'extractor dependency missing')
    require([c.get('label') for c in report.get('commands',[])]==['dependencies','compile','runtime-libraries','weights1','weights2'] and all(c.get('exit_code')==0 for c in report['commands']),'reference commands incomplete')
    require(report.get('negative_controls')==[name+':'+kind for name,_,_ in Q4_CASES for kind in ('nonfinite','corruption')],'numerical negative controls differ')
    require(json.loads(file_bytes(reference,'commands.json'))==report['commands'],'command record differs')
    weights={}
    with model.open('rb') as f:
        for name,(offset,size) in WEIGHTS.items():
            f.seek(offset);weights[name]=f.read(size);require(len(weights[name])==size,'model extraction short')
            for repeat in ('weights1','weights2'):require(file_bytes(reference,repeat+'/'+name+'.weights.q4_0',size)==weights[name],'raw GGUF weights differ')
    embedding=Path(authenticated['assets']['rmsnorm/embedding-1.cpu.f32le']).read_bytes();require(len(embedding)==8192,'embedding size differs')
    packaged=validate_corpus(reference,report,embedding,weights,oracle)
    for name in ('q8','q4'):packaged['q4/'+name+'.hsaco']=str(fixture/(name+'.hsaco'))
    closure=dict(inputs);closure.update(modules);closure[str(receipt)]=receipt_sha
    for n,row in artifacts.items():closure[str(reference/n)]=row['sha256']
    closure.update(self_inputs)
    # Detect changes during numerical regeneration, including oracle/module inputs.
    for n,d in closure.items():require(sha(n)==d,'input changed during asset validation')
    return dict(module_inputs=modules,reference=dict(receipt=str(receipt),receipt_sha256=receipt_sha),inputs=closure,assets=packaged)
