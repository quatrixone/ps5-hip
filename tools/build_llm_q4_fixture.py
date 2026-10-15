"""Source-only selected upstream Q8_1 preparation and nonfused Q4_0 MMVQ objects.

Selection removes unreachable kernels, not instructions from the selected kernel.
This is an offline operator fixture, not runtime admission or GPU execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from qualify_public_hip_rmsnorm import fixture_inputs
from probe_hip_routes import audit_object
from build_llm_rmsnorm_fixture import CONSTANTS
from fetch_locked_deps import source, read, sources

ROOT=Path(__file__).resolve().parents[1]
Q8='_ZL13quantize_q8_1PKfPvlllllj15HIP_vector_typeIjLj3EE'
Q4='_ZL13mul_mat_vec_qIL9ggml_type2ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj'
PROFILES={
 'q8':dict(source='quantize.cu',symbol=Q8,offsets=[0,8,16,24,32,40,48,56,60],sizes=[8,8,8,8,8,8,8,4,12],pointers=[0,1],explicit=72,implicit=256,static=0,threads=256,rsrc1=0x60af00c1,rsrc2=0x38c),
 'q4':dict(source='mmvq.cu',symbol=Q4,offsets=[0,8,16,24,104,112,116,128,132,136,140,152,156,160,164,176,180,184,188],sizes=[8,8,8,80,8,4,12,4,4,4,12,4,4,4,12,4,4,4,4],pointers=[0,1,2,4],explicit=192,implicit=0,static=384,threads=128,rsrc1=0x60af00c3,rsrc2=0xb8c)}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def bind(inputs,path):
    path=str(Path(path).resolve());digest=sha(path)
    if path in inputs and inputs[path]!=digest:raise ValueError('input changed: '+path)
    inputs[path]=digest

def canonical_function(ir,symbol):
    matches=re.findall(r'(?m)^define [^\n]*@'+re.escape(symbol)+r'\([^\n]*\) [^\n]*\{\n.*?^\}',ir,re.S|re.M)
    if len(matches)!=1:raise ValueError('exact selected function missing')
    text=matches[0]
    attrs=dict(re.findall(r'(?m)^attributes #(\d+) = (.*)$',ir))
    text=re.sub(r'#(\d+)',lambda m:attrs[m[1]],text)
    metadata=dict(re.findall(r'(?m)^!(\d+) = (.*)$',ir))
    ids={};pending=[]
    def number(m):
        key=m[1]
        if key not in metadata:raise ValueError('missing IR metadata')
        if key not in ids:ids[key]=len(ids);pending.append(key)
        return '!'+str(ids[key])
    text=re.sub(r'!(\d+)',number,text)
    rows=[]
    for key in pending:rows.append(re.sub(r'!(\d+)',number,metadata[key]))
    return text+'\n'+'\n'.join(rows)

def preserved_function(original,selected,symbol):
    names=re.findall(r'(?m)^define amdgpu_kernel void @(\w+)\(',selected)
    if names!=[symbol]:raise ValueError('selected kernel inventory changed')
    before=canonical_function(original,symbol);after=canonical_function(selected,symbol)
    if before!=after:raise ValueError('selected kernel IR changed')
    return hashlib.sha256(before.encode()).hexdigest()

def preserved_data(original,selected,symbol):
    original_types=dict(re.findall(r'(?m)^(%[^\n]+?) = type (.*)$',original))
    selected_types=dict(re.findall(r'(?m)^(%[^\n]+?) = type (.*)$',selected))
    if any(original_types.get(name)!=value for name,value in selected_types.items()):
        raise ValueError('retained named type changed')
    original_globals=dict(re.findall(r'(?m)^(@[^\n]+?) = (.*)$',original))
    selected_globals=dict(re.findall(r'(?m)^(@[^\n]+?) = (.*)$',selected))
    allowed='@_ZZ'+symbol[2:]+'E10tmp_shared'
    transforms=[]
    for name,value in selected_globals.items():
        old=original_globals.get(name)
        if old==value:continue
        before='internal unnamed_addr addrspace(3) global [3 x [1 x [1 x [32 x float]]]] undef, align 16'
        after='external hidden unnamed_addr addrspace(3) global [3 x [1 x [1 x [32 x float]]]], align 16'
        if symbol!=Q4 or name!=allowed or old!=before or value!=after:
            raise ValueError('retained global changed')
        transforms.append(dict(name=name,before=before,after=after,reason='llvm-extract externalizes uninitialized LDS declaration; identical384-byte addrspace3 type and16-byte alignment, linked descriptor independently checked'))
    if symbol==Q4 and set(selected_globals)!={allowed}:raise ValueError('selected LDS global missing')
    return dict(named_types=selected_types,globals=selected_globals,allowed_lds_transformations=transforms)


def preserved_definitions(original,selected,symbol):
    kernel=preserved_function(original,selected,symbol)
    names=re.findall(r'(?m)^define [^\n]*?@([A-Za-z0-9_.$]+)\(',selected)
    if len(names)!=len(re.findall(r'(?m)^define ',selected)) or len(names)!=len(set(names)):
        raise ValueError('unsupported retained function definition')
    digests={}
    for name in names:
        before=canonical_function(original,name);after=canonical_function(selected,name)
        if before!=after:raise ValueError('retained helper IR changed: '+name)
        digests[name]=hashlib.sha256(before.encode()).hexdigest()
    for field in ('target triple','target datalayout'):
        before=re.findall(r'(?m)^'+field+r' = (.*)$',original)
        after=re.findall(r'(?m)^'+field+r' = (.*)$',selected)
        if len(before)!=1 or before!=after:raise ValueError('IR target changed')
    if '"amdgcn-amd-amdhsa"' not in selected:raise ValueError('wrong IR target')
    return kernel,digests


def metadata_contract(notes,profile):
    if 'amdhsa.target:   amdgcn-amd-amdhsa--gfx1013' not in notes:raise ValueError('wrong metadata target')
    if re.findall(r'(?m)^    \.name:\s+(\S+)',notes)!=[profile['symbol']]:raise ValueError('module kernel inventory changed')
    chunks=re.split(r'(?m)^      - ',notes.split('  - .args:',1)[1].split('    .group_segment_fixed_size:',1)[0])[1:]
    arguments=[]
    for chunk in chunks:
        row={}
        for key in ('offset','size','value_kind'):
            match=re.search(r'(?m)^\s*\.'+key+r':\s+(\S+)',chunk)
            if not match:raise ValueError('argument metadata missing')
            row[key]=match[1] if key=='value_kind' else int(match[1])
        arguments.append(row)
    expected=[dict(offset=o,size=s,value_kind='global_buffer' if i in profile['pointers'] else 'by_value') for i,(o,s) in enumerate(zip(profile['offsets'],profile['sizes']))]
    base=profile['explicit']
    hidden=[(0,4,'block_count_x'),(4,4,'block_count_y'),(8,4,'block_count_z'),(12,2,'group_size_x'),(14,2,'group_size_y'),(16,2,'group_size_z'),(18,2,'remainder_x'),(20,2,'remainder_y'),(22,2,'remainder_z'),(40,8,'global_offset_x'),(48,8,'global_offset_y'),(56,8,'global_offset_z'),(64,2,'grid_dims')]
    if profile['implicit']:expected.extend(dict(offset=base+o,size=s,value_kind='hidden_'+k) for o,s,k in hidden)
    if arguments!=expected:raise ValueError('complete argument ABI changed')
    resources=dict(kernarg_segment_size=base+profile['implicit'],kernarg_segment_align=8,group_segment_fixed_size=profile['static'],private_segment_fixed_size=0,max_flat_workgroup_size=profile['threads'],wavefront_size=32)
    for key,value in resources.items():
        match=re.search(r'(?m)^    \.'+key+r':\s+(\d+)',notes)
        if not match or int(match[1])!=value:raise ValueError('resource changed: '+key)
    return dict(arguments=arguments,resources=resources)

def descriptor_contract(descriptor,profile):
    expected=dict(group_bytes=profile['static'],private_bytes=0,kernarg_bytes=profile['explicit']+profile['implicit'],rsrc1=profile['rsrc1'],rsrc2=profile['rsrc2'],rsrc3=0,properties=0x409,preload=0)
    if descriptor!=expected:raise ValueError('descriptor changed: '+repr(descriptor))

def build(prerequisite,output):
    prerequisite=Path(prerequisite).resolve();output=Path(output).resolve()
    if not any(output.is_relative_to(ROOT/p) for p in ('build','results')):raise ValueError('output must stay below build/ or results/')
    def locked_sources():
        source("llama-client",read("llm-inputs.json")["client"],False)
        for name in ("hip-headers","clr"):source(name,sources()[name],False)
    locked_sources()
    original_inputs=fixture_inputs(prerequisite);inputs=dict(original_inputs)
    for name in ('tools/build_llm_q4_fixture.py','tests/llm_q4_fixture_test.py','tools/qualify_public_hip_rmsnorm.py'):
        bind(inputs,ROOT/name)
    bind(inputs,sys.executable)
    tools={}
    for name in ('clang++-18','llvm-config-18','llvm-dis-18','llvm-extract-18','llc-18','ld.lld-18','llvm-nm-18','llvm-readobj-18','llvm-objdump-18'):
        path=shutil.which(name)
        if not path:raise ValueError('missing tool: '+name)
        version=subprocess.check_output([path,'--version'],text=True).strip()
        if '18.1.8' not in version:raise ValueError('LLVM18.1.8 required: '+name)
        bind(inputs,path);tools[name]=dict(path=str(Path(path).resolve()),sha256=sha(path),version=version)
    output.mkdir(parents=True,exist_ok=False);commands=[]
    def run(label,argv,timeout=600):
        argv=list(map(str,argv));record=dict(label=label,argv=argv);commands.append(record)
        try:
            result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
            record['exit_code']=result.returncode
            (output/(label+'.stdout')).write_text(result.stdout);(output/(label+'.stderr')).write_text(result.stderr)
            if result.returncode:raise RuntimeError(label+': '+result.stderr[-2500:])
            return result.stdout
        except subprocess.TimeoutExpired as error:
            record['timed_out']=True
            for suffix,value in (('stdout',error.stdout),('stderr',error.stderr)):(output/(label+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
            raise
        finally:(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    bindir=run('llvm-bindir',['llvm-config-18','--bindir']).strip()
    flags=['clang++-18','-x','hip','-nogpuinc','--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-DNDEBUG','-O2','-fPIC','-DGGML_USE_HIP','-DGGML_HIP_NO_VMM','-DGGML_HIP_NO_MMQ_MFMA','-DGGML_CUDA_NO_FA']
    client=ROOT/'.deps/sources/llama-client'
    flags+=['-I'+str(p) for p in (ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',ROOT/'ports/hip',client/'ggml/include',client/'ggml/src',prerequisite/'diagnostic-hipblas')]
    flags+=['-include','__clang_hip_runtime_wrapper.h','--cuda-device-only','--no-gpu-bundle-output','-B'+bindir,'--rocm-device-lib-path='+str(prerequisite/'libs')]
    order=['ockl-workitem-only','mul_hi','ocml',*(n for n in CONSTANTS if n!='oclc_isa_version_1030')]
    flags+=['--hip-device-lib='+n+'.bc' for n in order]
    profiles={}
    for name,profile in PROFILES.items():
        print('Building '+name+' unchanged '+profile['source'],flush=True)
        tu_source=client/'ggml/src/ggml-cuda'/profile['source'];bind(inputs,tu_source)
        run(name+'-dependencies',flags+[tu_source,'-M','-MF',output/(name+'.d')])
        for path in shlex.split((output/(name+'.d')).read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):bind(inputs,path)
        run(name+'-bitcode',flags+[tu_source,'-emit-llvm','-c','-o',output/(name+'-full.bc')])
        run(name+'-ir',['llvm-dis-18',output/(name+'-full.bc'),'-o',output/(name+'-full.ll')])
        run(name+'-extract',['llvm-extract-18','--recursive','--func='+profile['symbol'],output/(name+'-full.bc'),'-o',output/(name+'.bc')])
        run(name+'-selected-ir',['llvm-dis-18',output/(name+'.bc'),'-o',output/(name+'.ll')])
        data_preservation=preserved_data((output/(name+'-full.ll')).read_text(),(output/(name+'.ll')).read_text(),profile['symbol'])
        body,definitions=preserved_definitions((output/(name+'-full.ll')).read_text(),(output/(name+'.ll')).read_text(),profile['symbol'])
        run(name+'-object',['llc-18','-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx1013','-filetype=obj',output/(name+'.bc'),'-o',output/(name+'.o')])
        run(name+'-link',['ld.lld-18','-shared',output/(name+'.o'),'-o',output/(name+'.hsaco')])
        if run(name+'-undefined',['llvm-nm-18','--undefined-only',output/(name+'.hsaco')]).strip():raise ValueError('unresolved selected device imports')
        notes=run(name+'-notes',['llvm-readobj-18','--notes',output/(name+'.hsaco')]);metadata=metadata_contract(notes,profile)
        isa=run(name+'-isa',['llvm-objdump-18','--disassemble','--mcpu=gfx1013',output/(name+'.hsaco')])
        data=(output/(name+'.hsaco')).read_bytes()
        if data[:9]!=b'\x7fELF\x02\x01\x01\x40\x03':raise ValueError('ELF64 AMDGPU code object V5 required')
        audit=audit_object(data,notes,isa,[profile['symbol']]);descriptor=audit['descriptors'][profile['symbol']];descriptor_contract(descriptor,profile)
        profiles[name]=dict(symbol=profile['symbol'],selected_ir_sha256=body,retained_definitions=definitions,data_preservation=data_preservation,body_preserved=True,metadata=metadata,descriptor=descriptor,isa=audit['isa'],undefined_symbols=[])
    locked_sources()
    if fixture_inputs(prerequisite)!=original_inputs:raise ValueError('prerequisite changed')
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('build input changed: '+path)
    receipt=dict(schema=1,status='PASS',classification='source-only-selected-upstream-q4-q8-fixture',prerequisite=str(prerequisite/'receipt.json'),inputs=inputs,tools=tools,profiles=profiles,library_order=order,commands=commands,native_executed=False,gpu_executed=False,artifacts={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()},limits=['Selected operator compile/link proof only; no runtime admission, numerical execution or inference claim.','Unchanged upstream translation units, recursive LLVM extraction; unreachable unrelated kernels excluded. Selected and retained helper bodies, attributes, metadata and named types unchanged; only exact uninitialized384-byte LDS external declaration transformation allowed and recorded.','HIP6.2 diagnostic hipBLAS headers inherited from authenticated prerequisite; modern header integration remains separate.','Q4 selection is nonfused with null nested fusion pointers; fused gate/up lifetime unqualified.'])
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS '+str(output/'receipt.json'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rmsnorm-fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.rmsnorm_fixture,args.output)
