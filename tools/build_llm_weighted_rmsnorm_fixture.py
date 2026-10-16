"""Authenticated source-only selected weighted RMSNorm; no GPU execution claim."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import re
import shlex
import struct
from build_llm_q4_fixture import ROOT, sha, bind, descriptor_contract, preserved_definitions, preserved_data
from probe_hip_routes import audit_object
from qualify_public_hip_rmsnorm import checked_inputs, checked_tools
from build_llm_rmsnorm_fixture import SYMBOL as ORIGINAL_SYMBOL, selected_metadata, CONSTANTS
from fetch_locked_deps import source, read, sources

CLASSIFICATION = 'source-only-selected-upstream-weighted-rmsnorm-fixture'
SYMBOL = ORIGINAL_SYMBOL.replace('ILi1024ELb0ELb0ELb0E','ILi1024ELb1ELb0ELb0E')
PROFILE = dict(static=0,explicit=232,implicit=256,rsrc1=0x60af0184,rsrc2=0x38c)
SOURCES = ('tools/build_llm_weighted_rmsnorm_fixture.py','tests/llm_weighted_rmsnorm_fixture_test.py',
           'tools/build_llm_q4_fixture.py','tools/build_llm_rmsnorm_fixture.py',
           'tools/qualify_public_hip_rmsnorm.py','tools/probe_hip_routes.py')
TOOLS = ('clang++-18','llvm-config-18','llvm-dis-18','llvm-extract-18','llc-18','ld.lld-18',
         'llvm-readobj-18','llvm-objdump-18','llvm-nm-18')
LABELS = ('dependencies','bitcode','full-ir','extract','ir','object','link','notes','isa','undefined')
REQUIRED = ('weighted.hsaco','weighted.ll','norm-full.ll','norm-full.bc','weighted.bc','weighted.o',
            'norm.d','weighted-notes.stdout','weighted-isa.stdout','weighted-undefined.stdout','commands.json')


def original_inputs(folder):
    folder=Path(folder).resolve();report=json.loads((folder/'receipt.json').read_text())
    if report.get('classification') is not None or report.get('selected',{}).get('symbol') != ORIGINAL_SYMBOL:
        raise ValueError('original unweighted RMSNorm prerequisite required')
    from qualify_public_hip_rmsnorm import fixture_inputs as validate_original
    return validate_original(folder)


def metadata_contract(notes):
    if re.findall(r'(?m)^amdhsa.target:\s+(\S+)',notes)!=['amdgcn-amd-amdhsa--gfx1013']:
        raise ValueError('wrong metadata target')
    if re.findall(r'(?m)^    \.name:\s+(\S+)',notes) != [SYMBOL]:
        raise ValueError('selected kernel inventory changed')
    # Reuse the exact40-argument ABI parser; projection does not alter the artifact.
    metadata=selected_metadata(notes.replace(SYMBOL,ORIGINAL_SYMBOL));metadata['symbol']=SYMBOL
    return metadata


def object_header(data):
    if (len(data)<64 or data[:9]!=b'\x7fELF\x02\x01\x01\x40\x03' or
            struct.unpack_from('<H',data,18)[0]!=224 or struct.unpack_from('<I',data,48)[0]!=0x142):
        raise ValueError('AMDGPU V5 gfx1013 header required')


def contracts(folder):
    folder=Path(folder);notes=(folder/'weighted-notes.stdout').read_text()
    metadata=metadata_contract(notes)
    isa=(folder/'weighted-isa.stdout').read_text()
    data=(folder/'weighted.hsaco').read_bytes()
    object_header(data)
    audit=audit_object(data,notes,isa,[SYMBOL]);descriptor=audit['descriptors'][SYMBOL]
    descriptor_contract(descriptor,PROFILE)
    if (folder/'weighted-undefined.stdout').read_text().strip():raise ValueError('undefined selected symbols')
    full=(folder/'norm-full.ll').read_text();selected=(folder/'weighted.ll').read_text()
    body,definitions=preserved_definitions(full,selected,SYMBOL)
    globals=preserved_data(full,selected,SYMBOL)
    return dict(metadata=metadata,descriptor=descriptor,isa=audit['isa'],undefined_symbols=[],
                body_sha256=body,retained_definitions=definitions,data_preservation=globals)


def fixture_inputs(folder):
    folder=Path(folder).resolve();receipt=folder/'receipt.json';report=json.loads(receipt.read_text())
    if (report.get('classification')!=CLASSIFICATION or report.get('status')!='PASS' or
            report.get('gpu_executed') is not False or report.get('native_executed') is not False):
        raise ValueError('unqualified weighted fixture')
    original=Path(report['original_fixture']).resolve()
    if original==folder:raise ValueError('recursive prerequisite')
    parent=original_inputs(original)
    inputs=checked_inputs(report,'inputs',[ROOT/name for name in SOURCES])
    if any(inputs.get(n)!=d for n,d in parent.items()):raise ValueError('prerequisite closure missing')
    for n,d in checked_tools(report,'tools',TOOLS).items():
        if inputs.get(n)!=d:raise ValueError('tool closure missing')
    artifacts=report.get('artifacts',{})
    if not set(REQUIRED)<=artifacts.keys():raise ValueError('required weighted artifact missing')
    for name,row in artifacts.items():
        path=(folder/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder) or path==receipt:raise ValueError('invalid artifact path')
        if sha(path)!=row.get('sha256') or path.stat().st_size!=row.get('bytes'):raise ValueError('weighted artifact changed')
        if str(path) in inputs and inputs[str(path)]!=row['sha256']:raise ValueError('identity collision')
        inputs[str(path)]=row['sha256']
    for n in shlex.split((folder/'norm.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):
        if str(Path(n).resolve()) not in inputs:raise ValueError('compiler dependency missing')
    if report.get('profile')!=contracts(folder):raise ValueError('weighted profile changed')
    commands=report.get('commands',[])
    if [c.get('label') for c in commands]!=list(LABELS) or any(c.get('exit_code')!=0 for c in commands):raise ValueError('incomplete commands')
    if json.loads((folder/'commands.json').read_text())!=commands:raise ValueError('command record differs')
    inputs[str(receipt)]=sha(receipt)
    return inputs


def discover_tool(name):
    found = shutil.which(name)
    if not found: raise ValueError('missing tool: '+name)
    # Preserve argv[0]: the multicall lld driver selects its flavour by this name.
    path = Path(found).absolute()
    version = subprocess.check_output([str(path),'--version'],text=True)
    if '18.1.8' not in version: raise ValueError('wrong tool version')
    return dict(path=str(path),sha256=sha(path),version=version)

def build(prerequisite, output):
    prerequisite,output=Path(prerequisite).resolve(),Path(output).resolve()
    if not any(output.is_relative_to(ROOT/name) for name in ('build','results')):raise ValueError('output outside build/results')
    def locked_sources():
        source('llama-client',read('llm-inputs.json')['client'],False)
        for name in ('hip-headers','clr'):source(name,sources()[name],False)
    locked_sources();parent=original_inputs(prerequisite);inputs=dict(parent)
    for name in SOURCES:bind(inputs,ROOT/name)
    bind(inputs,sys.executable);tools={name:discover_tool(name) for name in TOOLS}
    for row in tools.values():bind(inputs,row['path'])
    output.mkdir(parents=True,exist_ok=False);commands=[]
    def run(label,argv):
        argv=list(map(str,argv));record=dict(label=label,argv=argv);commands.append(record)
        try:
            result=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=180);record['exit_code']=result.returncode
            for suffix,data in [('stdout',result.stdout),('stderr',result.stderr)]:(output/('weighted-'+label+'.'+suffix)).write_bytes(data)
            if result.returncode:raise RuntimeError(label+' failed; logs retained')
            return result.stdout.decode()
        except subprocess.TimeoutExpired as error:
            record['timed_out']=True
            for suffix,data in [('stdout',error.stdout),('stderr',error.stderr)]:(output/('weighted-'+label+'.'+suffix)).write_bytes(data or b'')
            raise
        finally:(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    client=ROOT/'.deps/sources/llama-client';tu=client/'ggml/src/ggml-cuda/norm.cu'
    bindir=subprocess.check_output([tools['llvm-config-18']['path'],'--bindir'],text=True).strip()
    flags=[tools['clang++-18']['path'],'-x','hip','-nogpuinc','--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-DNDEBUG','-O2','-fPIC','-DGGML_USE_HIP','-DGGML_HIP_NO_VMM','-DGGML_HIP_NO_MMQ_MFMA','-DGGML_CUDA_NO_FA']
    flags+=['-I'+str(p) for p in (ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',ROOT/'ports/hip',client/'ggml/include',client/'ggml/src',prerequisite/'diagnostic-hipblas')]
    flags+=['-include','__clang_hip_runtime_wrapper.h','--cuda-device-only','--no-gpu-bundle-output','-B'+bindir,'--rocm-device-lib-path='+str(prerequisite/'libs')]
    order=['ockl-workitem-only','mul_hi','ocml',*(n for n in CONSTANTS if n!='oclc_isa_version_1030')]
    flags+=['--hip-device-lib='+n+'.bc' for n in order]
    run('dependencies',flags+[tu,'-M','-MF',output/'norm.d'])
    for n in shlex.split((output/'norm.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]):bind(inputs,n)
    run('bitcode',flags+[tu,'-emit-llvm','-c','-o',output/'norm-full.bc'])
    run('full-ir',[tools['llvm-dis-18']['path'],output/'norm-full.bc','-o',output/'norm-full.ll'])
    run('extract',[tools['llvm-extract-18']['path'],'--recursive','--func='+SYMBOL,output/'norm-full.bc','-o',output/'weighted.bc'])
    run('ir',[tools['llvm-dis-18']['path'],output/'weighted.bc','-o',output/'weighted.ll'])
    run('object',[tools['llc-18']['path'],'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx1013','-filetype=obj',output/'weighted.bc','-o',output/'weighted.o'])
    run('link',[tools['ld.lld-18']['path'],'-shared',output/'weighted.o','-o',output/'weighted.hsaco'])
    run('notes',[tools['llvm-readobj-18']['path'],'--notes',output/'weighted.hsaco'])
    run('isa',[tools['llvm-objdump-18']['path'],'--disassemble','--mcpu=gfx1013',output/'weighted.hsaco'])
    run('undefined',[tools['llvm-nm-18']['path'],'--undefined-only',output/'weighted.hsaco'])
    profile=contracts(output);locked_sources()
    if original_inputs(prerequisite)!=parent:raise ValueError('prerequisite changed')
    for n,d in inputs.items():
        if sha(n)!=d:raise ValueError('input changed: '+n)
    report=dict(classification=CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,
        original_fixture=str(prerequisite),inputs=inputs,tools=tools,commands=commands,profile=profile,
        artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()},
        limits=['Selected upstream body/helpers preserved; no runtime or numerical GPU qualification.'])
    (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    fixture_inputs(output);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rmsnorm-fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.rmsnorm_fixture,args.output)
    print('PASS selected weighted RMSNorm fixture: '+str(args.output))
