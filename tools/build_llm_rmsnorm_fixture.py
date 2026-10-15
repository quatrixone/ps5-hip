"""Build unchanged upstream RMSNorm with authenticated source-only prerequisites.

The HIP6.2-matched hipBLAS2.2 headers are diagnostic compatibility inputs,
not a selected library implementation. No host runtime or GPU execution.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request

from fetch_locked_deps import ROOT, source, sources, read
from probe_ocml_math import CONSTANTS, audit_constant
from probe_hip_routes import audit_object

SYMBOL = '_ZL12rms_norm_f32ILi1024ELb0ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f'
OFFSETS = [0,8,16,20,24,32,40,48,56,64,72,80,88,96,108,120,132,144,152,160,168,176,188,200,212,224]
SIZES = [8,8,4,4,4,8,8,8,4,8,8,8,8,12,12,12,12,8,8,8,8,12,12,12,12,4]

HIDDEN = [(232,4,'hidden_block_count_x'),(236,4,'hidden_block_count_y'),(240,4,'hidden_block_count_z'),
          (244,2,'hidden_group_size_x'),(246,2,'hidden_group_size_y'),(248,2,'hidden_group_size_z'),
          (250,2,'hidden_remainder_x'),(252,2,'hidden_remainder_y'),(254,2,'hidden_remainder_z'),
          (272,8,'hidden_global_offset_x'),(280,8,'hidden_global_offset_y'),(288,8,'hidden_global_offset_z'),
          (296,2,'hidden_grid_dims'),(352,4,'hidden_dynamic_lds_size')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_hash(kind, data):
    return hashlib.sha1(kind.encode() + b' ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verify_tree(tree, expected):
    if tree.get('truncated') or tree.get('sha') != expected:
        raise ValueError('wrong or truncated device-library tree')
    entries = tree['tree']
    paths = set()
    for e in entries:
        p = PurePosixPath(e['path'])
        if p.is_absolute() or '..' in p.parts or str(p) != e['path'] or e['path'] in paths:
            raise ValueError('unsafe/duplicate tree path')
        paths.add(e['path'])
        if (e['type'], e['mode']) not in {('tree','040000'), ('blob','100644'), ('blob','100755')}:
            raise ValueError('unsupported tree entry')
    directories = {'': expected, **{e['path']:e['sha'] for e in entries if e['type']=='tree'}}
    for folder, digest in directories.items():
        children = [e for e in entries if str(PurePosixPath(e['path']).parent) == (folder or '.')]
        children.sort(key=lambda e: (PurePosixPath(e['path']).name + ('/' if e['type']=='tree' else '')).encode())
        raw = b''.join((e['mode'].lstrip('0')+' '+PurePosixPath(e['path']).name).encode()+b'\0'+bytes.fromhex(e['sha']) for e in children)
        if git_hash('tree',raw) != digest:
            raise ValueError('tree identity mismatch: '+folder)
    if any(str(PurePosixPath(e['path']).parent) not in directories and str(PurePosixPath(e['path']).parent) != '.' for e in entries):
        raise ValueError('missing tree parent')
    return [e for e in entries if e['type']=='blob']


def fetch(url, limit):
    req = urllib.request.Request(url, headers={'User-Agent':'ps5-hip-source-fixture'})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(limit+1)
    if len(data)>limit:
        raise ValueError('download exceeds bound')
    return data


def selected_metadata(notes):
    blocks = re.split(r'(?m)^  - \.args:\s*$', notes)[1:]
    matches = [b for b in blocks if re.search(r'(?m)^    \.name:\s+'+re.escape(SYMBOL)+r'\s*$', b)]
    if len(matches)!=1:
        raise ValueError('selected RMSNorm kernel missing/duplicate')
    block=matches[0]
    args=[]
    argument_block = re.split(r'(?m)^    \.', block, maxsplit=1)[0]
    for chunk in re.split(r'(?m)^      - ', argument_block)[1:]:
        def field(key):
            found=re.search(r'(?m)^\s*\.'+key+r':\s+(\S+)',chunk)
            if not found: raise ValueError('argument metadata missing '+key)
            return found[1]
        args.append(dict(offset=int(field('offset')),size=int(field('size')),kind=field('value_kind')))
    expected=[dict(offset=o,size=s,kind='global_buffer' if i in (0,1,9,17) else 'by_value') for i,(o,s) in enumerate(zip(OFFSETS,SIZES))]
    expected += [dict(offset=o,size=s,kind=k) for o,s,k in HIDDEN]
    if args!=expected: raise ValueError('RMSNorm complete argument ABI changed')
    fields={}
    for name, wanted in {'kernarg_segment_size':488,'kernarg_segment_align':8,'group_segment_fixed_size':0,'private_segment_fixed_size':0,'max_flat_workgroup_size':1024,'wavefront_size':32}.items():
        match=re.search(r'(?m)^    \.'+name+r':\s+(\d+)',block)
        if not match or int(match[1])!=wanted: raise ValueError('RMSNorm resource changed: '+name)
        fields[name]=wanted
    if [a for a in args if a['kind']=='hidden_dynamic_lds_size'] != [dict(offset=352,size=4,kind='hidden_dynamic_lds_size')]:
        raise ValueError('dynamic LDS ABI changed')
    return dict(symbol=SYMBOL,arguments=args,resources=fields)


def build(output):
    output=output.resolve()
    if not any(output.is_relative_to(ROOT/p) for p in ('results','build')):
        raise ValueError('output must be under results/ or build/')
    output.mkdir(parents=True,exist_ok=False)
    commands=[]
    def run(label, argv, timeout=300):
        argv=list(map(str,argv))
        try:
            result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        except subprocess.TimeoutExpired as error:
            commands.append(dict(label=label,argv=argv,timed_out=True))
            (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            raise RuntimeError(label+' timed out') from error
        (output/(label+'.stdout')).write_text(result.stdout)
        (output/(label+'.stderr')).write_text(result.stderr)
        commands.append(dict(label=label,argv=argv,exit_code=result.returncode))
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode: raise RuntimeError(label+': '+result.stderr[-2000:])
        return result.stdout
    inputs={str(ROOT/n):sha(ROOT/n) for n in ('tools/build_llm_rmsnorm_fixture.py','tools/fetch_locked_deps.py','tools/probe_ocml_math.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py','tools/ocml_math_inputs.json','evidence/offline-llm-hip-compile-probe.json','dependencies.json','llm-inputs.json','ports/clr/manifest.json')}
    inputs[str(Path(sys.executable).resolve())]=sha(sys.executable)
    pins=read('tools/ocml_math_inputs.json')
    client=source('llama-client',read('llm-inputs.json')['client'],False)
    for name in ('hip-headers','clr'): source(name,sources()[name],False)
    tools={}
    for name in ('clang-18','clang++-18','llvm-config-18','llvm-dis-18','llvm-nm-18','llvm-readobj-18','llvm-objdump-18','llvm-link-18','ld.lld-18','opt-18','cmake','ninja'):
        path=shutil.which(name)
        if not path: raise RuntimeError('missing '+name)
        version=run(name+'-version',[path,'--version'])
        if name.endswith('-18') and '18.1.8' not in version: raise ValueError('LLVM18.1.8 required')
        tools[name]=dict(path=path,sha256=sha(path),version=version.splitlines()[0])
    print('Fetching authenticated device-library subtree',flush=True)
    tree=json.loads(fetch('https://api.github.com/repos/ROCm/llvm-project/git/trees/'+pins['device_libraries_subtree']+'?recursive=1',2_000_000))
    blobs=verify_tree(tree,pins['device_libraries_subtree'])
    (output/'device-tree.json').write_text(json.dumps(tree,indent=2)+'\n')
    src=output/'device-libs-source';src.mkdir()
    def download(e):
        data=fetch('https://raw.githubusercontent.com/ROCm/llvm-project/'+pins['device_libraries_commit']+'/amd/device-libs/'+e['path'],e['size'])
        if len(data)!=e['size'] or git_hash('blob',data)!=e['sha']: raise ValueError('blob identity mismatch: '+e['path'])
        p=src/e['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        return str(p),sha(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        original=dict(pool.map(download,blobs))
    policy=src/'cmake/OCL.cmake';text=policy.read_text();old='cmake_policy(SET CMP0053 OLD)'
    if text.count(old)!=1: raise ValueError('CMake policy derivation precondition changed')
    policy.write_text(text.replace(old,'cmake_policy(SET CMP0053 NEW)'))
    derived=dict(original);derived[str(policy)]=sha(policy);inputs.update(derived)
    headers=output/'diagnostic-hipblas';headers.mkdir()
    hspec=read('evidence/offline-llm-hip-compile-probe.json')['header_comparisons']['diagnostic_6_2']
    for name, record in hspec['files'].items():
        data=fetch(record['url'],record['bytes'])
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']: raise ValueError('header identity mismatch')
        dest=headers/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);inputs[str(dest)]=sha(dest)
    (headers/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.21)
project(hipblas_header_probe LANGUAGES CXX)
add_library(hipblas SHARED IMPORTED)
include(GenerateExportHeader)
set_target_properties(hipblas PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
generate_export_header(hipblas EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/hipblas/hipblas-export.h)
set(hipblas_VERSION_MAJOR 2)
set(hipblas_VERSION_MINOR 2)
set(hipblas_VERSION_PATCH 0)
set(hipblas_VERSION_TWEAK 0)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hipblas/hipblas-version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/hipblas/hipblas-version.h @ONLY)
''')
    inputs[str(headers/'CMakeLists.txt')]=sha(headers/'CMakeLists.txt')
    run('headers-configure',['cmake','-S',headers,'-B',headers/'build','-G','Ninja','-DCMAKE_CXX_COMPILER='+tools['clang++-18']['path']])
    wrapper=output/'wrapper';wrapper.mkdir()
    bindir=run('llvm-bindir',['llvm-config-18','--bindir']).strip()
    llvmdir=run('llvm-cmakedir',['llvm-config-18','--cmakedir']).strip()
    (wrapper/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.20)
project(ps5hip_rms_libraries LANGUAGES C CXX)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
find_package(LLVM REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
foreach(tool clang llvm-link opt llvm-objdump)
 if(NOT TARGET ${tool})
  add_executable(${tool} IMPORTED GLOBAL)
  set_target_properties(${tool} PROPERTIES IMPORTED_LOCATION "'''+bindir+'''/${tool}")
 endif()
endforeach()
set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
add_subdirectory("../device-libs-source" upstream)
''')
    inputs[str(wrapper/'CMakeLists.txt')]=sha(wrapper/'CMakeLists.txt')
    run('libraries-configure',['cmake','-S',wrapper,'-B',output/'cmake','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+tools['clang-18']['path'],'-DCMAKE_CXX_COMPILER='+tools['clang++-18']['path'],'-DLLVM_DIR='+llvmdir])
    constants={k:v for k,v in CONSTANTS.items() if k!='oclc_isa_version_1030'}
    print('Building upstream OCML/configuration and two OCKL slices',flush=True)
    run('libraries-build',['cmake','--build',output/'cmake','--target','ocml',*constants,'--parallel','4'],900)
    libs=output/'libs';libs.mkdir()
    for name in ('ocml',*constants): shutil.copyfile(output/'cmake/upstream/amdgcn/bitcode'/(name+'.bc'),libs/(name+'.bc'))
    lines=(output/'cmake/build.ninja').read_text().splitlines()
    for name,output_name in (('workitem','ockl-workitem-only'),('mul_hi','mul_hi')):
        line=next(l for l in lines if 'COMMAND =' in l and '/ockl/src/'+name+'.cl -o ' in l)
        argv=shlex.split(line.split(' && ')[1]);argv=argv[:argv.index('-o')]+['-o',str(output/(name+'.bc'))]
        run(name+'-compile',argv)
        run(name+'-prepare',[output/'cmake/upstream/utils/prepare-builtins/prepare-builtins','-o',libs/(output_name+'.bc'),output/(name+'.bc')])
    for name, expected in constants.items(): audit_constant(run(name+'-ir',['llvm-dis-18',libs/(name+'.bc'),'-o','-']),expected)
    for path in libs.glob('*.bc'): inputs[str(path)]=sha(path)
    inputs[str(output/'cmake/upstream/utils/prepare-builtins/prepare-builtins')]=sha(output/'cmake/upstream/utils/prepare-builtins/prepare-builtins')
    argv=['clang++-18','-x','hip','-nogpuinc','--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-DNDEBUG','-O2','-fPIC','-DGGML_USE_HIP','-DGGML_HIP_NO_VMM','-DGGML_HIP_NO_MMQ_MFMA','-DGGML_CUDA_NO_FA']
    argv+=['-I'+str(p) for p in (ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',ROOT/'ports/hip',client/'ggml/include',client/'ggml/src',headers)]
    argv+=['-include','__clang_hip_runtime_wrapper.h','--cuda-device-only',client/'ggml/src/ggml-cuda/norm.cu','--no-gpu-bundle-output','-B'+bindir,'--rocm-device-lib-path='+str(libs)]
    ordered=['ockl-workitem-only','mul_hi','ocml',*constants]
    argv+=['--hip-device-lib='+n+'.bc' for n in ordered]
    run('norm-dependencies',[*argv,'-M','-MF',output/'norm.d'])
    for name in shlex.split((output/'norm.d').read_text().replace('\\\n',' ').split(':',1)[1]):
        path=Path(name).resolve();inputs[str(path)]=sha(path)
    print('Compiling and linking unchanged norm.cu',flush=True)
    run('norm-link',[*argv,'-o',output/'norm.hsaco'])
    undefined=run('norm-undefined',['llvm-nm-18','--undefined-only',output/'norm.hsaco'])
    if undefined.strip(): raise ValueError('unresolved device imports')
    notes=run('norm-notes',['llvm-readobj-18','--notes',output/'norm.hsaco'])
    selected=selected_metadata(notes)
    names=re.findall(r'(?m)^    \.name:\s+(\S+)',notes)
    if len(names)!=16: raise ValueError('norm module kernel inventory changed')
    isa=run('norm-isa',['llvm-objdump-18','--disassemble','--mcpu=gfx1013',output/'norm.hsaco'])
    audit=audit_object((output/'norm.hsaco').read_bytes(),notes,isa,names)
    desc=audit['descriptors'][SYMBOL]
    if (desc['group_bytes'],desc['private_bytes'],desc['kernarg_bytes'],desc['preload'])!=(0,0,488,0): raise ValueError('RMSNorm descriptor storage changed')
    if (desc['rsrc1'],desc['rsrc2'],desc['properties'],desc['rsrc3'])!=(0x60af00c3,0x38c,0x409,0): raise ValueError('RMSNorm descriptor changed')
    for path,digest in inputs.items():
        if sha(path)!=digest: raise ValueError('input changed: '+path)
    for tool in tools.values():
        if sha(tool['path'])!=tool['sha256']: raise ValueError('tool changed')
    source('llama-client',read('llm-inputs.json')['client'],False)
    for name in ('hip-headers','clr'):source(name,sources()[name],False)
    receipt=dict(schema=1,status='PASS',gpu_executed=False,native_executed=False,inputs=inputs,tools=tools,device_library_original_sources=original,device_library_tree=pins['device_libraries_subtree'],selected=selected,descriptor=desc,kernel_count=len(names),undefined_symbols=[],library_order=ordered,diagnostic_hipblas=hspec,commands=commands,artifacts={n:dict(sha256=sha(output/n),bytes=(output/n).stat().st_size) for n in ('norm.hsaco','norm.d','norm-notes.stdout','norm-isa.stdout','commands.json','device-tree.json')},limits=['Compiler/link/metadata proof only; no numerical or runtime admission proof.','Diagnostic hipBLAS2.2 headers match locked HIP6.2; no hipBLAS implementation or final client dependency integration.'])
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS '+str(output/'receipt.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
