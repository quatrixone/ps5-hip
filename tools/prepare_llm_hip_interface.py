"""Generate an isolated, authenticated modern hipBLAS client header adaptation.

Supply official header inputs matching committed probe evidence. This tool does
not fetch dependencies, change the checkout, implement BLAS or qualify execution.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR = 'ggml/src/ggml-cuda/vendors/hip.h'
OLD = '#if HIP_VERSION >= 60500000'
NEW = '#if hipblasVersionMajor >= 3 || HIP_VERSION >= 60500000'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(data):
    text = data.decode('utf-8')
    if text.count(OLD) != 1 or NEW in text:
        raise ValueError('client interface guard drift')
    return text.replace(OLD, NEW, 1).encode()


def checked_header_files(folder, specification):
    result = {}
    for name, entry in specification['files'].items():
        path = folder/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('unsafe official header path')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('official header input drift: '+name)
        result[name] = path
    return result


def clean_client(client, revision):
    def git(*args):
        return subprocess.check_output(['git','-C',str(client),*args])
    if git('rev-parse','--show-toplevel').decode().strip() != str(client) or git('rev-parse','HEAD').decode().strip() != revision or git('status','--porcelain').strip():
        raise ValueError('clean locked client checkout required')
    data = git('show',revision+':'+VENDOR)
    if (client/VENDOR).is_symlink() or (client/VENDOR).read_bytes() != data:
        raise ValueError('client vendor header differs from locked Git object')
    return data


def prepare(client, headers, output):
    client, headers, output = (Path(p).resolve() for p in (client,headers,output))
    if output.exists() or output == ROOT/'results' or not output.is_relative_to(ROOT/'results'):
        raise ValueError('fresh output beneath results/ required')
    lock_path = ROOT/'llm-inputs.json'
    evidence_path = ROOT/'evidence/offline-llm-hip-compile-probe.json'
    lock = json.loads(lock_path.read_text());evidence = json.loads(evidence_path.read_text())
    revision = lock['client']['revision']
    if revision != evidence['client_revision']:
        raise ValueError('client lock and reviewed probe differ')
    original = clean_client(client,revision)
    changed = adapt(original)
    spec = evidence['header_comparisons']['current_main']
    files = checked_header_files(headers,spec)
    version = re.search(r'set\s*\(\s*VERSION_STRING\s+"(\d+)\.(\d+)\.(\d+)"\s*\)',files['hipblas-CMakeLists.txt'].read_text())
    if not version or tuple(map(int,version.groups())) != (3,8,0):
        raise ValueError('reviewed modern hipBLAS version differs')
    cmake = Path(shutil.which('cmake') or '')
    compiler = Path(shutil.which('clang++-18') or '')
    if not cmake.is_file() or not compiler.is_file():raise ValueError('CMake and clang++-18 required')
    compiler_version = subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]
    if 'version 18.1.8' not in compiler_version:
        raise ValueError('interface generator requires LLVM18.1.8')
    # Bind the actual CMake module used to derive visibility/export declarations.
    system = subprocess.check_output([str(cmake),'--system-information'],text=True)
    match = re.search(r'^CMAKE_ROOT "([^"]+)"$',system,re.M)
    if not match:raise ValueError('CMake module root unavailable')
    modules = Path(match.group(1))/'Modules'
    identities = {str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'tests/llm_hip_interface_test.py',lock_path,evidence_path,client/VENDOR,cmake,compiler,*files.values(),modules/'GenerateExportHeader.cmake']}
    output.mkdir(parents=True)
    include = output/'include'
    for name in ('hipblas/hipblas.h','hipblas/hipblas-version.h.in','hipblas-common/hipblas-common.h'):
        target=include/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(files[name],target)
    project=output/'CMakeLists.txt'
    project.write_text('''cmake_minimum_required(VERSION 3.21)
project(ps5hip_interface LANGUAGES CXX)
add_library(hipblas SHARED IMPORTED)
include(GenerateExportHeader)
set_target_properties(hipblas PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
generate_export_header(hipblas EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas/hipblas-export.h)
set(hipblas_VERSION_MAJOR 3)
set(hipblas_VERSION_MINOR 8)
set(hipblas_VERSION_PATCH 0)
set(hipblas_VERSION_TWEAK 0)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas/hipblas-version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas/hipblas-version.h @ONLY)
''')
    command=[str(cmake),'-S',str(output),'-B',str(output/'cmake'),'-DCMAKE_CXX_COMPILER='+str(compiler)]
    with (output/'configure.stdout').open('w') as stdout,(output/'configure.stderr').open('w') as stderr:
        subprocess.run(command,stdout=stdout,stderr=stderr,check=True,timeout=90)
    template=files['hipblas/hipblas-version.h.in'].read_text()
    for key,value in zip(('MAJOR','MINOR','PATCH','TWEAK'),('3','8','0','0')):
        template=template.replace('@hipblas_VERSION_'+key+'@',value)
    if (include/'hipblas/hipblas-version.h').read_text()!=template:raise ValueError('generated version header differs')
    adapted=output/'hip.h';adapted.write_bytes(changed)
    (output/'client.patch').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),changed.decode().splitlines(True),fromfile='a/'+VENDOR,tofile='b/'+VENDOR)))
    overlay=output/'overlay.json'
    overlay.write_text(json.dumps({'version':0,'roots':[{'type':'file','name':str(client/VENDOR),'external-contents':str(adapted)}]},indent=2)+'\n')
    if clean_client(client,revision)!=original or any(sha(p)!=digest for p,digest in identities.items()):raise ValueError('interface inputs changed')
    artifact_paths=[adapted,overlay,project,output/'client.patch',*include.rglob('*.h'),include/'hipblas/hipblas-version.h.in']
    receipt=dict(schema=1,status='PASS',classification='isolated-modern-hipblas-interface',client_revision=revision,hipblas_revision=spec['revision'],hipblas_version='3.8.0',compiler_version=compiler_version,inputs=identities,artifacts={str(p.relative_to(output)):sha(p) for p in artifact_paths},command=command,compiler_flags=['-ivfsoverlay',str(overlay),'-I'+str(include)],native_executed=False,limits=['Supplied official inputs are authenticated; no dependency fetch recipe.','Header adaptation only; no BLAS implementation or runtime qualification.','CMake imported target generates visibility macros only; no library implementation is built.'])
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client',type=Path,default=ROOT/'.deps/sources/llama-client')
    parser.add_argument('--headers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.client,args.headers,args.output)
