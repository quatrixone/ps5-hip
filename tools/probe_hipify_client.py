#!/usr/bin/env python3
"""Translate and compile one ordinary CUDA client; no CUDA or PS5 execution."""
import argparse,difflib,hashlib,json,os,pathlib,re,shutil,struct,subprocess
import probe_hip_routes as shared
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section

ROOT=pathlib.Path(__file__).resolve().parents[1]
REVISION='0776ed88a0386fea6a10c991485083bb6b8b3433'
TRANSLATOR_SHA='0a123408a7bd7cc0a4c2bc1c414d0ba254393b935e671a60ba97a4a4a6c7495f'
NAME='translated_add'
MAPPING={'cudaError_t':'hipError_t','cudaSuccess':'hipSuccess','cudaGetErrorString':'hipGetErrorString',
    'cudaMalloc':'hipMalloc','cudaMemcpy':'hipMemcpy','cudaMemcpyHostToDevice':'hipMemcpyHostToDevice',
    'cudaMemcpyDeviceToHost':'hipMemcpyDeviceToHost','cudaGetLastError':'hipGetLastError',
    'cudaDeviceSynchronize':'hipDeviceSynchronize','cudaFree':'hipFree'}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def check_translation(original,translated):
    expected=original.replace('#include <cuda_runtime.h>','#include <hip/hip_runtime.h>')
    expected=re.sub(r'\b(?:'+ '|'.join(MAPPING)+r')\b',lambda m:MAPPING[m[0]],expected)
    expected='#include "hip/hip_runtime.h"\n'+expected
    require(translated==expected,'HIPIFY changed more than the declared headers/API names')
    require('<<<(n+63)/64, 64>>>' in translated,'ordinary launch syntax changed')
    return {name:len(re.findall(r'\b'+name+r'\b',original)) for name in MAPPING}

def device_object(bundle):
    magic=b'__CLANG_OFFLOAD_BUNDLE__'
    require(bundle.startswith(magic),'bundle magic')
    require(struct.unpack_from('<Q',bundle,len(magic))[0]==2,'bundle count')
    at=len(magic)+8;selected=[]
    for _ in range(2):
        offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24
        identity=bundle[at:at+length].decode();at+=length
        require(offset<=len(bundle) and size<=len(bundle)-offset,'bundle bounds')
        if identity=='hipv4-amdgcn-amd-amdhsa--gfx1013':selected.append(bundle[offset:offset+size])
    require(len(selected)==1,'device selection')
    return selected[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=pathlib.Path,required=True)
    p.add_argument('--hipify',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();ref=a.reference_root.resolve();hipify=a.hipify.resolve();out=a.output.resolve()
    require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    source=ROOT/'tests/cuda_vector_add.cu';translator=hipify/'bin/hipify-perl'
    files=[source,translator,hipify/'LICENSE.txt',pathlib.Path(__file__),ROOT/'tools/probe_hip_routes.py',
           ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py',ROOT/'dependencies.json']
    inputs={str(path):sha(path) for path in files};commands=[];cases=[]
    def run(label,argv):
        argv=list(map(str,argv))
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=180,
            env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18')))
        (out/(label+'.stdout')).write_text(result.stdout);(out/(label+'.stderr')).write_text(result.stderr)
        commands.append({'label':label,'argv':argv,'exit_code':result.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(result.returncode==0,label+' failed: '+result.stderr[-1000:])
        return result.stdout,result.stderr
    try:
        require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=hipify,text=True).strip()==REVISION,'HIPIFY revision')
        require(not subprocess.check_output(['git','status','--porcelain'],cwd=hipify,text=True),'HIPIFY dirty')
        require(sha(translator)==TRANSLATOR_SHA,'HIPIFY script changed')
        perl=pathlib.Path(shutil.which('perl'));inputs[str(perl)]=sha(perl)
        run('perl-version',[perl,'--version'])
        translated,_=run('hipify',[perl,translator,source])
        conversions=check_translation(source.read_text(),translated)
        hip=out/'translated.hip';hip.write_text(translated)
        (out/'translation.diff').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),translated.splitlines(True),fromfile='cuda_vector_add.cu',tofile='translated.hip')))
        _,stage,libs,prepared=shared.prepared_inputs(ref);inputs.update(prepared)
        tools={name:shutil.which(name+'-18') for name in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')}
        require(all(tools.values()),'LLVM18 tools missing')
        versions={name:run(name+'-version',[path,'--version'])[0] for name,path in tools.items()}
        require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version'])
        inputs.update({path:sha(path) for path in tools.values()})
        sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
        require((sdk/'.ps5-sdk-revision').read_text().strip()==json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision'],'SDK revision')
        for host in ('linux','ps5','linux-model'):
            d=out/host;d.mkdir();obj=d/'translated.o'
            command=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),
                '--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__',
                '-std=c++17','-O2','-fno-fast-math','-c',hip,'--hip-path='+str(stage),'-I'+str(stage/'include'),
                '-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),
                '--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc',
                '--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
            if host=='ps5':command+=['-Xarch_device','-fno-short-wchar']
            if host=='linux-model':command+=['-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
            _,driver=run(host+'-driver',command+['-###']);require_device_linker(driver,tools['ld.lld']);run(host+'-compile',command)
            bundle=section(obj.read_bytes(),b'.hip_fatbin');(d/'translated.bundle').write_bytes(bundle)
            code=d/'translated.hsaco';code.write_bytes(device_object(bundle))
            metadata,_=run(host+'-metadata',[tools['llvm-readobj'],'--notes',code])
            isa,_=run(host+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code])
            audit=shared.audit_object(code.read_bytes(),metadata,isa,(NAME,))
            hostread,_=run(host+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',obj])
            disasm,_=run(host+'-host-isa',[tools['llvm-objdump'],'-dr',obj])
            lifecycle=shared.audit_registration(disasm,(NAME,))
            wrapper_audit=shared.audit_wrapper(section(obj.read_bytes(),b'.hipFatBinSegment'),hostread,host)
            undefined,_=run(host+'-undefined',[tools['llvm-nm'],'--undefined-only',obj])
            require('hipLaunchKernel' in undefined and '__hipPushCallConfiguration' in undefined,'genuine launch missing')
            require(not re.search(r'\b(?:cuda|__cuda)\w+',undefined),'CUDA runtime reference survived translation')
            cases.append({'host':host,'audit':audit,'lifecycle':lifecycle,'wrapper':wrapper_audit,
                'object_sha256':sha(obj),'device_sha256':sha(code)})
            print(host+' compiled '+sha(code),flush=True)
        require(all(sha(path)==digest for path,digest in inputs.items()),'consumed input changed')
        record={'schema':1,'classification':'ordinary-cuda-hipify-compiler-proof','hipify_revision':REVISION,
            'hipify_source':'https://github.com/ROCm/HIPIFY/tree/'+REVISION,'inputs':inputs,'commands':commands,
            'translation_conversions':conversions,'hand_edits_to_generated_source':0,'cases':cases,
            'cuda_execution':False,'native_linked':False,'hardware_executed':False,
            'limits':['Exact translated example only; no CUDA execution or general CUDA compatibility claim.',
                      'Compilation and registration inspection do not establish runtime behavior.',
                      'Installed resource/SDK support trees not fully inventoried; no hermetic build claim.'],
            'artifacts':{str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()}}
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print('PASS '+sha(out/'receipt.json'))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'inputs':inputs,'commands':commands},indent=2)+'\n')
        raise

if __name__=='__main__':main()
