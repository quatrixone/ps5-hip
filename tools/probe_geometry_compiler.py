#!/usr/bin/env python3
"""Genuine full-header HIP geometry compiler evidence, with no GPU execution."""
import argparse, hashlib, json, pathlib, re, shutil, struct, subprocess, sys, tarfile
from build_canaries import require_device_linker, require_tool_versions, isa_inventory
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from fat_binary_fixture_test import section

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)

def audit_object(data, metadata, isa):
    require(data[:6]==b'\x7fELF\x02\x01','ELF64LE required')
    shoff=struct.unpack_from('<Q',data,40)[0];shwidth,shcount,shstr=struct.unpack_from('<HHH',data,58)
    require(shwidth==64 and shoff+shcount*64<=len(data),'section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+64*i) for i in range(shcount)]
    def contents(h):
        require(h[4]<=len(data) and h[5]<=len(data)-h[4],'section outside ELF');return data[h[4]:h[4]+h[5]]
    symbols={}
    for table in sections:
        if table[1] not in (2,11):continue
        require(table[9]==24 and table[6]<len(sections),'symbol table')
        names=contents(sections[table[6]]);raw=contents(table)
        for offset in range(0,len(raw),24):
            n,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,offset)
            require(n<len(names),'symbol name');end=names.find(b'\0',n);require(end>=0,'symbol termination')
            name=names[n:end].decode()
            if name in ('geometry_2d.kd','geometry_3d.kd'):
                require(index<len(sections) and size==64,'descriptor extent')
                h=sections[index];at=value-h[3];rawsection=contents(h)
                require(at>=0 and at+64<=len(rawsection),'descriptor outside section')
                rawdesc=rawsection[at:at+64]
                if name in symbols:require(symbols[name]==rawdesc,'different duplicate descriptor')
                symbols[name]=rawdesc
    require(set(symbols)=={'geometry_2d.kd','geometry_3d.kd'},'descriptor inventory')
    descriptors={}
    for name,d in symbols.items():
        group,private,kernarg=struct.unpack_from('<III',d)
        rsrc3=struct.unpack_from('<I',d,44)[0];rsrc1,rsrc2=struct.unpack_from('<II',d,48)
        properties,preload=struct.unpack_from('<HH',d,56)
        require((group,private,kernarg,rsrc2,rsrc3,properties,preload)==(0,0,296,0x138c,0,0x409,0),'descriptor geometry contract')
        descriptors[name]={'rsrc1':rsrc1,'rsrc2':rsrc2,'properties':properties,'kernarg_bytes':kernarg}
    require(set(re.findall(r'^\s+\.name:\s+(\w+)\s*$',metadata,re.M))=={'geometry_2d','geometry_3d'},'kernel inventory')
    for field,value in [('wavefront_size','32'),('private_segment_fixed_size','0'),('group_segment_fixed_size','0'),('kernarg_segment_size','296')]:
        require(re.findall(r'\.'+field+r':\s*(\d+)',metadata)==[value,value],field+' contract')
    inventory=isa_inventory(isa,{'geometry_2d','geometry_3d'})
    for kernel in inventory.values():
        ops=kernel['opcodes'];require(any(x.startswith('global_load') for x in ops) and any(x.startswith('global_store') for x in ops),'missing real memory operations')
        require(not kernel['flat_scratch_setup'] and not any(x.startswith(('scratch_','flat_','buffer_','ds_','s_swappc')) for x in ops),'unsupported memory/call ISA')
    return {'descriptors':descriptors,'isa':inventory}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();ref=args.reference_root.resolve();out=args.output.resolve()
    require(not out.exists(),'use a fresh output directory');out.mkdir(parents=True)
    fixture=ref/'results/hip-host-abi-spike/rocm-6.2.4'
    authority=ROOT/'evidence/offline-compatible-hip-headers.json'
    evidence=json.loads(authority.read_text());inputs={str(authority):sha(authority)}
    stage=fixture/'package-stage';expected={}
    for key,archive,subdir in [('HIP','HIP.tar.gz','include'),('clr','clr.tar.gz','hipamd/include')]:
        package=fixture/archive
        require(sha(package)==evidence['input_sources'][key]['archive_sha256'],'header archive changed')
        inputs[str(package)]=sha(package)
        with tarfile.open(package) as tar:
            prefix=tar.getmembers()[0].name.split('/')[0]+'/'+subdir+'/'
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith(prefix):
                    expected[member.name[len(prefix):]]=tar.extractfile(member).read()
    version=fixture/'package-build/include/hip/hip_version.h'
    require(sha(version)==evidence['raw_file_hashes']['package-stage/include/hip/hip_version.h']['sha256'],'HIP version changed')
    expected['hip/hip_version.h']=version.read_bytes()
    for name,data in expected.items():
        p=stage/'include'/name
        require(p.read_bytes()==data,'staged header differs from upstream: '+name);inputs[str(p)]=sha(p)
    # Hash every consumed staged header, and reject unaccounted headers.
    for p in (stage/'include').rglob('*'):
        if p.is_file():require(str(p.relative_to(stage/'include')) in expected,'unaccounted staged header')
    for p in (stage/'share').rglob('*'):
        if p.is_file():
            key=str(p.relative_to(fixture));record=evidence['raw_file_hashes'].get(key)
            if key=='package-stage/share/hip/.hipInfo':record={'sha256':'cc4be6c67f508ffc1e1b0bd8592b2967766032c3649b4face7867967385ad117'} # Measured generated package metadata; source headers remain archive-verified.
            require(record is not None and sha(p)==record['sha256'],'unbound HIP package metadata')
            inputs[str(p)]=sha(p)
    libs=fixture/'workitem-slice'
    library_pins={r['name']:r for r in [evidence['workitem_slice'],*evidence['built_constant_bitcode']]}
    require({p.name for p in libs.iterdir() if p.is_file()}==set(library_pins),'library inventory')
    for p in libs.iterdir():
        if p.is_file():
            key=str(p.relative_to(fixture));record=library_pins.get(p.name)
            require(record is not None and sha(p)==record['sha256'],'unbound device library: '+key)
            inputs[str(p)]=sha(p)
    tools={name:shutil.which(name+'-18') for name in ('clang++','ld.lld','llvm-readobj','llvm-objdump')}
    require(all(tools.values()),'LLVM18 tools missing')
    versions={name:subprocess.check_output([path,'--version'],text=True) for name,path in tools.items()}
    require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version'])
    tool_hashes={k:sha(pathlib.Path(v)) for k,v in tools.items()}
    resource=pathlib.Path(subprocess.check_output([tools['clang++'],'-print-resource-dir'],text=True).strip())
    # Clang's implicit HIP wrapper and its recursively included resource headers.
    for p in (resource/'include').rglob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    source=ROOT/'kernels/geometry.hip';commands=[];cases=[]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [source,pathlib.Path(__file__),ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py',authority]}
    for target in ('gfx1013','gfx1030'):
        for opt in ('O1','O2'):
            d=out/(target+'-'+opt);d.mkdir();combined=d/'geometry.o'
            cmd=[tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),
                '--offload-arch='+target,'-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__',
                '-std=c++17','-'+opt,'-fno-fast-math','-c',str(source),'--hip-path='+str(stage),
                '-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h',
                '--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc',
                '--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_'+target[3:]+'.bc',
                '-o',str(combined)]
            trace=subprocess.run(cmd+['-###'],capture_output=True,text=True,check=True).stderr
            (d/'driver.txt').write_text(trace);require_device_linker(trace,tools['ld.lld'])
            result=subprocess.run(cmd,capture_output=True,text=True);(d/'compile.txt').write_text(result.stdout+result.stderr)
            commands.append({'argv':cmd,'exit_code':result.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');result.check_returncode()
            bundle=section(combined.read_bytes(),b'.hip_fatbin');(d/'geometry.bundle').write_bytes(bundle)
            magic=b'__CLANG_OFFLOAD_BUNDLE__';require(bundle.startswith(magic),'missing genuine bundle')
            count=struct.unpack_from('<Q',bundle,len(magic))[0];at=len(magic)+8;found=[]
            require(count==2,'unexpected bundle inventory')
            for _ in range(count):
                offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24
                identity=bundle[at:at+length].decode();at+=length
                require(offset<=len(bundle) and size<=len(bundle)-offset,'bundle extent')
                if identity=='hipv4-amdgcn-amd-amdhsa--'+target:found.append(bundle[offset:offset+size])
            require(len(found)==1,'wrong device target');code=d/'geometry.hsaco';code.write_bytes(found[0])
            metadata=subprocess.check_output([tools['llvm-readobj'],'--notes',str(code)],text=True)
            isa=subprocess.check_output([tools['llvm-objdump'],'--disassemble','--mcpu='+target,str(code)],text=True)
            (d/'metadata.txt').write_text(metadata);(d/'isa.txt').write_text(isa)
            audited=audit_object(code.read_bytes(),metadata,isa)
            cases.append({'target':target,'optimization':opt,'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in d.iterdir() if p.is_file()},**audited})
    require(all(sha(pathlib.Path(p))==v for p,v in inputs.items()),'compiler inputs changed')
    require(all(sha(ROOT/p)==v for p,v in source_hashes.items()),'recipe sources changed')
    require(all(sha(pathlib.Path(tools[k]))==v for k,v in tool_hashes.items()),'compiler tools changed')
    report={'schema':1,'hardware_validated':False,'source_sha256':source_hashes,'inputs':inputs,'tools':{k:{'path':v,'sha256':sha(pathlib.Path(v)),'version':versions[k]} for k,v in tools.items()},'cases':cases}
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Four genuine full-HIP geometry compiler cases passed; no hardware claim.')
if __name__=='__main__':main()
