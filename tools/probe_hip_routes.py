#!/usr/bin/env python3
"""Fresh genuine single-TU HIP route compiler proof, no runtime/GPU execution."""
import argparse,hashlib,json,pathlib,re,shutil,struct,subprocess,sys,tarfile,os
from build_canaries import require_device_linker,require_tool_versions,isa_inventory
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from fat_binary_fixture_test import section
NAMES=('routes_integer','routes_lds','routes_spill','routes_geometry')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def audit_object(data,metadata,isa,kernel_names=NAMES):
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
            if name in {n+'.kd' for n in kernel_names}:
                require(index<len(sections) and size==64,'descriptor extent')
                h=sections[index];at=value-h[3];rawsection=contents(h)
                require(at>=0 and at+64<=len(rawsection),'descriptor outside section')
                rawdesc=rawsection[at:at+64]
                if name in symbols:require(symbols[name]==rawdesc,'different duplicate descriptor')
                symbols[name]=rawdesc
    require(set(symbols)=={n+'.kd' for n in kernel_names},'genuine descriptor inventory')
    descriptors={}
    for name,d in symbols.items():
        group,private,kernarg=struct.unpack_from('<III',d);rsrc3=struct.unpack_from('<I',d,44)[0]
        rsrc1,rsrc2=struct.unpack_from('<II',d,48);properties,preload=struct.unpack_from('<HH',d,56)
        descriptors[name[:-3]]={'group_bytes':group,'private_bytes':private,'kernarg_bytes':kernarg,'rsrc1':rsrc1,'rsrc2':rsrc2,'rsrc3':rsrc3,'properties':properties,'preload':preload}
    require(set(re.findall(r'^\s+\.name:\s+(\w+)\s*$',metadata,re.M))==set(kernel_names),'kernel metadata inventory')
    inventory=isa_inventory(isa,set(kernel_names))
    return {'descriptors':descriptors,'isa':inventory}

def audit_registration(disasm,names=NAMES):
    functions={}
    for block in re.split(r'(?m)^([0-9a-f]+ <[^>]+>:)\n',disasm)[1:]:
        if re.fullmatch(r'[0-9a-f]+ <[^>]+>:',block):name=block.split('<',1)[1][:-2];functions[name]=''
        else:functions[name]+=block
    require('__hip_module_ctor' in functions and '__hip_module_dtor' in functions,'compiler lifecycle functions missing')
    ctor=functions['__hip_module_ctor'];dtor=functions['__hip_module_dtor']
    def references(text,name):return len(re.findall(r'R_X86_64_[A-Z0-9_]+\s+'+re.escape(name)+r'(?:-|\s|$)',text))
    require(references(ctor,'__hipRegisterFatBinary')==1 and references(ctor,'__hipRegisterFunction')==len(names) and references(ctor,'atexit')==1,'compiler constructor registration/atexit count')
    require(references(dtor,'__hipUnregisterFatBinary')==1,'compiler destructor unregister missing')
    require(all(references(ctor,n)==1 for n in names),'constructor does not bind each genuine kernel token')
    return {'register_module_calls':1,'register_function_calls':len(names),'atexit_calls':1,'unregister_calls':1}

def audit_wrapper(raw,readobj,host):
    # ASan adds a redzone to this global only in the instrumented host model.
    size=64 if host=='linux-model' else 24
    require(host in ('linux','ps5','linux-model'),'unknown wrapper host')
    require(len(raw)==size and raw[:8]==bytes.fromhex('4650494801000000') and raw[8:]==bytes(size-8),'exact compiler wrapper missing')
    symbols=re.findall(r'  Symbol \{\n(.*?)\n  \}',readobj,re.S)
    wrappers=[s for s in symbols if re.search(r'^    Name: __hip_fatbin_wrapper \(',s,re.M)]
    require(len(wrappers)==1 and re.search(r'^    Value: 0x0$',wrappers[0],re.M) and re.search(r'^    Size: '+str(size)+r'$',wrappers[0],re.M) and re.search(r'^    Section: \.hipFatBinSegment \(',wrappers[0],re.M),'wrapper symbol extent changed')
    relocations=re.findall(r'  Section \(\d+\) \.rela\.hipFatBinSegment \{\n(.*?)\n  \}',readobj,re.S)
    require(len(relocations)==1 and relocations[0].strip()=='0x8 R_X86_64_64 .hip_fatbin 0x0','wrapper bundle relocation changed')
    return {'payload_bytes':24,'section_bytes':size,'model_redzone_bytes':size-24}

def check_profiles(audit):
    d=audit['descriptors']
    for n in NAMES:
        require(d[n]['rsrc3']==0 and d[n]['preload']==0 and d[n]['properties']==0x409,'unsupported entry properties '+n)
        require(not audit['isa'][n]['flat_scratch_setup'],'flat scratch setup '+n)
        ops=audit['isa'][n]['opcodes']
        require(not any(x.startswith(('flat_','scratch_','s_swappc')) for x in ops),'unsupported ISA '+n)
    require(tuple(d['routes_integer'][k] for k in ('group_bytes','private_bytes','kernarg_bytes','rsrc2'))==(0,0,24,0x8c),'initial profile changed')
    require(tuple(d['routes_lds'][k] for k in ('group_bytes','private_bytes','kernarg_bytes','rsrc2'))==(256,0,16,0x8c),'explicit-only LDS profile changed')
    require(tuple(d['routes_geometry'][k] for k in ('group_bytes','private_bytes','kernarg_bytes','rsrc2'))==(0,0,296,0x138c),'geometry profile changed')
    require(d['routes_spill']['group_bytes']==0 and d['routes_spill']['private_bytes']==212 and d['routes_spill']['kernarg_bytes']==16 and d['routes_spill']['rsrc2']==0x8d,'explicit-only buffer spill profile changed')
    require(any(x.startswith('buffer_load') for x in audit['isa']['routes_spill']['opcodes']) and any(x.startswith('buffer_store') for x in audit['isa']['routes_spill']['opcodes']),'actual spill/reload instructions missing')
    require(any(x.startswith('ds_') for x in audit['isa']['routes_lds']['opcodes']) and 's_barrier' in audit['isa']['routes_lds']['opcodes'],'actual LDS/barrier missing')

def prepared_inputs(ref):
    fixture=ref/'results/hip-host-abi-spike/rocm-6.2.4'
    authority=ROOT/'evidence/offline-compatible-hip-headers.json'
    evidence=json.loads(authority.read_text());inputs={str(authority):sha(authority),str(ROOT/'dependencies.json'):sha(ROOT/'dependencies.json')}
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
    return fixture,stage,libs,inputs

def native_proof(ref,out,core,common,execute,inputs,adapter=None):
    from probe_host_registration import audit_image
    from native_import_libraries_test import read_imports,nid
    def run(label,*argv):return execute(label,argv)[0]
    # Native source copies are private to this fresh output; no cached export is edited.
    lock = json.loads((ROOT/'dependencies.json').read_text())
    native_source = ref/'.deps'/('native-tools-'+lock['native_tools']['revision'])
    native = out/'native-tools'
    native_authority = ROOT/'evidence/offline-lds-native-readiness.json'
    inputs[str(native_authority)]=sha(native_authority)
    native_expected = json.loads(native_authority.read_text())['profiles'][0]['native_tool_source_sha256']
    for name, digest in native_expected.items():
        require(sha(native_source/name) == digest, 'pinned native export changed: '+name)
    shutil.copytree(native_source, native)
    before = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    original_script = (native/'ps5-pie.ld').read_text()
    needle = '    .rodata : { *(.rodata .rodata.*) } : ro'
    require(original_script.count(needle) == 1, 'native linker source changed')
    modified = original_script.replace(needle, '''    .rodata : {
        HIDDEN(__ps5hip_names_begin = .);
        *(.rodata .rodata.*)
        HIDDEN(__ps5hip_names_end = .);
        . = ALIGN(4096);
        HIDDEN(__ps5hip_bundle_begin = .);
        KEEP(*(.hip_fatbin))
        HIDDEN(__ps5hip_bundle_end = .);
    } : ro''').replace('        *(.data.rel.ro .data.rel.ro.*)', '''        HIDDEN(__ps5hip_relro_begin = .);
        *(.data.rel.ro .data.rel.ro.*)
        . = ALIGN(8);
        HIDDEN(__ps5hip_wrappers_begin = .);
        KEEP(*(.hipFatBinSegment))
        HIDDEN(__ps5hip_wrappers_end = .);
        HIDDEN(__ps5hip_relro_end = .);''')
    (native/'ps5-pie.ld').write_text(modified)
    sdk = (ref/'.deps/sdk').resolve()
    require((sdk/'.ps5-sdk-revision').read_text().strip() == lock['sdk']['revision'], 'SDK revision changed')
    native_libs = sdk/'target/lib'
    sdk_inputs = {n:sha(native_libs/n) for n in ('libc++.a','libc++abi.a','libunwind.a','libc.a','libps5platform.a','libkernel.so','libSceLibcInternal.so','libSceNet.so')}
    objects = [out/'ps5/routes.o']
    for source in [*core,adapter or ROOT/'tools/hip_routes_control.cpp',ROOT/'src/agc_backend.cpp',native/'app_crt.cpp']:
        obj = out/('ps5-'+source.stem+'.o')
        run('native-compile-'+source.stem, sdk/'bin/prospero-clang++', *common, '-fPIC', '-fvisibility=hidden', '-c', source, '-o', obj)
        objects.append(obj)
    stubs=[]
    for library,declarations in {
        'libSceAgc':'int sceAgcInit(unsigned){return -1;} int sceAgcSuspendPoint(){return -1;}',
        'libSceAgcDriver':'int sceAgcDriverSubmitDcb(void*){return -1;}'}.items():
        source=out/(library+'.cpp');source.write_text('extern "C" {'+declarations+'}\n')
        obj=out/(library+'.o');stub=out/(library+'.so')
        run('stub-compile-'+library,sdk/'bin/prospero-clang++','-fPIC','-c',source,'-o',obj)
        run('stub-link-'+library,'ld.lld-18','-shared','-soname',library+'.prx',obj,'-o',stub)
        stubs.append(stub)
    (out/'symbols.map').write_text('{ local: *; };\n')
    unwind = [part for name in ('eh_frame', 'eh_frame_hdr') for part in
              ('--defsym', f'__{name}_start=ADDR(.{name})', '--defsym', f'__{name}_end=ADDR(.{name})+SIZEOF(.{name})')]
    libs = sdk/'target/lib'
    pie, elf = out/'llvm-pie.elf', out/'eboot.elf'
    run('native-pie-link', 'ld.lld-18', '-pie', '-T', native/'ps5-pie.ld', '--eh-frame-hdr', *unwind,
        '-z', 'max-page-size=0x4000', '--hash-style=gnu', '--version-script', out/'symbols.map', '--exclude-libs=ALL',
        '-L', libs, '-e', '_start', '-o', pie, *objects, *stubs, '--start-group',
        *[libs/n for n in ('libc++.a', 'libc++abi.a', 'libunwind.a', 'libc.a', 'libps5platform.a')],
        '--end-group', '--as-needed', *[libs/n for n in ('libkernel.so', 'libSceLibcInternal.so', 'libSceNet.so')])
    zlib = ref/'.deps'/('host-zlib-'+lock['host_zlib']['version'])/'root'
    inputs[str(zlib.parent/'receipt.json')]=sha(zlib.parent/'receipt.json')
    zlib_receipt = json.loads((zlib.parent/'receipt.json').read_text())
    require(zlib_receipt['source_sha256'] == lock['host_zlib']['sha256'], 'zlib source identity changed')
    for name, digest in zlib_receipt['artifacts'].items():
        require(sha(zlib/name) == digest, 'zlib artifact changed')
        inputs[str(zlib/name)]=digest
    converter = out/'ps5-native-tool'
    run('converter-build', 'clang++-18', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
        *[native/n for n in ('native_app_builder.cpp', 'elf_object.cpp', 'sce_module_writer.cpp', 'self_container.cpp')],
        '-I', zlib/'include', zlib/'lib/libz.a', '-o', converter)
    run('native-convert', converter, 'link', '--in', pie, '--out', elf, '--stub-dir', libs, '--stub',stubs[0],'--stub',stubs[1],'--file-name','eboot.elf')
    symbols = {}
    symbol_lines = run('native-symbols', 'llvm-nm-18', '-n', pie).splitlines()
    for line in symbol_lines:
        parts = line.split()
        if len(parts) == 3:
            symbols[parts[2]] = int(parts[0], 16)
    expected_bundle = (out/'ps5/routes.bundle').read_bytes()
    audits = {'raw_pie': audit_image(pie, symbols, expected_bundle),
              'converted_eboot': audit_image(elf, symbols, expected_bundle, native=True)}
    modules, libraries, imports = read_imports(elf)
    (out/'native-imports.json').write_text(json.dumps({'modules': modules, 'libraries': libraries, 'symbols': imports}, indent=2)+'\n')
    named_imports = {parts[1]: imports[nid(parts[1])] for line in symbol_lines
                     if len(parts := line.split()) == 2 and parts[0] == 'U' and nid(parts[1]) in imports}
    require(len(named_imports) == len(imports), 'unresolved native import identity')
    run('native-agc-providers','python3',ROOT/'tests/native_agc_imports_test.py',elf)
    require(not any(name.startswith('hip') or name.startswith('__hip') for name in named_imports),'HIP ABI unexpectedly imported')
    (out/'native-import-names.json').write_text(json.dumps({'eboot_sha256': sha(elf),
        'native_executed': False, 'imports': named_imports}, indent=2)+'\n')
    after = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    require(before == after, 'shared native inputs changed')
    require(all(sha(native_libs/n)==v for n,v in sdk_inputs.items()),'native SDK libraries changed')
    return {'audits':audits,'native_source_revision':lock['native_tools']['revision'],'native_inputs_before_after':before,'sdk_inputs':sdk_inputs,'imports':len(named_imports),'HIP_imports':0,'native_executed':False}

def runtime_proof(ref,out,stage,execute):
    from hip_routes_recipe_test import validate_model
    lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
    common=['-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-I',ROOT/'include','-I',stage/'include','-I',platform]
    core=[ROOT/'src'/n for n in ('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp','hip_runtime.cpp')]
    sources=[*core,ROOT/'tools/hip_routes_control.cpp',ROOT/'tests/hip_routes_model.cpp',ROOT/'src/agc_backend.cpp',ROOT/'tools/probe_host_registration.py',ROOT/'tests/native_import_libraries_test.py',ROOT/'tests/native_agc_imports_test.py',ROOT/'tests/hip_routes_recipe_test.py',ROOT/'tests/hip_routes_contract_test.cpp',*list((ROOT/'include/ps5hip').glob('*.h'))]
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    linux_script=out/'linux.ld';linux_script.write_text('''SECTIONS {
 .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); }
} INSERT AFTER .rodata;
SECTIONS {
 .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); }
} INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata));
HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array));
HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
    contract=out/'routes-contract'
    execute('routes-contract-build',['clang++-18',*common,'-fsanitize=address,undefined','-fno-omit-frame-pointer',ROOT/'src/code_object.cpp',ROOT/'src/host_launch.cpp',ROOT/'src/launch.cpp',ROOT/'tools/hip_routes_control.cpp',ROOT/'tests/hip_routes_contract_test.cpp','-o',contract])
    for host in ('linux','ps5','linux-model'):execute('routes-contract-'+host,[contract,out/host/'routes.hsaco'])
    results=[]
    for partial in (False,True):
        binary=out/('model-partial' if partial else 'model-routes')
        execute(binary.name+'-build',['clang++-18',*common,'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1',*(['-DPS5HIP_RUNTIME_PARTIAL=1'] if partial else []),'-fPIE','-pie','-fuse-ld=lld-18',*core,ROOT/'tools/hip_routes_control.cpp',ROOT/'tests/hip_routes_model.cpp',out/'linux-model/routes.o','-Wl,-T,'+str(linux_script),'-o',binary])
        listing,_=execute(binary.name+'-symbols',['llvm-nm-18','-S','-n',binary])
        entries=[p for line in listing.splitlines() if len(p:=line.split()) in (3,4)]
        symbols={p[-1]:int(p[0],16) for p in entries}
        begin=symbols['__ps5hip_wrappers_begin']
        require(begin%32==0 and symbols['__hip_fatbin_wrapper']==begin and symbols['__ps5hip_wrappers_end']==begin+24,'model wrapper payload bounds changed')
        require([int(p[1],16) for p in entries if len(p)==4 and p[-1]=='__hip_fatbin_wrapper']==[64],'model wrapper redzone not retained')
        from probe_host_registration import load_regions
        reader,_,_=load_regions(binary)
        reader(begin,64) # Full instrumented object must remain mapped, including its redzone.
        modes=['partial'] if partial else ['normal','preflush','init','missing','corrupt','input','guard','stale-second-round','wrong-route-output','scratch-map','scratch-submit','scratch-kick','scratch-timeout','scratch-throw','scratch-unknown','scratch-postflush','scratch-cleanup','late-partial']
        for mode in modes:
            code=30 if mode.startswith('scratch-') and mode!='scratch-map' else 20 if mode in ('init','missing','corrupt','input','guard','stale-second-round','wrong-route-output','scratch-map') else 0
            stdout,stderr=execute('model-'+mode,[binary,mode],expected=code);validate_model(stdout+stderr,mode,code);results.append({'mode':mode,'exit_code':code})
    require(before=={str(p.relative_to(ROOT)):sha(p) for p in sources},'runtime sources changed during model proof')
    return core,common,before,results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--compiler-only',action='store_true');p.add_argument('--host-only',action='store_true');p.add_argument('--title-entry',action='store_true');args=p.parse_args()
    require(not args.title_entry or args.compiler_only, '--title-entry requires --compiler-only')
    ref=args.reference_root.resolve();out=args.output.resolve();require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    commands=[];inputs={};cases=[]
    def run(label,argv,expected=0):
        q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'),ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode,'environment':{'ASAN_OPTIONS':'protect_shadow_gap=0','LLVM_CONFIG':shutil.which('llvm-config-18')}});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(q.returncode==expected,label+' failed: '+q.stderr[-3000:]);return q.stdout,q.stderr
    try:
        fixture,stage,libs,inputs=prepared_inputs(ref)
        tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')};require(all(tools.values()),'LLVM18 tools missing')
        versions={n:run(n+'-version',[path,'--version'])[0] for n,path in tools.items()};require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version']);tool_hashes={n:sha(pathlib.Path(path)) for n,path in tools.items()}
        sdk=(ref/'.deps/sdk').resolve();lock=json.loads((ROOT/'dependencies.json').read_text());require((sdk/'.ps5-sdk-revision').read_text().strip()==lock['sdk']['revision'],'SDK revision changed')
        wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
        source=ROOT/'tests/hip_routes.hip';source_hashes={str(x.relative_to(ROOT)):sha(x) for x in (source,pathlib.Path(__file__),ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py',ROOT/'docs/HIP_ROUTES_COMPILER.md')}
        for host in ('linux','ps5','linux-model'):
            directory=out/host;directory.mkdir();combined=directory/'routes.o'
            compiler=wrapper if host=='ps5' else tools['clang++']
            cmd=[compiler,'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),'--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-fno-fast-math','-c',source,'--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',combined]
            if args.title_entry:cmd+=['-DPS5HIP_ROUTES_TITLE=1','-Dmain=ps5hip_routes_client_main']
            if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
            if host=='linux-model':cmd+=['-DPS5HIP_RUNTIME_MODEL=1','-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
            _,trace=run(host+'-driver',cmd+['-###']);require_device_linker(trace,tools['ld.lld']);run(host+'-compile',cmd)
            bundle=section(combined.read_bytes(),b'.hip_fatbin');(directory/'routes.bundle').write_bytes(bundle);magic=b'__CLANG_OFFLOAD_BUNDLE__';require(bundle.startswith(magic),'genuine bundle missing');count=struct.unpack_from('<Q',bundle,len(magic))[0];at=len(magic)+8;found=[];targets=[];require(count==2,'bundle inventory')
            for _ in range(count):
                offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24;identity=bundle[at:at+length].decode();at+=length;require(offset<=len(bundle) and size<=len(bundle)-offset,'bundle extent');targets.append(identity)
                if identity=='hipv4-amdgcn-amd-amdhsa--gfx1013':found.append(bundle[offset:offset+size])
            require(len(found)==1,'device selection');code=directory/'routes.hsaco';code.write_bytes(found[0])
            meta,_=run(host+'-metadata',[tools['llvm-readobj'],'--notes',code]);isa,_=run(host+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code]);audit=audit_object(code.read_bytes(),meta,isa)
            # Save observed profiles before applying the prospective admission gate.
            (directory/'observed.json').write_text(json.dumps(audit,indent=2)+'\n');check_profiles(audit)
            hostread,_=run(host+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',combined]);disasm,_=run(host+'-host-disassembly',[tools['llvm-objdump'],'-dr',combined]);nm,_=run(host+'-undefined',[tools['llvm-nm'],'--undefined-only',combined])
            require(all(n in nm for n in ('__hipRegisterFatBinary','__hipRegisterFunction','__hipUnregisterFatBinary','__hipPushCallConfiguration','__hipPopCallConfiguration','hipLaunchKernel','atexit')),'genuine registration/launch references missing')
            if args.title_entry:
                require(re.search(r'^    Name: _Z25ps5hip_routes_client_mainiPPc \(',hostread,re.M) is not None and re.search(r'^    Name: main \(',hostread,re.M) is None,'renamed title entry missing or bare main retained')
            wrapper_bytes=section(combined.read_bytes(),b'.hipFatBinSegment');wrapper_audit=audit_wrapper(wrapper_bytes,hostread,host)
            lifecycle=audit_registration(disasm)
            cases.append({'host':host,'target':'gfx1013','optimization':'O2','bundle_targets':targets,'wrapper_hex':wrapper_bytes.hex(),'wrapper_audit':wrapper_audit,'lifecycle':lifecycle,**audit,'files':{x.name:{'sha256':sha(x),'bytes':x.stat().st_size} for x in directory.iterdir() if x.is_file()}})
        device_relation={'whole_file_identical':len({c['files']['routes.hsaco']['sha256'] for c in cases})==1,'text_sha256':{c['host']:hashlib.sha256(section((out/c['host']/'routes.hsaco').read_bytes(),b'.text')).hexdigest() for c in cases},'statement':'Each Linux/PS5/model artifact separately inspected and admitted by actual-object tests; distinct whole ELF identities are never assumed equivalent.'}
        device_relation['text_identical']=len(set(device_relation['text_sha256'].values()))==1
        models=[];native={};runtime_sources={}
        if not args.compiler_only:
            core,common,runtime_sources,models=runtime_proof(ref,out,stage,run)
            if not args.host_only:native=native_proof(ref,out,core,common,run,inputs)
        require(all(sha(ROOT/n)==v for n,v in runtime_sources.items()),'runtime sources changed during native proof')
        require(all(sha(pathlib.Path(n))==v for n,v in inputs.items()),'inputs changed');require(all(sha(ROOT/n)==v for n,v in source_hashes.items()),'sources changed');require(all(sha(pathlib.Path(tools[n]))==v for n,v in tool_hashes.items()),'tools changed')
        receipt={'schema':1,'classification':'genuine-single-TU-four-route-offline-proof-pass','models':models,'native':native,'runtime_sources':runtime_sources,'device_relation':device_relation,'hardware_validated':False,'title_entry':args.title_entry,'source_sha256':source_hashes,'inputs':inputs,'tools':{n:{'path':tools[n],'sha256':tool_hashes[n],'version':versions[n]} for n in tools},'cases':cases,'limits':['No GPU execution: host callback arithmetic is not ISA emulation; native proof is compile/link only.','LLVM18.1.8/full pinned6.2.4 headers and exact workitem/constants only; no arbitraryROCm support.','InstalledSDK/resource support trees not fully inventoried; no hermetic SDK claim.']}
        receipt['artifacts']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Genuine Linux/PS5 four-route compiler proof passed; noGPU execution.')
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'commands':commands,'inputs_before':inputs,'source_sha256_before':locals().get('source_hashes',{}),'hardware_validated':False},indent=2)+'\n');raise
if __name__=='__main__':main()
