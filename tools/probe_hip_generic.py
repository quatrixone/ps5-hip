#!/usr/bin/env python3
"""Genuine variable-count HIP client, bounded runtime model and native link proof."""
import argparse, hashlib, json, os, pathlib, re, shutil, struct, subprocess
import probe_hip_routes as shared
from probe_hip_routes import ROOT, sha, require, prepared_inputs, audit_object, audit_registration, audit_wrapper, native_proof
from build_canaries import require_device_linker, require_tool_versions
from fat_binary_fixture_test import section

CORE=('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp','hip_runtime.cpp')
# Transitive local includes of hip_runtime_control.h in a compiled HIP title.
COMPILER_HEADERS=('hip_runtime_control.h','native_launch.h','submission.h','fat_binary.h','launch.h','native_data.h','native_memory.h','code_object.h')
def compiler_sources(title_entry, memory_fill=False):
    sources=[ROOT/n for n in ('dependencies.json','tests/hip_generic.hip','tests/hip_generic_second.hip','tools/probe_hip_generic.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py')]
    if memory_fill:sources.append(ROOT/'tests/hip_memory_fill_client.h')
    return sources+([ROOT/'include/ps5hip'/n for n in COMPILER_HEADERS] if title_entry else [])
NAMES=('affine_words','block_sum')

def validate_model(text,mode,code,functions):
    require(not any(x in text for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer')), 'sanitizer diagnostic')
    terminal=mode in ('submit','kick','timeout','throw','unknown','cleanup')
    clean=mode in ('init','preflush','missing','corrupt','input','guard')
    count={'zero':0,'one':1,'three':3}.get(mode,7)
    if terminal:
        require(code==30 and f'GENERIC_TERMINAL live={3 if mode=="cleanup" else 6} unresolved={0 if mode=="cleanup" else 1} submits=1' in text,'terminal ownership')
        require('GENERIC_CLIENT_PASS' not in text and 'GENERIC_MODEL_FINAL' not in text,'terminal false closure')
        return
    require(code==(20 if clean else 0),'model exit')
    final=re.findall(r'^GENERIC_MODEL_FINAL live=(\d+) backends=(\d+) submits=(\d+) releases=(\d+) finalized=(\d+)$',text,re.M)
    require(len(final)==1 and final[0][0]=='0','final live ownership')
    # Finalized means safe lifecycle closure, never that the application's oracle passed.
    expected_final=mode not in ('init','partial','second-module','failed-module','late-partial')
    require(final[0][4]==str(int(expected_final)),'false finalized status')
    if mode in ('partial','second-module'):
        require('GENERIC_PARTIAL no_gpu=true' in text and final[0][1:3]==('0','0'),'partial registration cleanup')
    elif mode=='failed-module':
        require('GENERIC_FAILED_MODULE latched=true no_gpu=true' in text and final[0][1:3]==('0','0'),'failed module latch')
    elif mode=='late-partial':
        require('GENERIC_LATE_PARTIAL completed=7 finalized=false' in text and final[0][2]=='7','late registration latch')
    elif clean:
        require('GENERIC_CLIENT_FAILURE clean=1 terminal=0' in text and 'GENERIC_CLIENT_PASS' not in text,'clean failure')
    else:
        require(f'GENERIC_CLIENT_PASS launches={count} variable_sizes=true full_guards=true unused_function_allowed=true' in text,'client result')
        require(final[0][1]==str(int(count>0)) and final[0][2]==str(count),'lazy backend/count')
    require(text.count('VECTOR_FUNCTION bound=true\n')==functions or mode in ('partial','second-module'),'genuine function inventory')

def check_profiles(audit,names):
    for name,d in audit['descriptors'].items():
        require(d['properties']==0x409 and d['rsrc2']==0x8c and not d['rsrc3'] and not d['preload'] and not d['private_bytes'],'unsupported entry '+name)
        require(d['group_bytes']==(256 if name=='block_sum' else 0),'group bytes '+name)
        require(d['kernarg_bytes']=={'affine_words':32,'block_sum':20,'deliberately_unused':12}[name],'argument layout '+name)
        ops=audit['isa'][name]['opcodes']
        require(not audit['isa'][name]['flat_scratch_setup'] and not any(o.startswith(('flat_','scratch_','s_swappc')) for o in ops),'unsupported memory/call ISA')
        if name=='block_sum':require('s_barrier' in ops and any(o.startswith('ds_') for o in ops),'LDS not emitted')
    require(set(audit['descriptors'])==set(names),'profile inventory')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--compiler-only',action='store_true');p.add_argument('--host-only',action='store_true');p.add_argument('--title-entry',action='store_true');p.add_argument('--memory-fill',action='store_true');args=p.parse_args()
    require(not args.memory_fill or args.title_entry,'memory-fill requires title entry')
    require(not args.title_entry or args.compiler_only,'title-entry requires compiler-only')
    ref=args.reference_root.resolve();out=args.output.resolve();require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    commands=[];inputs={}
    def run(label,argv,expected=0):
        q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'),ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr)
        commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode,'expected':expected,'environment':{'ASAN_OPTIONS':'protect_shadow_gap=0'}})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(q.returncode==expected,label+' failed: '+q.stderr[-3000:]);return q.stdout,q.stderr
    sources=[ROOT/'src'/n for n in CORE]+[ROOT/'src/hip_launch_profile.cpp',ROOT/'src/agc_backend.cpp',ROOT/'tests/hip_generic.hip',ROOT/'tests/hip_generic_model.cpp',ROOT/'tests/hip_launch_profile_test.cpp',pathlib.Path(__file__),ROOT/'tools/probe_hip_routes.py',ROOT/'tools/probe_host_registration.py',ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py',ROOT/'tests/native_import_libraries_test.py',ROOT/'tests/native_agc_imports_test.py',*list((ROOT/'include/ps5hip').glob('*.h'))]
    sources += [ROOT/'tests/hip_generic_second.hip',ROOT/'tests/hip_generic_recipe_test.py',ROOT/'docs/BOUNDED_HIP_RUNTIME.md']
    if args.compiler_only:
        sources=compiler_sources(args.title_entry,args.memory_fill)
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    try:
        _,stage,libs,inputs=prepared_inputs(ref);lock=json.loads((ROOT/'dependencies.json').read_text())
        tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')};require(all(tools.values()),'LLVM18 missing')
        versions={n:run(n+'-version',[s,'--version'])[0] for n,s in tools.items()};require_tool_versions(versions,lock['llvm']['version']);tool_hashes={n:sha(pathlib.Path(s)) for n,s in tools.items()}
        sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
        inputs[str(sdk/'.ps5-sdk-revision')]=sha(sdk/'.ps5-sdk-revision')
        require((sdk/'.ps5-sdk-revision').read_text().strip()==lock['sdk']['revision'],'SDK revision')
        cases=[]
        for extra in (False,True):
            names=NAMES+(('deliberately_unused',) if extra else ())
            for host in ('linux','ps5','linux-model'):
                label=host+('-extra' if extra else '');d=out/label;d.mkdir();obj=d/'routes.o'
                cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),'--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-fno-fast-math','-c',ROOT/'tests/hip_generic.hip','--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
                if extra:cmd+=['-DPS5HIP_GENERIC_EXTRA_KERNEL=1']
                if args.title_entry:cmd+=['-DPS5HIP_GENERIC_TITLE=1','-Dmain='+('ps5hip_memory_fill_client_main' if args.memory_fill else 'ps5hip_generic_client_main'),'-I'+str(ROOT/'include')]
                if args.memory_fill:cmd+=['-DPS5HIP_MEMORY_FILL_TITLE=1']
                if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
                if host=='linux-model':cmd+=['-DPS5HIP_RUNTIME_MODEL=1','-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
                _,trace=run(label+'-driver',cmd+['-###']);require_device_linker(trace,tools['ld.lld']);run(label+'-compile',cmd)
                bundle=section(obj.read_bytes(),b'.hip_fatbin');(d/'routes.bundle').write_bytes(bundle)
                magic=b'__CLANG_OFFLOAD_BUNDLE__';require(bundle.startswith(magic),'bundle magic');count=struct.unpack_from('<Q',bundle,len(magic))[0];require(count==2,'bundle count');at=len(magic)+8;found=[]
                for _ in range(count):
                    offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24;identity=bundle[at:at+length].decode();at+=length;require(offset<=len(bundle) and size<=len(bundle)-offset,'bundle extent')
                    if identity=='hipv4-amdgcn-amd-amdhsa--gfx1013':found.append(bundle[offset:offset+size])
                require(len(found)==1,'selected device');code=d/'routes.hsaco';code.write_bytes(found[0])
                metadata,_=run(label+'-metadata',[tools['llvm-readobj'],'--notes',code]);isa,_=run(label+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code]);audit=audit_object(code.read_bytes(),metadata,isa,names)
                (d/'observed.json').write_text(json.dumps(audit,indent=2)+'\n');check_profiles(audit,names)
                hostread,_=run(label+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',obj]);disasm,_=run(label+'-host-disassembly',[tools['llvm-objdump'],'-dr',obj])
                if args.title_entry:
                    require(re.search(r'^    Name: '+('_Z30ps5hip_memory_fill_client_mainiPPc' if args.memory_fill else '_Z26ps5hip_generic_client_mainiPPc')+r' \(',hostread,re.M) and not re.search(r'^    Name: main \(',hostread,re.M),'renamed generic title entry missing')
                cases.append({'profile':label,'kernels':names,'wrapper':audit_wrapper(section(obj.read_bytes(),b'.hipFatBinSegment'),hostread,host),'registration':audit_registration(disasm,names),**audit})
        # A separately compiled, real second HIP constructor is unsupported. Keep
        # both objects and their init arrays; no hand-written registration substitute.
        second=out/'second-module.o';second_cmd=list(cmd)
        second_cmd[second_cmd.index(ROOT/'tests/hip_generic.hip')]=ROOT/'tests/hip_generic_second.hip'
        second_cmd[second_cmd.index('-o')+1]=second
        second_cmd.remove('-DPS5HIP_GENERIC_EXTRA_KERNEL=1')
        run('second-module-compile',second_cmd)
        second_asm,_=run('second-module-disassembly',[tools['llvm-objdump'],'-dr',second])
        second_registration=audit_registration(second_asm,('second_module_kernel',))
        models=[];native={}
        if not args.compiler_only:
            platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';common=['-pthread','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-I',ROOT/'include','-I',stage/'include','-I',platform]
            core=[ROOT/'src'/n for n in CORE];policy=ROOT/'src/hip_launch_profile.cpp'
            script=out/'linux.ld';script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
            test=out/'profile-test';run('profile-build',['clang++-18',*common,'-fsanitize=address,undefined',ROOT/'tests/hip_launch_profile_test.cpp',policy,ROOT/'src/code_object.cpp',ROOT/'src/host_launch.cpp',ROOT/'src/launch.cpp','-o',test])
            legacy=ref/'.local/g4-routes-title/results/routes-standalone-regression-v1/linux/routes.hsaco';inputs[str(legacy)]=sha(legacy)
            run('profile-test',[test,*[out/c['profile']/'routes.hsaco' for c in cases],legacy])
            for extra,partial,two in ((False,False,False),(True,False,False),(False,True,False),(False,False,True)):
                label='model'+('-extra' if extra else '')+('-partial' if partial else '')+('-second' if two else '');binary=out/label
                run(label+'-build',['clang++-18',*common,'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1',*(['-DPS5HIP_RUNTIME_PARTIAL=1'] if partial else []),'-fPIE','-pie','-fuse-ld=lld-18',*core,policy,ROOT/'tests/hip_generic_model.cpp',out/('linux-model-extra' if extra else 'linux-model')/'routes.o',*([second] if two else []),'-Wl,-T,'+str(script),'-o',binary])
                modes=['second-module'] if two else ['partial'] if partial else ['zero','one','three','seven'] if extra else ['zero','one','three','seven','failed-module','late-partial','init','preflush','missing','corrupt','input','guard','submit','kick','timeout','throw','unknown','cleanup']
                for mode in modes:
                    expected=30 if mode in ('submit','kick','timeout','throw','unknown','cleanup') else 20 if mode in ('init','preflush','missing','corrupt','input','guard') else 0
                    stdout,stderr=run(label+'-'+mode,[binary,mode],expected);validate_model(stdout+stderr,mode,expected,3 if extra else 2);models.append({'variant':label,'mode':mode,'exit_code':expected})
            if not args.host_only:native=native_proof(ref,out,core,common,run,inputs,adapter=policy)
        require(before=={str(p.relative_to(ROOT)):sha(p) for p in sources},'source drift');require(all(sha(pathlib.Path(p))==v for p,v in inputs.items()),'input drift');require(all(sha(pathlib.Path(tools[n]))==v for n,v in tool_hashes.items()),'tool drift')
        receipt={'schema':2,'compiler_only':args.compiler_only,'memory_fill':args.memory_fill,'classification':'bounded-generic-runtime-offline-pass','title_entry':args.title_entry,'hardware_validated':False,'sources':before,'inputs':inputs,'tools':{n:{'path':tools[n],'sha256':v,'version':versions[n]} for n,v in tool_hashes.items()},'cases':cases,'second_module_registration':second_registration,'models':models,'native':native,'limits':['Host callbacks are not ISA emulation; native link is not execution.','Single thread/module; caller proves kernel indexing bounds.','SDK/resource/static host support trees not fully inventoried; not hermetic.']}
        receipt['artifacts']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Generic offline proof passed')
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'sources_before':before,'inputs_before':inputs,'commands':commands},indent=2)+'\n');raise

if __name__=='__main__':main()
