#!/usr/bin/env python3
"""Genuine tiled-matmul compiler and owned-runtime model proof; no GPU execution."""
import argparse,hashlib,json,os,pathlib,shutil,struct,subprocess,re
import probe_hip_routes as shared
from probe_hip_routes import ROOT,require
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
from probe_geometry_lds import audit_host,audit_cfg,oracle_cases
from probe_hip_generic import CORE, COMPILER_HEADERS
def compiler_sources():
    names=('dependencies.json','tests/hip_tiled_matmul_blockdim.hip','tools/probe_hip_matmul.py','tools/probe_geometry_lds.py','tools/probe_hip_generic.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py')
    return [ROOT/n for n in names]+[ROOT/'include/ps5hip'/n for n in COMPILER_HEADERS]
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section
NAME='tiled_matmul'
SHAPES=((8,8,8),(13,21,19),(1,1,17),(17,9,1))
TERMINAL=('submit','kick','timeout','throw','unknown','cleanup')
CLEAN=('shape','dynamic','alloc1','alloc2','alloc3','map1','map2','map3','init','preflush','missing','corrupt','input','input-b','guard','tail')
MODES=('normal','failed-module','late-partial')+CLEAN+TERMINAL

def shader(text):
    start=text.index('extern "C" __global__ __launch_bounds__(64)')
    end=text.index('\n}\n',start)+3
    return text[start:end]

def check_profile(audit):
    require(set(audit['descriptors'])=={NAME},'kernel inventory')
    d=audit['descriptors'][NAME];ops=audit['isa'][NAME]['opcodes']
    require(d['properties']==0x409 and d['rsrc2']==0x98c and d['group_bytes']==512 and d['kernarg_bytes']==296,'measured geometry/LDS profile')
    require(not any(d[k] for k in ('private_bytes','rsrc3','preload')),'unexpected resources')
    require('s_barrier' in ops and any(x.startswith('ds_read') for x in ops) and any(x.startswith('ds_write') for x in ops),'shared/barrier instructions missing')
    require(not audit['isa'][NAME]['flat_scratch_setup'] and not any(x.startswith(('flat_','scratch_','s_swappc')) for x in ops),'unsupported memory/calls')

def validate_model(text,mode,code):
    require(mode in MODES,'unknown model mode')
    require(not any(x in text for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:')),'sanitizer diagnostic')
    require(text.count('VECTOR_FUNCTION bound=true\n')==1,'genuine function registration')
    rows=re.findall(r'^MATMUL_COMPLETE shape=(\d+) round=(\d+) m=(\d+) n=(\d+) k=(\d+) exact=true guards=true inputs=unchanged$',text,re.M)
    full=[tuple(map(str,(index,round,*shape))) for index,shape in enumerate(SHAPES) for round in range(2)]
    require(rows==(full if mode in ('normal','late-partial') else []),'completion order/count')
    require(text.count('MATMUL_COMPLETE')==len(rows),'malformed completion')
    if mode in TERMINAL:
        marker=f'MATMUL_TERMINAL live={4 if mode=="cleanup" else 7} unresolved={0 if mode=="cleanup" else 1} submits=1'
        require(code==30 and text.splitlines().count(marker)==1,'terminal retained ownership')
        require(not any(x in text for x in ('MATMUL_CLIENT_PASS','MATMUL_MODEL_FINAL','MATMUL_CLIENT_FAILURE')),'terminal false closure')
        return
    require(code==(20 if mode in CLEAN else 0),'model exit')
    final=re.findall(r'^MATMUL_MODEL_FINAL live=(\d+) backends=(\d+) submits=(\d+) releases=(\d+) finalized=(\d+)$',text,re.M)
    require(len(final)==1 and final[0][0]=='0','live ownership/final audit')
    before_backend=mode in ('failed-module','shape','dynamic','alloc1','alloc2','alloc3','map1','map2','map3')
    expected_submits=8 if mode in ('normal','late-partial') else 0 if before_backend or mode in ('init','preflush') else 1
    require(final[0][1]==str(int(not before_backend)) and final[0][2]==str(expected_submits),'backend/submit counts')
    require(final[0][4]==str(int(mode not in ('failed-module','late-partial','init'))),'ownership finalization')
    completed=expected_submits
    require(re.findall(r'^RUNTIME_COMPLETE sequence=(\d+) kernel=tiled_matmul blocking=true$',text,re.M)==[str(i) for i in range(1,completed+1)],'actual runtime completion observations')
    require(text.splitlines().count('VECTOR_UNREGISTER after_main=true active=0')==1,'genuine destructor unregister')
    expected=f'VECTOR_FINAL registered=1 removed=1 completed={completed} active=0 terminal=0 main_finished=1 hardware_validated=false'
    require(text.splitlines().count(expected)==1,'runtime destruction record')

    if mode=='normal':
        require(text.splitlines().count('MATMUL_CLIENT_PASS cases=4 rounds=2 dispatches=8 values=982 exact_fp32_integer_oracle=true')==1,'application oracle')
    else:
        require('MATMUL_CLIENT_PASS' not in text,'false application pass')
        if mode in CLEAN:require(text.splitlines().count('MATMUL_CLIENT_FAILURE clean=1 terminal=0')==1,'clean unwind')
        elif mode=='failed-module':require('MATMUL_FAILED_MODULE latched=true no_gpu=true' in text,'registration latch')
        else:require('MATMUL_LATE_PARTIAL completed=8 finalized=false' in text,'late registration latch')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--compiler-only',action='store_true');p.add_argument('--title-entry',action='store_true');args=p.parse_args()
    require(not args.title_entry or args.compiler_only,'title-entry requires compiler-only')
    ref=args.reference_root.resolve();out=args.output.resolve();require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    inputs={};commands=[];cases=[]
    source='tests/hip_tiled_matmul_blockdim.hip'
    names=[source,'tests/hip_tiled_matmul_model.cpp','tests/hip_matmul_recipe_test.py','tools/probe_hip_matmul.py','tools/probe_geometry_lds.py','tools/probe_hip_generic.py','tools/probe_hip_routes.py','tools/probe_host_registration.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py','docs/HIP_TILED_MATMUL_CLIENT.md']
    names += ['src/'+n for n in CORE]+['src/hip_launch_profile.cpp']+[str(p.relative_to(ROOT)) for p in (ROOT/'include/ps5hip').glob('*.h')]
    if args.compiler_only:
        names=[str(p.relative_to(ROOT)) for p in compiler_sources()]
    sources={n:sha(ROOT/n) for n in names}
    def run(label,argv,expected=0):
        argv=list(map(str,argv));z=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=180,
            env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'),ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(z.stdout);(out/(label+'.stderr')).write_text(z.stderr)
        commands.append({'label':label,'argv':argv,'exit_code':z.returncode,'expected':expected,'ASAN_OPTIONS':'protect_shadow_gap=0'});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(z.returncode==expected,label+' failed: '+z.stderr[-2000:]);return z.stdout,z.stderr
    try:
        _,stage,libs,inputs=shared.prepared_inputs(ref)
        frozen=ref/'.local/g4-geometry-lds/tests/hip_tiled_matmul_blockdim.hip'
        require(sha(frozen)=='45193c130f9a0c044da0a5ab2e9a67ad2a2e2a0a522b9b67520de16900b9e377','frozen measured source')
        inputs[str(frozen)]=sha(frozen)
        require(shader((ROOT/source).read_text())==shader(frozen.read_text()),'measured shader body changed')
        tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')};require(all(tools.values()),'LLVM18 tools missing')
        versions={n:run(n+'-version',[path,'--version'])[0] for n,path in tools.items()};require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version'])
        tool_hashes={n:sha(path) for n,path in tools.items()}
        sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
        inputs[str(sdk/'.ps5-sdk-revision')]=sha(sdk/'.ps5-sdk-revision')
        require((sdk/'.ps5-sdk-revision').read_text().strip()==json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision'],'SDK revision changed')
        for host in ('linux','ps5','linux-model'):
            for optimization in (('O2',) if host=='linux-model' else ('O1','O2')):
                label=host+'-'+optimization;directory=out/label;directory.mkdir();combined=directory/'matmul.o'
                cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),
                    '--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-'+optimization,
                    '-fno-fast-math','-ffp-contract=off','-c',ROOT/source,'--hip-path='+str(stage),'-I'+str(stage/'include'),
                    '-I'+str(ROOT/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc',
                    '--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',combined]
                if args.title_entry:cmd+=['-DPS5HIP_MATMUL_TITLE=1','-Dmain=ps5hip_matmul_client_main']
                if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
                if host=='linux-model':cmd+=['-DPS5HIP_RUNTIME_MODEL=1','-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
                _,driver=run(label+'-driver',cmd+['-###']);require_device_linker(driver,tools['ld.lld']);run(label+'-compile',cmd)
                bundle=section(combined.read_bytes(),b'.hip_fatbin');(directory/'matmul.bundle').write_bytes(bundle)
                magic=b'__CLANG_OFFLOAD_BUNDLE__';require(bundle.startswith(magic),'genuine bundle missing');count=struct.unpack_from('<Q',bundle,len(magic))[0];require(count==2,'bundle target count')
                at=len(magic)+8;selected=[];targets=[]
                for _ in range(count):
                    offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24;identity=bundle[at:at+length].decode();at+=length
                    require(offset<=len(bundle) and size<=len(bundle)-offset,'bundle extent');targets.append(identity)
                    if identity=='hipv4-amdgcn-amd-amdhsa--gfx1013':selected.append(bundle[offset:offset+size])
                require(len(selected)==1,'device target selection');code=directory/'matmul.hsaco';code.write_bytes(selected[0])
                meta,_=run(label+'-metadata',[tools['llvm-readobj'],'--notes',code]);isa,_=run(label+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code])
                audit=shared.audit_object(code.read_bytes(),meta,isa,(NAME,));(directory/'observed.json').write_text(json.dumps(audit,indent=2)+'\n')
                check_profile(audit)
                readobj,_=run(label+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',combined]);disasm,_=run(label+'-host-isa',[tools['llvm-objdump'],'-dr',combined])
                if args.title_entry:require(re.search(r'^    Name: _Z25ps5hip_matmul_client_mainiPPc \(',readobj,re.M) and not re.search(r'^    Name: main \(',readobj,re.M),'renamed matmul entry missing')
                lifecycle=audit_host(disasm);wrapper_audit=shared.audit_wrapper(section(combined.read_bytes(),b'.hipFatBinSegment'),readobj,host)
                # Caller CFG is separate from the final linked AMD ISA; library linkage is not inferred.
                # Retain -c: the SDK wrapper otherwise appends crt1.o, creating a
                # second input during the diagnostic -S -emit-llvm operation.
                ir=directory/'caller.ll';ir_cmd=cmd.copy();ir_cmd[ir_cmd.index('-o')+1]=ir
                ir_cmd+=['--cuda-device-only','--no-gpu-bundle-output','-nogpulib','-S','-emit-llvm']
                run(label+'-caller-ir',ir_cmd);cfg=audit_cfg(ir.read_text());(directory/'cfg.json').write_text(json.dumps(cfg,indent=2)+'\n')
                ops=audit['isa'][NAME]['opcodes'];descriptor=audit['descriptors'][NAME]
                require('s_barrier' in ops and any(x.startswith('ds_read') for x in ops) and any(x.startswith('ds_write') for x in ops),'actual shared/barrier ISA missing')
                require(descriptor['group_bytes']>0,'compiler removed static LDS')
                cases.append({'host':host,'optimization':optimization,'target':'gfx1013','bundle_targets':targets,'lifecycle':lifecycle,'wrapper':wrapper_audit,
                    **audit,'caller_cfg':{'blocks':len(cfg['blocks']),'barrier_sites':cfg['barrier_sites'],'stage':cfg['stage']},
                    'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in directory.iterdir() if p.is_file()}})
                print(label,descriptor,flush=True)
        (out/'oracle-cases.json').write_text(json.dumps(oracle_cases(),indent=2)+'\n')
        models=[]
        if not args.compiler_only:
            platform=ref/'.deps'/('sdk-source-'+json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision'])/'platform/include'
            script=out/'linux.ld';script.write_text("""SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
    SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
    HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
    HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
    """)
            binary=out/'model'
            run('model-build',[tools['clang++'],'-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-I',ROOT/'include','-I',stage/'include','-I',platform,'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1','-fPIE','-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE],ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/hip_tiled_matmul_model.cpp',out/'linux-model-O2/matmul.o','-Wl,-T,'+str(script),'-o',binary])
            for mode in MODES:
                expected=30 if mode in TERMINAL else 20 if mode in CLEAN else 0
                stdout,stderr=run('model-'+mode,[binary,mode],expected)
                validate_model(stdout+stderr,mode,expected);models.append({'mode':mode,'exit_code':expected})
        require(all(sha(ROOT/n)==h for n,h in sources.items()),'sources changed')
        require(all(sha(n)==h for n,h in inputs.items()),'inputs changed');require(all(sha(tools[n])==h for n,h in tool_hashes.items()),'tools changed')
        report={'schema':2,'classification':'genuine-tiled-matmul-runtime-offline-pass','title_entry':args.title_entry,'compiler_only':args.compiler_only,'sources':sources,'inputs':inputs,
            'tools':{n:{'path':tools[n],'sha256':tool_hashes[n],'version':versions[n]} for n in tools},
            'commands':commands,'cases':cases,'models':models,'shader_body_sha256':__import__('hashlib').sha256(shader((ROOT/source).read_text()).encode()).hexdigest(),
            'hardware_validated':False,'native_linked':False,
            'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
            'limits':['Host callbacks validate transport and ownership, not AMDGPU instruction execution.',
              'Four finite integer-valued FP32 shapes do not qualify general floating-point distributions.',
              'Caller remains responsible for shader indexing bounds. Single module and serialized blocking runtime.',
              'PS5 combined-object compilation is not native runtime linkage or execution.',
              'SDK/resource support trees not fully inventoried; no hermetic claim.']}
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('MEASURED '+sha(out/'receipt.json'),flush=True)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'commands':commands,'sources':sources,'inputs_before':inputs,'hardware_validated':False},indent=2)+'\n');raise
if __name__=='__main__':main()
