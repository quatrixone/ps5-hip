#!/usr/bin/env python3
"""Measure genuine geometry+static-LDS HIP objects. No runtime admission or GPU execution."""
import argparse,hashlib,json,os,pathlib,re,shutil,struct,subprocess
import probe_hip_routes as shared
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section
ROOT=pathlib.Path(__file__).resolve().parents[1]
NAME='tiled_matmul'
shared.NAMES=(NAME,)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(yes,message):
    if not yes:raise ValueError(message)

def audit_host(text):
    blocks={}
    for m in re.finditer(r'^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',text,re.M|re.S):blocks[m[2]]=m[3]
    require('__hip_module_ctor' in blocks and '__hip_module_dtor' in blocks,'genuine compiler lifecycle missing')
    def refs(body,name):return len(re.findall(r'R_X86_64_[A-Z0-9_]+\s+'+re.escape(name)+r'(?:-|\s|$)',body))
    ctor=blocks['__hip_module_ctor'];dtor=blocks['__hip_module_dtor']
    for name in ('__hipRegisterFatBinary','__hipRegisterFunction','atexit',NAME):require(refs(ctor,name)==1,'compiler binding count '+name)
    require(refs(dtor,'__hipUnregisterFatBinary')==1,'genuine compiler unregister missing')
    require(refs(text,'__hipPushCallConfiguration')>=1 and refs(text,'__hipPopCallConfiguration')>=1 and refs(text,'hipLaunchKernel')>=1,'ordinary HIP launch stub missing')
    return {'modules':1,'functions':1,'atexit':1,'unregister':1,'ordinary_launch_stub':True}

def audit_cfg(text):
    match=re.search(r'^define [^\n]*@'+NAME+r'\([^\n]*\)[^\n]*\{\n(.*?)^\}',text,re.M|re.S)
    require(match is not None,'kernel LLVM definition missing')
    blocks={};current='entry';lines=[]
    for line in match[1].splitlines():
        label=re.match(r'^([A-Za-z0-9$._-]+):(?:\s*;.*)?$',line)
        if label:
            if lines:require(current not in blocks,'duplicate CFG label');blocks[current]='\n'.join(lines)
            current=label[1];lines=[]
        else:lines.append(line)
    require(current not in blocks,'duplicate CFG label');blocks[current]='\n'.join(lines)
    report={}
    for name,body in blocks.items():
        successors=re.findall(r'\blabel %([A-Za-z0-9$._-]+)',body)
        require(all(target in blocks for target in successors),'CFG edge to missing block')
        report[name]={'successors':successors,'barriers':body.count('call void @llvm.amdgcn.s.barrier('),
                      'returns':bool(re.search(r'\bret void\b',body)),'instructions':body}
    return {'blocks':report,'barrier_sites':sum(b['barriers'] for b in report.values()),
            'stage':'optimized caller IR without device-library linkage',
            'barrier_uniformity_automatically_proven':False}

def oracle_cases():
    result=[]
    for m,n,k in ((8,8,8),(13,21,19),(1,1,17),(17,9,1)):
        rounds=[]
        for iteration in range(2):
            a=[(r*3+i*5+iteration*11)%7-3 for r in range(m) for i in range(k)]
            b=[(i*7+c*3+iteration*11)%5-2 for i in range(k) for c in range(n)]
            outputs=[sum(a[r*k+i]*b[i*n+c] for i in range(k)) for r in range(m) for c in range(n)]
            require(6*k<2**24,'FP32 exact integer range exceeded')
            rounds.append({'input_a':a,'input_b':b,'output_bits':[struct.unpack('<I',struct.pack('<f',x))[0] for x in outputs]})
        require(rounds[0]['input_a']!=rounds[1]['input_a'] and rounds[0]['input_b']!=rounds[1]['input_b'],'changed input rounds missing')
        gx,gy=(n+7)//8,(m+7)//8
        result.append({'m':m,'n':n,'k':k,'groups':[gx,gy,1],'threads':[8,8,1],
            'k_tiles':(k+7)//8,'logical_outputs':m*n,'inactive_threads':gx*gy*64-m*n,'rounds':rounds})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--variant',choices=('fixed','blockdim'),default='fixed');args=p.parse_args()
    ref=args.reference_root.resolve();out=args.output.resolve();require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    inputs={};commands=[];cases=[]
    source='tests/hip_tiled_matmul'+('_blockdim' if args.variant=='blockdim' else '')+'.hip'
    names=(source,'tests/geometry_lds_recipe_test.py','tools/probe_geometry_lds.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py')
    sources={n:sha(ROOT/n) for n in names}
    def run(label,argv):
        argv=list(map(str,argv));z=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=180,
            env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18')))
        (out/(label+'.stdout')).write_text(z.stdout);(out/(label+'.stderr')).write_text(z.stderr)
        commands.append({'label':label,'argv':argv,'exit_code':z.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(z.returncode==0,label+' failed: '+z.stderr[-2000:]);return z.stdout,z.stderr
    try:
        _,stage,libs,inputs=shared.prepared_inputs(ref)
        tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')};require(all(tools.values()),'LLVM18 tools missing')
        versions={n:run(n+'-version',[path,'--version'])[0] for n,path in tools.items()};require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version'])
        tool_hashes={n:sha(path) for n,path in tools.items()}
        sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
        require((sdk/'.ps5-sdk-revision').read_text().strip()==json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision'],'SDK revision changed')
        for host in ('linux','ps5'):
            for optimization in ('O1','O2'):
                label=host+'-'+optimization;directory=out/label;directory.mkdir();combined=directory/'matmul.o'
                cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),
                    '--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-'+optimization,
                    '-fno-fast-math','-ffp-contract=off','-c',ROOT/source,'--hip-path='+str(stage),'-I'+str(stage/'include'),
                    '-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc',
                    '--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',combined]
                if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
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
                audit=shared.audit_object(code.read_bytes(),meta,isa);(directory/'observed.json').write_text(json.dumps(audit,indent=2)+'\n')
                readobj,_=run(label+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',combined]);disasm,_=run(label+'-host-isa',[tools['llvm-objdump'],'-dr',combined])
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
        require(all(sha(ROOT/n)==h for n,h in sources.items()),'sources changed')
        require(all(sha(n)==h for n,h in inputs.items()),'inputs changed');require(all(sha(tools[n])==h for n,h in tool_hashes.items()),'tools changed')
        report={'schema':1,'classification':'genuine-geometry-lds-compiler-measurement','base_commit':'5b3c68443ff26aed46f4239208e4db25356403c1',
            'variant':args.variant,'source_sha256':sources,'inputs':inputs,'tools':{n:{'path':tools[n],'sha256':tool_hashes[n],'version':versions[n]} for n in tools},
            'commands':commands,'cases':cases,'runtime_admission_changed':False,'hardware_validated':False,
            'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
            'limits':['Compiler/host ABI measurement only; no native link, host callback or GPU execution.',
              'Caller IR CFG is not final device-linked IR and does not automatically prove barrier uniformity.',
              'No compiler profile is admitted by existing runtime from this receipt.',
              'Finite integer-valued FP32 oracle cases do not qualify general floating-point distributions.',
              'Installed SDK/resource support trees not fully inventoried; no hermetic claim.']}
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('MEASURED '+sha(out/'receipt.json'),flush=True)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'commands':commands,'source_sha256':sources,'inputs_before':inputs,'hardware_validated':False},indent=2)+'\n');raise
if __name__=='__main__':main()
