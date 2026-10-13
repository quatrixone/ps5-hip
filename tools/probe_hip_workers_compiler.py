#!/usr/bin/env python3
"""Compile the unchanged worker kernel for Linux/model/PS5; no native application."""
import argparse,hashlib,json,os,pathlib,re,shutil,subprocess
from probe_hip_routes import ROOT,require,prepared_inputs,audit_object,audit_registration,audit_wrapper
from probe_hipify_client import device_object
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=('thread_words',);source=ROOT/'tests/hip_threads.hip'
    require(sha(source)=='be54758c88a68b6d7725dddaff089d312641c5811eb7b52b8c2b0f47c0990077','worker shader changed')
    files=[source,pathlib.Path(__file__),*[ROOT/n for n in ('dependencies.json','tools/probe_hip_routes.py','tools/probe_hipify_client.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py')]]
    before={str(f.relative_to(ROOT)):sha(f) for f in files};commands=[]
    def run(label,argv):
        argv=list(map(str,argv));env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'))
        q=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr)
        commands.append({'label':label,'argv':argv,'exit_code':q.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(q.returncode==0,label+' failed: '+q.stderr[-2000:]);return q.stdout,q.stderr
    _,stage,libs,inputs=prepared_inputs(ref);lock=json.loads((ROOT/'dependencies.json').read_text())
    tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm')}
    require(all(tools.values()),'LLVM18 missing')
    versions={n:run(n+'-version',[s,'--version'])[0] for n,s in tools.items()}
    require_tool_versions(versions,lock['llvm']['version'])
    sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++'
    require((sdk/'.ps5-sdk-revision').read_text().strip()==lock['sdk']['revision'],'SDK revision')
    inputs[str(wrapper)]=sha(wrapper);inputs[str(sdk/'.ps5-sdk-revision')]=sha(sdk/'.ps5-sdk-revision')
    baseline=ref/'.local/g4-threaded-runtime/results/threads-v10/receipt.json'
    require(sha(baseline)=='a1123e8e9861ef0326f331f02992a7ba6d0bb138991e25f0fec1607755e8e94d','prior kernel proof changed')
    inputs[str(baseline)]=sha(baseline);expected=json.loads(baseline.read_text())['audit'];cases=[]
    for host in ('linux','ps5','linux-model'):
        d=out/host;d.mkdir();obj=d/'threads.o'
        cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),
             '--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-c',source,
             '--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),
             '--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
        if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
        if host=='linux-model':cmd+=['-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
        _,trace=run(host+'-driver',cmd+['-###']);require_device_linker(trace,tools['ld.lld']);run(host+'-compile',cmd)
        bundle=section(obj.read_bytes(),b'.hip_fatbin');(d/'threads.bundle').write_bytes(bundle)
        code=d/'threads.hsaco';code.write_bytes(device_object(bundle))
        metadata,_=run(host+'-metadata',[tools['llvm-readobj'],'--notes',code])
        isa,_=run(host+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code])
        audit=audit_object(code.read_bytes(),metadata,isa,names)
        require(audit==expected,'kernel descriptor/ISA differs from qualified host kernel')
        hostread,_=run(host+'-host-readobj',[tools['llvm-readobj'],'--sections','--symbols','--relocations',obj])
        disasm,_=run(host+'-host-disassembly',[tools['llvm-objdump'],'-dr',obj])
        require(re.search(r'^    Name: threads_launch \(',hostread,re.M) is not None,'C launch entry absent')
        require(re.search(r'^    Name: main \(',hostread,re.M) is None,'unexpected main entry')
        cases.append({'host':host,'audit':audit,'lifecycle':audit_registration(disasm,names),
                      'wrapper':audit_wrapper(section(obj.read_bytes(),b'.hipFatBinSegment'),hostread,host)})
    require(before=={str(f.relative_to(ROOT)):sha(f) for f in files},'compiler sources changed')
    require(all(sha(path)==value for path,value in inputs.items()),'compiler inputs changed')
    report={'schema':1,'classification':'genuine-worker-kernel-compiler-proof','sources':before,'inputs':inputs,
            'tools':{n:{'path':v,'sha256':sha(v)} for n,v in tools.items()},'cases':cases,
            'hardware_executed':False,'native_linked':False,'kernel_source_unchanged':True,
            'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))

if __name__=='__main__':main()
