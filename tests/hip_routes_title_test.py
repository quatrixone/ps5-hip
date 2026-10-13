#!/usr/bin/env python3
"""Actual renamed HIP client/compiler lifecycle under host-only title services."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UNCERTAIN={'scratch-submit','scratch-kick','scratch-timeout','scratch-throw','scratch-unknown','scratch-postflush'}
CLIENT_FAILURE={'init','missing','corrupt','input','guard','stale-second-round','wrong-route-output','scratch-map'}
CASES={'normal','preflush','partial','late-partial','scratch-cleanup',*UNCERTAIN,*CLIENT_FAILURE,'elevation','config','log-exists','startup-sync','final-sync','skip-unregister','reject-unregister','late-live','shell-fail','late-partial-destructor'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def verify_receipt(output):
    receipt=json.loads((output/'receipt.json').read_text())
    require(receipt['classification']=='genuine-four-route-title-host-lifecycle-pass' and receipt['native_executed'] is False and receipt['genuine_compiler_ctor_dtor'] is True and receipt['model_is_isa_emulation'] is False,'title proof classification')
    for name,digest in receipt['source_hashes'].items():require(sha(ROOT/name)==digest,'title model source changed: '+name)
    for name,digest in receipt['input_hashes'].items():require(sha(Path(name))==digest,'title model input changed: '+name)
    for tool in receipt['tools'].values():require(sha(Path(tool['path']))==tool['sha256'],'title model tool changed')
    for name,digest in receipt['artifacts'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'title model artifact escapes output')
        require(sha(output/path)==digest,'title model artifact changed: '+name)
    inputs=receipt['inputs']
    require(sha(Path(inputs['compiler_receipt']))==inputs['frozen_receipt_sha256'] and sha(Path(inputs['combined_object']))==inputs['combined_sha256'],'genuine compiler input changed')
    counts=check(output)
    require(counts==(receipt['cases'],receipt['transcript_mutations']),'title proof counts changed')
    return counts
def check(output):
    cases=json.loads((output/'cases.json').read_text());require(set(cases)==CASES,'case inventory')
    for name,record in cases.items():
        p=output/record['log'];require(sha(p)==record['sha256'],'case log changed');log=p.read_text()
        captured=(output/('run-'+name+'.stdout')).read_text()+(output/('run-'+name+'.stderr')).read_text()
        require(not any(x in log+captured for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','LeakSanitizer')),'sanitizer diagnostic '+name)
        success=name in ('normal','preflush','shell-fail')
        if success:
            events=['HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0']
            for round in (1,2):
                for route,kernel in enumerate(('integer','lds','spill','geometry')):
                    events += [f'RUNTIME_COMPLETE sequence={(round-1)*4+route+1} kernel=routes_{kernel} blocking=true',f'ROUTES_COMPLETE round={round} route={route} blocking=true full_guards=true inputs=unchanged']
            events += ['ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true','HIP_TITLE COMPUTE complete=true dispatches=8 word_comparisons=393216 finalization=pending','PS5HIP SHUTDOWN main-return=0','PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed','VECTOR_UNREGISTER after_main=true active=0','VECTOR_FINAL registered=1 removed=1 completed=8 active=0 terminal=0 main_finished=1','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=8','PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin']
            positions=[log.index(e) for e in events];require(positions==sorted(positions),'actual callback order '+name)
            require('QUARANTINED' not in log,'false success '+name)
            require(('request-return=0xfffffff9' if name=='shell-fail' else 'request-return=0') in log,'shell result')
        else:
            require('PS5HIP SHELL_EXIT request=begin' not in log,'failed title requested exit '+name)
            if name!='final-sync':require('HIP_TITLE FINAL ' not in log,'false final proof '+name)
        if name in UNCERTAIN|{'scratch-cleanup'}:
            live=4 if name=='scratch-cleanup' else 8
            require(f'ROUTES_CLIENT_TERMINAL_PASS mode={name} unresolved={int(name!="scratch-cleanup")} live={live} submits=3' in log,'terminal retention proof')
            require('reason=client-or-ownership' in log,'terminal title did not quarantine')
        if name in CLIENT_FAILURE:
            require('ROUTES_CLIENT_REJECT mode='+name+' ' in log and 'terminal=0' in log,'clean failure point')
        if name in ('skip-unregister','reject-unregister','late-live','late-partial-destructor'):
            require('reason=finalization' in log and 'finalized=0 model_only=true' in log,'post-destruction proof failure')
        if name=='late-partial':require('ROUTES_CLIENT_LATE_PARTIAL_PASS completed=8 stale_tokens=true finalized_required=false' in log and 'reason=client-or-ownership' in log,'late partial gate')
        if name=='partial':require('reason=registration' in log and 'RUNTIME_COMPLETE' not in log,'partial constructor gate')
        if name=='final-sync':require('reason=final-log' in log,'final persistence gate')
        if name=='log-exists':require(log=='previous-result\n','old result overwritten')
        if name not in ('elevation','config','log-exists'):
            live=8 if name in UNCERTAIN else 4 if name=='scratch-cleanup' else 0
            require(f'HIP_TITLE_MODEL STOP shell_requests={int(success)} live={live} model_only=true' in log,'model stop '+name)
    sys.path.insert(0,str(ROOT/'tools'))
    from check_canary_log import check_log
    normal=(output/cases['normal']['log']).read_text()
    lines=normal.splitlines();instrumentation=[x for x in lines if x.startswith(('ROUTES_MODEL_FINAL ','HIP_TITLE_MODEL STOP '))]
    require(len(instrumentation)==2,'unexpected model instrumentation')
    production='\n'.join(x for x in lines if x not in instrumentation)+'\n'
    require(check_log(production.encode(),'gfx1013','hip_routes',shell_exit_probe=True)['application_result']=='application-pass','production transcript')
    mutations=[]
    for token in ('VECTOR_UNREGISTER after_main=true active=0','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=8','RUNTIME_COMPLETE sequence=3 kernel=routes_spill blocking=true','ROUTES_COMPLETE round=2 route=2 blocking=true full_guards=true inputs=unchanged','PS5HIP SHELL_EXIT request-return=0'):
        mutations.append(production.replace(token+'\n',''))
    for old,new in [('removed=1','removed=0'),('terminal=0','terminal=1'),('state_destroyed=true','state_destroyed=false'),('word_comparisons=393216','word_comparisons=393215'),('functions=4','functions=1'),('kernel=routes_spill','kernel=routes_integer')]:mutations.append(production.replace(old,new))
    mutations.append(production+'HIP_TITLE FINAL duplicate\n')
    for log in mutations:require(check_log(log.encode(),'gfx1013','hip_routes',shell_exit_probe=True)['application_result']!='application-pass','mutated production proof accepted')
    return len(cases),len(mutations)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',type=Path);p.add_argument('--reference-root',type=Path);p.add_argument('--probe-dir',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.check:print('Checked source-bound title cases/mutations:',verify_receipt(a.check));return
    require(a.reference_root and a.probe_dir and a.output,'reference-root/probe-dir/output required')
    ref,probe,out=a.reference_root.resolve(),a.probe_dir.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((probe/'receipt.json').read_text());require(receipt.get('title_entry') is True,'genuine title entry required')
    for group in ('source_sha256','runtime_sources'):
        for n,v in receipt[group].items():require(sha(ROOT/n)==v,'compiler source changed '+n)
    for n,v in receipt['artifacts'].items():require(sha(probe/n)==v,'compiler artifact changed '+n)
    external=dict(receipt['inputs'])
    for n,v in external.items():require(sha(Path(n))==v,'compiler input changed')
    lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';elevation=ref/'.deps/elevation/client'
    stage=ref/'results/hip-host-abi-spike/rocm-6.2.4/package-stage/include'
    for h in (elevation/'elevation.hpp',elevation/'protocol.hpp',platform/'ps5platform/kernel.h'):external[str(h)]=sha(h)
    common=['clang++-18','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_ROUTES_TITLE=1','-DPS5HIP_RUNTIME_MODEL=1']
    for d in (ROOT/'include',stage,platform,elevation):common+=['-I',str(d)]
    core=[ROOT/'src'/n for n in ('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp')]
    sources=core+[ROOT/n for n in ('src/hip_runtime.cpp','tools/hip_routes_control.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp','tools/native_runtime.h','tests/hip_routes.hip','tests/hip_routes_model.cpp','tests/hip_routes_title_model.cpp','tools/check_canary_log.py','tests/hip_routes_title_test.py')]+list((ROOT/'include/ps5hip').glob('*.h'))
    before={str(s.relative_to(ROOT)):sha(s) for s in sources};commands=[]
    tools={n:{'path':shutil.which(n),'sha256':sha(Path(shutil.which(n)))} for n in ('clang++-18','ld.lld-18','llvm-nm-18')}
    def run(label,argv):
        result=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(result.stdout);(out/(label+'.stderr')).write_text(result.stderr);commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':result.returncode,'ASAN_OPTIONS':'protect_shadow_gap=0'});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(result.returncode==0,label+': '+result.stderr[-3000:]);require(not any(x in result.stdout+result.stderr for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','LeakSanitizer')),label+': captured sanitizer diagnostic');return result.stdout
    linker=out/'linux.ld';linker.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
    objects=[]
    for source in core+[ROOT/n for n in ('tools/hip_routes_control.cpp','tests/hip_routes_model.cpp','tools/hip_vector_title.cpp','tests/hip_routes_title_model.cpp','tools/native_shell_exit_probe.cpp')]:
        extra=[]
        if source.name=='hip_vector_title.cpp':extra=['-Dmain=ps5hip_routes_title_main','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync','-std=c++20']
        if source.name=='hip_routes_title_model.cpp':extra=['-std=c++20']
        if source.name=='native_shell_exit_probe.cpp':extra=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
        obj=out/(source.stem+'.o');run('compile-'+source.stem,[*common,*extra,'-c',source,'-o',obj]);objects.append(obj)
    cases={}
    for partial in (False,True):
        runtime=out/('runtime-partial.o' if partial else 'runtime.o');run(runtime.stem,[*common,*(['-DPS5HIP_RUNTIME_PARTIAL=1'] if partial else []),'-c',ROOT/'src/hip_runtime.cpp','-o',runtime])
        binary=out/('title-partial' if partial else 'title');run(binary.name+'-link',[*common,'-pie','-fuse-ld=lld-18',*objects,runtime,probe/'linux-model/routes.o','-Wl,-T,'+str(linker),'-Wl,--wrap=__hipUnregisterFatBinary','-o',binary])
        listing=run(binary.name+'-symbols',['llvm-nm-18','-n',binary]);symbols={v[2]:int(v[0],16) for line in listing.splitlines() if len(v:=line.split())==3};begin=symbols['__ps5hip_wrappers_begin'];require(begin%32==0 and symbols['__hip_fatbin_wrapper']==begin and symbols['__ps5hip_wrappers_end']==begin+24,'model wrapper ABI span')
        for name in (['partial'] if partial else sorted(CASES-{'partial'})):
            folder=out/'cases'/name;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('relative-bad' if name=='config' else str(folder));log=folder/'ps5-hip-gfx1013-hip_routes.log'
            if name=='log-exists':log.write_text('previous-result\n')
            run('run-'+name,[binary,name,config])
            if not log.exists():log.write_bytes((out/('run-'+name+'.stdout')).read_bytes()+(out/('run-'+name+'.stderr')).read_bytes())
            cases[name]={'log':str(log.relative_to(out)),'sha256':sha(log)}
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');count,mutations=check(out)
    require(before=={n:sha(ROOT/n) for n in before},'source changed during title model')
    require(all(sha(Path(n))==v for n,v in external.items()),'external input changed')
    require(all(sha(Path(t['path']))==t['sha256'] for t in tools.values()),'tool changed')
    result={'schema':1,'classification':'genuine-four-route-title-host-lifecycle-pass','source_hashes':before,'input_hashes':external,'tools':tools,'cases':count,'transcript_mutations':mutations,'genuine_compiler_ctor_dtor':True,'model_is_isa_emulation':False,'native_executed':False,'inputs':{'combined_object':str(probe/'linux-model/routes.o'),'compiler_receipt':str(probe/'receipt.json'),'combined_sha256':sha(probe/'linux-model/routes.o'),'frozen_receipt_sha256':sha(probe/'receipt.json')},'artifacts':{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()}}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',count,'actual routes title cases and',mutations,'production transcript mutations')
if __name__=='__main__':main()
