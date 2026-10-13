#!/usr/bin/env python3
"""Genuine 0/1/3/7 sequence title models; no GPU execution."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UNCERTAIN={'submit','kick','timeout','throw','unknown'}
CLIENT_FAILURE={'init','preflush','missing','corrupt','input','guard'}
CASES={'normal','partial','late-partial','cleanup',*UNCERTAIN,*CLIENT_FAILURE,'elevation','config','log-exists','startup-sync','final-sync','skip-unregister','reject-unregister','late-live','post-finish','shell-fail','late-partial-destructor'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def verify_receipt(output):
    r=json.loads((output/'receipt.json').read_text())
    require(r['classification']=='genuine-generic-title-host-lifecycle-pass' and not r['native_executed'] and r['genuine_compiler_ctor_dtor'] and not r['model_is_isa_emulation'],'classification')
    for n,h in r['source_hashes'].items():require(sha(ROOT/n)==h,'source drift '+n)
    for n,h in r['input_hashes'].items():require(sha(Path(n))==h,'input drift '+n)
    for n,h in r['artifacts'].items():require(sha(output/n)==h,'artifact drift '+n)
    for t in r['tools'].values():require(sha(Path(t['path']))==t['sha256'],'tool drift')
    i=r['inputs'];require(sha(Path(i['compiler_receipt']))==i['frozen_receipt_sha256'] and sha(Path(i['combined_object']))==i['combined_sha256'],'compiler binding')
    result=check(output);require(result==(r['cases'],r['transcript_mutations']),'counts');return result

def check(output):
    cases=json.loads((output/'cases.json').read_text());require(set(cases)==CASES,'case inventory')
    for name,record in cases.items():
        path=output/record['log'];require(sha(path)==record['sha256'],'log hash');log=path.read_text()
        captured=(output/('run-'+name+'.stdout')).read_text()+(output/('run-'+name+'.stderr')).read_text()
        require(not any(x in log+captured for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','LeakSanitizer')),'sanitizer '+name)
        success=name in ('normal','post-finish','shell-fail')
        if success:
            events=['HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=2 constructor_gpu_calls=0'];total=0
            for sequence,count in enumerate((0,1,3,7)):
                for iteration in range(count):
                    total+=1;kernel='affine_words' if iteration%2==0 else 'block_sum'
                    events.append(f'RUNTIME_COMPLETE sequence={total} kernel={kernel} blocking=true')
                events.append(f'GENERIC_SEQUENCE index={sequence} requested={count} observed={count} completed={total} prebackend={int(sequence<2)}')
            events+=['GENERIC_CLIENT_PASS launches=11 sequences=0,1,3,7 variable_sizes=true full_guards=true inputs_unchanged=true','HIP_TITLE COMPUTE complete=true dispatches=11 sequences=0,1,3,7 finalization=pending','PS5HIP SHUTDOWN main-return=0','PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed','VECTOR_UNREGISTER after_main=true active=0','VECTOR_FINAL registered=1 removed=1 completed=11 active=0 terminal=0 main_finished=1','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=11','PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin']
            positions=[log.index(e) for e in events];require(positions==sorted(positions),'callback order '+name)
            require('QUARANTINED' not in log and ('request-return=0xfffffff9' if name=='shell-fail' else 'request-return=0') in log,'exit result '+name)
        else:
            require('PS5HIP SHELL_EXIT request=begin' not in log,'failed title shell '+name)
            if name!='final-sync':require('HIP_TITLE FINAL ' not in log,'false final '+name)
        if name in UNCERTAIN|{'cleanup'}:
            require(f'GENERIC_TERMINAL live={3 if name=="cleanup" else 6} unresolved={int(name!="cleanup")} submits=1' in log and 'reason=client-or-ownership' in log,'retained boundary')
        if name in CLIENT_FAILURE:require('GENERIC_CLIENT_FAILURE clean=1 terminal=0' in log,'clean failure '+name)
        if name in ('skip-unregister','reject-unregister','late-partial-destructor'):require('reason=finalization' in log and 'finalized=0' in log,'postdestruction gate '+name)
        if name=='late-partial':require('GENERIC_LATE_PARTIAL completed=11 finalized=false' in log and 'reason=client-or-ownership' in log,'late registration')
        if name=='partial':require('reason=registration' in log and 'RUNTIME_COMPLETE' not in log,'constructor failure')
        if name in ('skip-unregister','reject-unregister','late-partial-destructor'):
            late=int(name=='late-partial-destructor')
            retained=f'RUNTIME_RETAINED registered=1 removed={late} completed=11 active={1-late} terminal=0 main_finished=1 contexts=0 pending=0 abandoned=0 registration_failed={late} hardware_validated=false'
            require(log.splitlines().count(retained)==1 and 'VECTOR_FINAL ' not in log,'exact retained State closure '+name)
        if name=='late-live':
            require('HIP_TITLE_MODEL PRE_FINISH live=1 rejection_expected=true' in log and 'reason=client-or-ownership' in log and 'HIP_TITLE_MODEL STOP shell_requests=0 live=1 model_only=true' in log,'live allocation blocks closure')
            require('VECTOR_UNREGISTER' not in log and 'VECTOR_FINAL' not in log and 'HIP_TITLE FINAL ' not in log,'dirty title destroyed runtime')
        if name=='post-finish':require('HIP_TITLE_MODEL POST_FINISH allocation_rejected=true ownership_unchanged=true' in log,'late allocation rejection absent')
        if name=='late-partial-destructor':
            require('HIP_TITLE_MODEL LATE_REGISTRATION rejected=true state_unchanged=true' in log and 'RUNTIME_RETAINED registered=1 removed=1' in log and 'registration_failed=1 hardware_validated=false' in log and 'VECTOR_FINAL ' not in log,'late registration not sticky or State destroyed')
        if name=='final-sync':require('reason=final-log' in log,'final persistence')
        if name=='log-exists':require(log=='previous-result\n','old log overwritten')
        if name not in ('elevation','config','log-exists'):
            live=1 if name=='late-live' else 6 if name in UNCERTAIN else 3 if name=='cleanup' else 0
            require(f'HIP_TITLE_MODEL STOP shell_requests={int(success)} live={live} model_only=true' in log,'stop '+name)
    sys.path.insert(0,str(ROOT/'tools'));from check_canary_log import check_log
    log=(output/cases['normal']['log']).read_text();lines=log.splitlines();extra=[x for x in lines if x.startswith(('GENERIC_MODEL_FINAL ','HIP_TITLE_MODEL STOP '))];require(len(extra)==2,'instrumentation')
    production='\n'.join(x for x in lines if x not in extra)+'\n'
    require(check_log(production.encode(),'gfx1013','hip_generic',shell_exit_probe=True)['application_result']=='application-pass','production transcript')
    mutations=[]
    for token in ('VECTOR_UNREGISTER after_main=true active=0','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=11','RUNTIME_COMPLETE sequence=3 kernel=block_sum blocking=true','GENERIC_SEQUENCE index=0 requested=0 observed=0 completed=0 prebackend=1','PS5HIP SHELL_EXIT request-return=0'):
        require(token in production,'mutation token');mutations.append(production.replace(token+'\n',''))
    for a,b in [('removed=1','removed=0'),('terminal=0','terminal=1'),('state_destroyed=true','state_destroyed=false'),('observed=3','observed=2'),('functions=2','functions=3'),('prebackend=1','prebackend=0'),('kernel=block_sum','kernel=affine_words'),('inputs_unchanged=true','inputs_unchanged=false')]:mutations.append(production.replace(a,b))
    mutations.append(production+'HIP_TITLE FINAL duplicate\n')
    for mutation in mutations:require(check_log(mutation.encode(),'gfx1013','hip_generic',shell_exit_probe=True)['application_result']!='application-pass','mutation accepted')
    return len(cases),len(mutations)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',type=Path);p.add_argument('--reference-root',type=Path);p.add_argument('--probe-dir',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.check:print('Checked source-bound title cases/mutations:',verify_receipt(a.check));return
    require(a.reference_root and a.probe_dir and a.output,'reference-root/probe-dir/output required')
    ref,probe,out=a.reference_root.resolve(),a.probe_dir.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((probe/'receipt.json').read_text());require(receipt.get('title_entry') is True,'genuine title entry required')
    for group in ('sources',):
        for n,v in receipt[group].items():require(sha(ROOT/n)==v,'compiler source changed '+n)
    for n,v in receipt['artifacts'].items():require(sha(probe/n)==v,'compiler artifact changed '+n)
    external=dict(receipt['inputs'])
    for n,v in external.items():require(sha(Path(n))==v,'compiler input changed')
    lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';elevation=ref/'.deps/elevation/client'
    stage=ref/'results/hip-host-abi-spike/rocm-6.2.4/package-stage/include'
    for h in (elevation/'elevation.hpp',elevation/'protocol.hpp',platform/'ps5platform/kernel.h'):external[str(h)]=sha(h)
    common=['clang++-18','-pthread','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_GENERIC_TITLE=1','-DPS5HIP_RUNTIME_MODEL=1']
    for d in (ROOT/'include',stage,platform,elevation):common+=['-I',str(d)]
    core=[ROOT/'src'/n for n in ('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp')]
    sources=core+[ROOT/n for n in ('src/hip_runtime.cpp','src/hip_launch_profile.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp','tools/native_runtime.h','tests/hip_generic.hip','tests/hip_generic_model.cpp','tests/hip_generic_title_model.cpp','tools/check_canary_log.py','tests/hip_generic_title_test.py')]+list((ROOT/'include/ps5hip').glob('*.h'))
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
    for source in core+[ROOT/n for n in ('src/hip_launch_profile.cpp','tests/hip_generic_model.cpp','tools/hip_vector_title.cpp','tests/hip_generic_title_model.cpp','tools/native_shell_exit_probe.cpp')]:
        extra=[]
        if source.name=='hip_vector_title.cpp':extra=['-Dmain=ps5hip_generic_title_main','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync','-std=c++20']
        if source.name=='hip_generic_title_model.cpp':extra=['-std=c++20']
        if source.name=='native_shell_exit_probe.cpp':extra=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
        obj=out/(source.stem+'.o');run('compile-'+source.stem,[*common,*extra,'-c',source,'-o',obj]);objects.append(obj)
    cases={}
    for partial in (False,True):
        runtime=out/('runtime-partial.o' if partial else 'runtime.o');run(runtime.stem,[*common,*(['-DPS5HIP_RUNTIME_PARTIAL=1'] if partial else []),'-c',ROOT/'src/hip_runtime.cpp','-o',runtime])
        binary=out/('title-partial' if partial else 'title');run(binary.name+'-link',[*common,'-pie','-fuse-ld=lld-18',*objects,runtime,probe/'linux-model/routes.o','-Wl,-T,'+str(linker),'-Wl,--wrap=__hipUnregisterFatBinary','-Wl,--wrap=ps5hip_runtime_finish','-o',binary])
        listing=run(binary.name+'-symbols',['llvm-nm-18','-n',binary]);symbols={v[2]:int(v[0],16) for line in listing.splitlines() if len(v:=line.split())==3};begin=symbols['__ps5hip_wrappers_begin'];require(begin%32==0 and symbols['__hip_fatbin_wrapper']==begin and symbols['__ps5hip_wrappers_end']==begin+24,'model wrapper ABI span')
        for name in (['partial'] if partial else sorted(CASES-{'partial'})):
            folder=out/'cases'/name;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('relative-bad' if name=='config' else str(folder));log=folder/'ps5-hip-gfx1013-hip_generic.log'
            if name=='log-exists':log.write_text('previous-result\n')
            run('run-'+name,[binary,name,config])
            if not log.exists():log.write_bytes((out/('run-'+name+'.stdout')).read_bytes()+(out/('run-'+name+'.stderr')).read_bytes())
            cases[name]={'log':str(log.relative_to(out)),'sha256':sha(log)}
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');count,mutations=check(out)
    require(before=={n:sha(ROOT/n) for n in before},'source changed during title model')
    require(all(sha(Path(n))==v for n,v in external.items()),'external input changed')
    require(all(sha(Path(t['path']))==t['sha256'] for t in tools.values()),'tool changed')
    result={'schema':1,'classification':'genuine-generic-title-host-lifecycle-pass','source_hashes':before,'input_hashes':external,'tools':tools,'cases':count,'transcript_mutations':mutations,'genuine_compiler_ctor_dtor':True,'model_is_isa_emulation':False,'native_executed':False,'inputs':{'combined_object':str(probe/'linux-model/routes.o'),'compiler_receipt':str(probe/'receipt.json'),'combined_sha256':sha(probe/'linux-model/routes.o'),'frozen_receipt_sha256':sha(probe/'receipt.json')},'artifacts':{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()}}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',count,'actual generic title cases and',mutations,'production transcript mutations')
if __name__=='__main__':main()
