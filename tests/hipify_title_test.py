#!/usr/bin/env python3
"""Genuine no-argument translated client and finite title lifecycle; no GPU execution."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_canary_log import check_log
from probe_hip_generic import CORE
UNCERTAIN={'submit','kick','timeout','throw','unknown','postflush'}
CLIENT_FAILURE={'init','preflush','data-allocate','missing','corrupt','input','guard','stale-second'}
CASES={'normal',*UNCERTAIN,*CLIENT_FAILURE,'cleanup','partial','no-finish','elevation','config','log-exists','startup-sync','final-sync','skip-unregister','reject-unregister','late-live','shell-fail','late-partial-destructor'}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def boundaries(production):
    require(check_log(production.encode(),'gfx1013','hipify',shell_exit_probe=True)['application_result']=='application-pass','strict application contract')
    mutations=[]
    for line in production.splitlines():mutations.append(production.replace(line+'\n','',1))
    for a,b in [('functions=1','functions=2'),('dispatches=6','dispatches=5'),('exact_values=662','exact_values=661'),('word_comparisons=2238','word_comparisons=2237'),('inputs_unchanged=true','inputs_unchanged=false'),('full_guards=true','full_guards=false'),('removed=1','removed=0'),('terminal=0','terminal=1'),('state_destroyed=true','state_destroyed=false'),('sequence=3','sequence=2'),('kernel=translated_add','kernel=other')]:mutations.append(production.replace(a,b))
    mutations += [production+production,production+'TRANSLATED_CLIENT_FAIL clean=0 error=bad\n',production.replace('VECTOR_UNREGISTER after_main=true active=0\n','').replace('PS5HIP START','VECTOR_UNREGISTER after_main=true active=0\nPS5HIP START',1)]
    for text in mutations:require(check_log(text.encode(),'gfx1013','hipify',shell_exit_probe=True)['application_result']!='application-pass','invalid transcript accepted')
    return len(mutations)
def good_log():
    lines=['PS5HIP START target=gfx1013 kernel=hipify','HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0','PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass']
    lines += [f'RUNTIME_COMPLETE sequence={i} kernel=translated_add blocking=true' for i in range(1,7)]
    lines += ['TRANSLATED_CLIENT_PASS launches=6 exact_u32=true inputs_unchanged=true full_guards=true','HIP_TITLE COMPUTE complete=true dispatches=6 exact_values=662 word_comparisons=2238 finalization=pending','PS5HIP SHUTDOWN main-return=0','PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed','VECTOR_UNREGISTER after_main=true active=0','VECTOR_FINAL registered=1 removed=1 completed=6 active=0 terminal=0 main_finished=1 hardware_validated=false','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=6','PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin','PS5HIP SHELL_EXIT request-return=0']
    return '\n'.join(lines)+'\n'
def check(output):
    cases=json.loads((output/'cases.json').read_text());require(set(cases)==CASES,'case inventory')
    for name,record in cases.items():
        log=(output/record['log']).read_text();require(sha(output/record['log'])==record['sha256'],'log changed')
        captured=(output/('run-'+name+'.stdout')).read_text()+(output/('run-'+name+'.stderr')).read_text()
        require(not any(x in log+captured for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','LeakSanitizer')),'sanitizer '+name)
        success=name in ('normal','shell-fail')
        if success:
            instrumentation=[x for x in log.splitlines() if x.startswith(('HIPIFY_MODEL_FINAL ','HIP_TITLE_MODEL STOP '))]
            require(len(instrumentation)==2,'model instrumentation inventory')
            production='\n'.join(x for x in log.splitlines() if x not in instrumentation)+'\n'
            if name=='normal':require(production==good_log(),'exact successful lifecycle')
            else:require('PS5HIP FAIL: shell-exit request-return=0xfffffff9' in production and check_log(production.encode(),'gfx1013','hipify',shell_exit_probe=True)['application_result']=='failed','shell failure classification')
            require('HIPIFY_MODEL_FINAL live=0 backends=1 submits=6 releases=33 completed=6 modules=0 functions=0 finalized=1 model_only=true' in log,'normal ownership closure')
        else:
            require('PS5HIP SHELL_EXIT request=begin' not in log,'unsafe failed shell request '+name)
            if name!='final-sync':require('HIP_TITLE FINAL ' not in log,'false final proof '+name)
        if name in UNCERTAIN|{'cleanup'}:
            live,freed,unresolved=(4,3,0) if name=='cleanup' else (7,0,1)
            require(f'HIPIFY_TITLE_RETENTION live={live} releases={freed} unresolved={unresolved} submits=1 no_retry=true model_only=true' in log and 'TRANSLATED_CLIENT_FAIL clean=0' in log and 'reason=client-or-ownership' in log,'terminal retention '+name)
            require('VECTOR_UNREGISTER' not in log and 'VECTOR_FINAL' not in log,'terminal false teardown')
        if name in CLIENT_FAILURE:require('TRANSLATED_CLIENT_FAIL clean=1' in log and 'reason=client-or-ownership' in log,'clean unwind '+name)
        if name in ('skip-unregister','reject-unregister','late-live','late-partial-destructor'):require('reason=finalization' in log and 'finalized=0' in log,'finalization gate '+name)
        if name=='partial':require('reason=registration' in log and 'RUNTIME_COMPLETE' not in log,'partial registration')
        if name=='no-finish':require('reason=client-or-ownership' in log and 'VECTOR_UNREGISTER' not in log,'missing finish')
        if name=='final-sync':require('reason=final-log' in log,'final persistence')
        if name=='log-exists':require(log=='previous-result\n','existing log overwritten')
        if name not in ('elevation','config','log-exists'):
            live=7 if name in UNCERTAIN else 4 if name=='cleanup' else 0
            require(f'HIP_TITLE_MODEL STOP shell_requests={int(success)} live={live} model_only=true' in log,'bounded model stop '+name)
    normal=(output/cases['normal']['log']).read_text()
    production='\n'.join(x for x in normal.splitlines() if not x.startswith(('HIPIFY_MODEL_FINAL ','HIP_TITLE_MODEL STOP ')))+'\n'
    return len(cases),boundaries(production)
def verify(output):
    r=json.loads((output/'receipt.json').read_text());require(r['classification']=='genuine-hipify-title-host-lifecycle-pass' and not r['hardware_executed'],'scope')
    for n,v in r['sources'].items():require(sha(ROOT/n)==v,'source drift '+n)
    for n,v in r['inputs'].items():require(sha(n)==v,'input drift '+n)
    for n,v in r['artifacts'].items():require(sha(output/n)==v,'artifact drift '+n)
    for t in r['tools'].values():require(sha(t['path'])==t['sha256'],'tool drift')
    require(check(output)==(r['cases'],r['mutations']),'result count');print('Checked actual title lifecycle',r['cases'],r['mutations'])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',type=pathlib.Path);p.add_argument('--boundaries',action='store_true');p.add_argument('--reference-root',type=pathlib.Path);p.add_argument('--probe-dir',type=pathlib.Path);p.add_argument('--compiler-source-root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
    if a.boundaries:print('Strict transcript mutations:',boundaries(good_log()));return
    if a.check:verify(a.check.resolve());return
    require(a.reference_root and a.probe_dir and a.compiler_source_root and a.output,'reference-root/probe-dir/compiler-source-root/output required')
    ref,probe,out=a.reference_root.resolve(),a.probe_dir.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    r=json.loads((probe/'receipt.json').read_text());require(sha(probe/'receipt.json')=='f635bff479f6b57e65068ae8fdf91c899e34f2b982d3799c0805871768b8f38d','reviewed integrated runtime proof')
    frozen=a.compiler_source_root.resolve();external=dict(r['inputs']);external.update({str(frozen/n):v for n,v in r['sources'].items()});external.update({str(probe/n):v for n,v in r['artifacts'].items()});external[str(probe/'receipt.json')]=sha(probe/'receipt.json')
    review=ref/'results/integrated-hipify-runtime-independent-review-v1.json';require(sha(review)=='0323a5245f95840778defa778427219929653cb36d0bfd793b28117cbc2e185e','runtime review changed');external[str(review)]=sha(review)
    require(all(sha(n)==v for n,v in external.items()),'frozen runtime inputs changed')
    lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include';elevation=ref/'.deps/elevation/client';headers=ref/'results/hip-host-abi-spike/rocm-6.2.4/package-stage/include'
    for h in (elevation/'elevation.hpp',elevation/'protocol.hpp'):external[str(h)]=sha(h)
    sources=[ROOT/'src'/n for n in CORE]+[ROOT/n for n in ('src/hip_launch_profile.cpp','tools/hip_vector_title.cpp','tools/native_shell_exit_probe.cpp','tools/native_runtime.h','tests/hipify_runtime_model.cpp','tests/hipify_title_model.cpp','tests/hipify_title_test.py','tools/check_canary_log.py','tools/math_native_fixture.py','tools/math_native_inputs.json','docs/HIPIFY_TITLE.md','CMakeLists.txt')]+list((ROOT/'include/ps5hip').glob('*.h'))
    before={str(f.relative_to(ROOT)):sha(f) for f in sources};tools={n:{'path':shutil.which(n),'sha256':sha(shutil.which(n))} for n in ('clang++-18','ld.lld-18','llvm-nm-18')};commands=[]
    def run(label,argv):
        argv=list(map(str,argv));q=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':argv,'exit_code':q.returncode,'ASAN_OPTIONS':'protect_shadow_gap=0'});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(q.returncode==0,label+': '+q.stderr[-3000:]);return q.stdout
    try:
        common=['clang++-18','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_HIPIFY_TITLE=1','-DPS5HIP_RUNTIME_MODEL=1']
        for d in (ROOT/'include',headers,platform,elevation):common+=['-I',str(d)]
        linker=out/'linux.ld';linker.write_bytes((probe/'linux.ld').read_bytes())
        objects=[]
        for source in [ROOT/'src'/n for n in CORE]+[ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/hipify_runtime_model.cpp',ROOT/'tools/hip_vector_title.cpp',ROOT/'tests/hipify_title_model.cpp',ROOT/'tools/native_shell_exit_probe.cpp']:
            extra=[]
            if source.name=='hip_vector_title.cpp':extra=['-Dmain=ps5hip_hipify_title_main','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync','-std=c++20']
            if source.name=='hipify_title_model.cpp':extra=['-std=c++20']
            if source.name=='native_shell_exit_probe.cpp':extra=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
            obj=out/(source.stem+'.o');run('compile-'+source.stem,[*common,*extra,'-c',source,'-o',obj]);objects.append(obj)
        binary=out/'title';run('title-link',[*common,'-pie','-fuse-ld=lld-18',*objects,probe/'linux-model/translated.o','-Wl,-T,'+str(linker),'-Wl,--wrap=__hipUnregisterFatBinary','-Wl,--wrap=ps5hip_runtime_finish','-o',binary])
        listing=run('title-symbols',['llvm-nm-18','-n',binary]);symbols={v[2]:int(v[0],16) for line in listing.splitlines() if len(v:=line.split())==3};begin=symbols['__ps5hip_wrappers_begin'];require(begin%32==0 and symbols['__hip_fatbin_wrapper']==begin and symbols['__ps5hip_wrappers_end']==begin+24,'actual wrapper span')
        cases={}
        for name in sorted(CASES):
            folder=out/'cases'/name;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('relative-bad' if name=='config' else str(folder));log=folder/'ps5-hip-gfx1013-hipify.log'
            if name=='log-exists':log.write_text('previous-result\n')
            run('run-'+name,[binary,name,config])
            if not log.exists():log.write_bytes((out/('run-'+name+'.stdout')).read_bytes()+(out/('run-'+name+'.stderr')).read_bytes())
            cases[name]={'log':str(log.relative_to(out)),'sha256':sha(log)}
        (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');count,mutations=check(out)
        require(all(sha(ROOT/n)==v for n,v in before.items()),'source drift');require(all(sha(n)==v for n,v in external.items()),'input drift');require(all(sha(t['path'])==t['sha256'] for t in tools.values()),'tool drift')
        result={'schema':1,'classification':'genuine-hipify-title-host-lifecycle-pass','sources':before,'inputs':external,'tools':tools,'cases':count,'mutations':mutations,'compiler_object':{'path':str(probe/'linux-model/translated.o'),'sha256':sha(probe/'linux-model/translated.o')},'genuine_compiler_ctor_dtor':True,'hardware_executed':False,'limits':['Host transport/services are models, not AMD ISA or native lifecycle qualification.','Original CUDA and generated HIP are unedited; no general compatibility claim.','ASan fixed shader-window mapping requires protect_shadow_gap=0; installed support trees not fully inventoried.'],'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',count,'actual lifecycle cases',mutations,'mutations',sha(out/'receipt.json'))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'sources_before':before,'inputs_before':external,'commands':commands},indent=2)+'\n');raise
if __name__=='__main__':main()
