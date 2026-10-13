#!/usr/bin/env python3
"""Local real-pthread proof and finite-title models; never contacts a console."""
import argparse, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CORE_CASES=('normal','repeat','create-first','create-second','key-create','main-key','worker-key','stack-attr','wait-timeout','busy-missing','missing-destructor','duplicate-destructor','join-error','mutex-close','cond-close','key-close','main-unlock','worker-unlock')
TITLE_CASES=('normal','shell-fail','elevation','config','log-exists','startup-sync','result-sync','final-sync','create-second','join-error','mutex-close','wait-timeout','missing-destructor','main-unlock','worker-unlock')
WRAPPED=('create','join','key_create','setspecific','attr_setstacksize','cond_timedwait','mutex_trylock','mutex_unlock','mutex_destroy','cond_destroy','key_delete')
REPORT='HOST_THREADS RESULT passed=1 clean=1 error=0 created=2 joined=2 busy=2 counter=256 turns=256 destructors=1,1 tls=29,256,257 stack_bytes=524288'
EVENTS=['PS5HIP START target=cpu kernel=host_threads','PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass',REPORT,'HOST_THREADS PASS workers=2 updates=256 ordered_turns=256 key_destructors=2 gpu_calls=0 finalization=pending','PS5HIP SHUTDOWN main-return=0','PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed','HOST_THREADS FINAL joined=2 destructors=1,1 synchronization=closed cpu_only=true','PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin','PS5HIP SHELL_EXIT request-return=0']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def check_title(raw):
    text=raw.decode('ascii');require(text.endswith('\n'),'truncated log')
    require(text.splitlines()==EVENTS,'exact CPU title sequence')
    return True

def check(out):
    r=json.loads((out/'receipt.json').read_text())
    require(r['classification']=='actual-pthread-host-proof-pass' and not r['native_executed'] and not r['gpu_executed'],'classification')
    for n,h in r['sources'].items():require(sha(ROOT/n)==h,'source '+n)
    for n,h in r['inputs'].items():require(sha(n)==h,'input '+n)
    for n,h in r['artifacts'].items():require(sha(out/n)==h,'artifact '+n)
    for t in r['tools'].values():require(sha(t['path'])==t['sha256'],'tool')
    require(set(r['core_cases'])==set(CORE_CASES) and set(r['emulated_tls_cases'])==set(CORE_CASES) and set(r['title_cases'])==set(TITLE_CASES),'cases')
    for name in CORE_CASES:
        text=(out/f'core-{name}.stdout').read_text();require('actual_pthreads=true' in text,'real threads')
        require(('passed=1' if name in ('normal','repeat') else 'passed=0') in text,'result '+name)
    for name in CORE_CASES:
        text=(out/f'emutls-{name}.stdout').read_text();require('actual_pthreads=true' in text,'emutls real threads')
        require(('passed=1' if name in ('normal','repeat') else 'passed=0') in text,'emutls result '+name)
    for c in r['commands']:require(c['returncode']==0,'command '+c['label'])
    raw=(out/'production-title.log').read_bytes();check_title(raw)
    for token in EVENTS:
        try:check_title(raw.replace((token+'\n').encode(),b''))
        except RuntimeError:pass
        else:raise RuntimeError('omitted event accepted')
    for old,new in [('destructors=1,1','destructors=0,2'),('counter=256','counter=255'),('busy=2','busy=0'),('tls=29,256,257','tls=29,256,256'),('clean=1','clean=0')]:
        try:check_title(raw.replace(old.encode(),new.encode()))
        except RuntimeError:pass
        else:raise RuntimeError('corruption accepted')
    return len(CORE_CASES),len(CORE_CASES),len(TITLE_CASES),len(EVENTS)+5

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check',type=Path);a=p.parse_args()
    if a.check:print('Verified core/title/mutations:',check(a.check.resolve()));return
    require(a.output and a.reference_root,'output/reference-root required')
    ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=['tools/host_threads.h','tools/host_threads.cpp','tools/host_threads_title.cpp','tools/native_runtime.h','tools/native_shell_exit_probe.cpp','include/ps5hip/code_object.h','tests/host_threads_test.cpp','tests/host_threads_title_model.cpp','tools/probe_host_threads.py','CMakeLists.txt','docs/HOST_THREADS_PROOF.md']
    before={n:sha(ROOT/n) for n in sources};elevation=ref/'.deps/elevation/client'
    inputs={str(elevation/n):sha(elevation/n) for n in ('elevation.hpp','protocol.hpp')}
    compiler=shutil.which('clang++-18');require(compiler,'clang++-18 unavailable')
    ccompiler=shutil.which('clang-18');require(ccompiler,'clang-18 unavailable')
    tools={'clang++-18':{'path':compiler,'sha256':sha(compiler)},'clang-18':{'path':ccompiler,'sha256':sha(ccompiler)}};commands=[]
    deps=json.loads((ROOT/'dependencies.json').read_text());emutls=ref/'.deps'/('sdk-source-'+deps['sdk']['revision'])/'libc/emutls.c'
    require(sha(emutls)=='86813748aa5668112a0a5891601f5ae36660294a749a013fef7c3bbbd805737c','pinned SDK emutls source changed');inputs[str(emutls)]=sha(emutls)
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
    def run(label,argv,timeout=90):
        try:completed=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,capture_output=True,timeout=timeout)
        except subprocess.TimeoutExpired as e:completed=subprocess.CompletedProcess(list(map(str,argv)),124,e.stdout or b'',(e.stderr or b'')+b'CHILD_TIMEOUT\\n')
        (out/(label+'.stdout')).write_bytes(completed.stdout);(out/(label+'.stderr')).write_bytes(completed.stderr)
        commands.append({'label':label,'argv':list(map(str,argv)),'returncode':completed.returncode,'timeout_seconds':timeout})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        text=completed.stdout+completed.stderr
        require(completed.returncode==0,label+': '+text.decode(errors='replace')[-2000:])
        require(not any(t in text for t in (b'AddressSanitizer',b'LeakSanitizer',b'UndefinedBehaviorSanitizer',b'runtime error:')),label+' sanitizer diagnostic')
    flags=[compiler,'-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread','-I',ROOT/'tools','-I',ROOT/'include','-I',elevation]
    definitions=['-Dpthread_'+n+'=ps5hip_test_pthread_'+n for n in WRAPPED]
    # Only the proof TU redirects calls; sanitizer/libc private pthread calls remain untouched.
    run('compile-core',[*flags,*definitions,'-c',ROOT/'tools/host_threads.cpp','-o',out/'core.o'])
    run('compile-test',[*flags,'-c',ROOT/'tests/host_threads_test.cpp','-o',out/'test.o'])
    run('link-core',[*flags,out/'core.o',out/'test.o','-o',out/'core-test'])
    for name in CORE_CASES:run('core-'+name,[out/'core-test',name],12)
    run('compile-sdk-emutls',[ccompiler,'-std=c11','-O1','-g','-pthread','-fsanitize=address,undefined','-fno-omit-frame-pointer','-c',emutls,'-o',out/'emutls.o'])
    run('compile-emulated-core',[*flags,*definitions,'-femulated-tls','-c',ROOT/'tools/host_threads.cpp','-o',out/'emulated-core.o'])
    run('link-emulated-core',[*flags,out/'emulated-core.o',out/'test.o',out/'emutls.o','-o',out/'emulated-core-test'])
    for name in CORE_CASES:run('emutls-'+name,[out/'emulated-core-test',name],12)
    run('compile-unmodified-core',[*flags,'-c',ROOT/'tools/host_threads.cpp','-o',out/'unmodified-core.o'])
    run('link-unmodified-core',[*flags,out/'unmodified-core.o',out/'test.o','-o',out/'unmodified-core-test'])
    run('unmodified-core',[out/'unmodified-core-test','normal'],12)
    run('compile-title-doubles',[*flags,'-DPS5HIP_THREADS_TITLE_TEST=1','-c',ROOT/'tests/host_threads_test.cpp','-o',out/'doubles.o'])
    run('compile-title',[*flags,'-std=c++20','-DPS5HIP_THREADS_MODEL=1','-Dmain=ps5hip_threads_title_main','-Dsleep=ps5hip_threads_model_sleep','-Dfsync=ps5hip_threads_model_fsync','-c',ROOT/'tools/host_threads_title.cpp','-o',out/'title.o'])
    run('compile-shell',[*flags,'-D_fini=ps5hip_threads_model_fini','-DsceShellCoreUtilExitApp=ps5hip_threads_model_shell_exit','-Dsleep=ps5hip_threads_model_sleep','-Dfsync=ps5hip_threads_model_fsync','-c',ROOT/'tools/native_shell_exit_probe.cpp','-o',out/'shell.o'])
    run('compile-title-model',[*flags,'-std=c++20','-c',ROOT/'tests/host_threads_title_model.cpp','-o',out/'title-model.o'])
    run('link-title',[*flags,*[out/n for n in ('core.o','doubles.o','title.o','shell.o','title-model.o')],'-o',out/'title-test'])
    cases={}
    for name in TITLE_CASES:
        folder=out/('title-'+name);folder.mkdir();config=folder/'results-dir.txt';config.write_text('/bad/../directory' if name=='config' else str(folder))
        log=folder/'ps5-hip-host_threads.log'
        if name=='log-exists':log.write_text('previous-result\n')
        run('title-'+name,[out/'title-test',name,config],12)
        raw=log.read_bytes() if log.exists() else b'';captured=(out/('title-'+name+'.stdout')).read_bytes()+(out/('title-'+name+'.stderr')).read_bytes()
        require(not any(t in raw+captured for t in (b'AddressSanitizer',b'LeakSanitizer',b'UndefinedBehaviorSanitizer',b'runtime error:')),'title sanitizer '+name)
        success=name in ('normal','shell-fail')
        require((b'PS5HIP SHELL_EXIT request=begin\n' in raw)==success,'shell gate '+name)
        if name=='normal':
            require(raw.endswith(b'THREAD_TITLE_MODEL STOP shell_requests=1 fini=1 actual_pthreads=true\n'),'stop')
            production=raw[:raw.rfind(b'THREAD_TITLE_MODEL STOP ')];check_title(production);(out/'production-title.log').write_bytes(production)
        elif name=='shell-fail':require(b'request-return=0xfffffff9' in raw,'shell failure')
        elif name=='log-exists':require(raw==b'previous-result\n','old log changed')
        elif name not in ('elevation','config'):require(b'QUARANTINED' in raw,'missing quarantine '+name)
        cases[name]={'log':str(log.relative_to(out)) if log.exists() else None}
    require(before=={n:sha(ROOT/n) for n in sources},'source changed')
    require(inputs=={n:sha(n) for n in inputs},'external source changed')
    artifacts={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    receipt={'schema':1,'classification':'actual-pthread-host-proof-pass','native_executed':False,'gpu_executed':False,'sources':before,'inputs':inputs,'tools':tools,'artifacts':artifacts,'commands':commands,'core_cases':list(CORE_CASES),'emulated_tls_cases':list(CORE_CASES),'title_cases':cases,'sanitizers':['address','undefined'],'environment':{k:env[k] for k in ('ASAN_OPTIONS','UBSAN_OPTIONS')},'limits':['Linux actual pthread scheduling and injected API failures; not PS5 execution.','Join itself has no portable timeout; worker waits have one absolute 3 second deadline and each host subprocess has a 12 second outer bound.','Join-error injection really joins then reports failure; unlock-error injection really unlocks then reports failure. Both prove conservative retention without abandoning or stranding Linux workers.','No HIP runtime, AGC, GPU allocation or submission is linked.','Installed system headers/static host runtime are not a hermetic toolchain inventory.']}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Verified core/title/mutations:',check(out))
if __name__=='__main__':main()
