#!/usr/bin/env python3
"""Finite wrapper and forwarding fault models; not genuine client/GPU execution."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from check_canary_log import check_log
CASES=('normal','join','unsafe','cleanup','created','phase','count','final-count','completion','agc-return','agc-actor','agc-extra','finish-rejected','skip-unregister','retained','registration','elevation','config','log-exists')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def inspect(out,variant):
    cases=CASES+(('prebackend','init-main','agc-missing') if variant=='init' else ())
    kernel='hip_workers_'+variant
    records=json.loads((out/'cases.json').read_text());require(set(records)==set(cases),'case inventory')
    for case,record in records.items():
        log=(out/record['log']).read_text();require(sha(out/record['log'])==record['sha256'],'log drift')
        if case=='normal':require(check_log(log.encode(),'gfx1013',kernel,shell_exit_probe=True)['application_result']=='application-pass','normal transcript')
        else:
            require('PS5HIP SHELL_EXIT request=begin' not in log,'unsafe shell exit '+case)
            if case in ('config','elevation'):require(not log,'early failure log')
            elif case=='log-exists':require(log=='previous-result\n','overwritten log')
            else:
                require('HIP_TITLE QUARANTINED' in log and 'WORKERS_TITLE_MODEL STOP ' in log,'not bounded quarantine '+case)
                if case in ('join','unsafe','cleanup','created'):require('reason=worker-lifecycle' in log,'worker retention gate')
                if case in ('join','unsafe'):require('finish_calls=0' in log and 'WORKERS_AGC' not in log,'unsafe runtime/trace access')
    normal=(out/records['normal']['log']).read_text();mutations=[]
    other='hip_workers_'+('main' if variant=='init' else 'init')
    require(check_log(normal.encode(),'gfx1013',other,shell_exit_probe=True)['application_result']!='application-pass','cross-profile transcript')
    require(check_log(normal.replace(kernel,other).encode(),'gfx1013',other,shell_exit_probe=True)['application_result']!='application-pass','relabelled cross-profile transcript')
    for line in normal.splitlines():mutations.extend((normal.replace(line+'\n','',1),normal.replace(line+'\n',line+'\n'+line+'\n',1)))
    for a,b in [('actor=A','actor=B'),('actor=1','actor=2'),('joined=2','joined=1'),('phase_checks=17','phase_checks=16'),('exact_values='+('660' if variant=='init' else '661'),'exact_values=0'),('word_comparisons='+('1084' if variant=='init' else '1355'),'word_comparisons=0'),('template_comparisons='+('1084' if variant=='init' else '1355'),'template_comparisons=0'),('backend_attempted=false','backend_attempted=true'),('result=0x00000000','result=0xffffffff'),('state_destroyed=true','state_destroyed=false')]:
        if a in normal:mutations.append(normal.replace(a,b))
    for log in mutations:require(check_log(log.encode(),'gfx1013',kernel,shell_exit_probe=True)['application_result']!='application-pass','mutation accepted')
    return len(records),len(mutations)
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('main','init'),default='main');p.add_argument('--reference-root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);p.add_argument('--check',type=pathlib.Path);a=p.parse_args()
    if a.check:
        out=a.check.resolve();r=json.loads((out/'receipt.json').read_text())
        for n,h in r['sources'].items():require(sha(ROOT/n)==h,'source drift '+n)
        for n,h in r['inputs'].items():require(sha(n)==h,'input drift '+n)
        for n,h in r['artifacts'].items():require(sha(out/n)==h,'artifact drift '+n)
        require(inspect(out,r['variant'])==(r['cases'],r['mutations']),'counts');print('Checked worker title model');return
    require(a.reference_root and a.output,'reference-root/output required');ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sdk=ref/'.deps/sdk-source-95c08f27386fc698f6bbe21dde3030140a41d10b/platform/include';elevation=ref/'.deps/elevation/client'
    paths=['tools/hip_vector_title.cpp','tools/native_runtime.h','tools/hip_workers_client.h','tools/hip_workers_agc.h','tools/hip_workers_agc.cpp','tests/hip_workers_title_model.cpp','tests/hip_workers_title_test.py','tools/check_canary_log.py','tools/math_native_fixture.py','tools/math_native_inputs.json']+[str(x.relative_to(ROOT)) for x in (ROOT/'include/ps5hip').glob('*.h')]
    sources={n:sha(ROOT/n) for n in paths};inputs={str(x):sha(x) for x in (elevation/'elevation.hpp',elevation/'protocol.hpp',sdk/'ps5platform/agc.h')}
    compiler=shutil.which('clang++-18');require(compiler,'clang++ missing');inputs[compiler]=sha(compiler)
    common=[compiler,'-std=c++20','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread','-DPS5HIP_RUNTIME_MODEL=1','-DPS5HIP_WORKERS_'+a.variant.upper()+'_TITLE=1',*[y for x in (ROOT/'include',ROOT/'tools',elevation,sdk) for y in ('-I',str(x))]]
    commands=[]
    def run(label,cmd):
        q=subprocess.run(list(map(str,cmd)),cwd=ROOT,capture_output=True,text=True,timeout=90,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':list(map(str,cmd)),'returncode':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(q.returncode==0,label+': '+q.stderr[-3000:]);require(not any(x in q.stderr+q.stdout for x in ('AddressSanitizer','runtime error:','LeakSanitizer')),'sanitizer '+label)
    run('compile-title',common+['-Dmain=workers_title_main','-c',ROOT/'tools/hip_vector_title.cpp','-o',out/'title.o'])
    run('link-model',common+[ROOT/'tests/hip_workers_title_model.cpp',ROOT/'tools/hip_workers_agc.cpp',out/'title.o','-o',out/'model'])
    records={}
    for case in CASES+(('prebackend','init-main','agc-missing') if a.variant=='init' else ()):
        folder=out/case;folder.mkdir();log=folder/('ps5-hip-gfx1013-hip_workers_'+a.variant+'.log');config=folder/'config';config.write_text('/../bad' if case=='config' else str(folder))
        if case=='log-exists':log.write_text('previous-result\n')
        run('run-'+case,[out/'model',case,config])
        if not log.exists():log.write_text('')
        records[case]={'log':str(log.relative_to(out)),'sha256':sha(log)}
    (out/'cases.json').write_text(json.dumps(records,indent=2)+'\n');cases,mutations=inspect(out,a.variant)
    require(all(sha(ROOT/n)==h for n,h in sources.items()),'source changed during proof')
    r={'classification':'worker-title-wrapper-host-model-pass','hardware_executed':False,'genuine_client_executed':False,'variant':a.variant,'cases':cases,'mutations':mutations,'sources':sources,'inputs':inputs,'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(cases,mutations,sha(out/'receipt.json'))
if __name__=='__main__':main()
