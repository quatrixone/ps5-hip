#!/usr/bin/env python3
"""Build/check the actual HIP title lifecycle against host-only service doubles."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def check(output):
    cases=json.loads((output/'cases.json').read_text())
    for name,record in cases.items():
        log=(output/record['log']).read_text()
        normal=name=='normal'
        if normal:
            events=['HIP_TITLE REGISTRATION','VECTOR_COMPLETE sequence=1','VECTOR_COMPLETE sequence=2',
                'VECTOR_CLIENT_PASS','HIP_TITLE COMPUTE','PS5HIP SHUTDOWN main-return=0',
                'PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed',
                'VECTOR_UNREGISTER after_main=true active=0','VECTOR_FINAL registered=1 removed=1 completed=2 active=0 terminal=0 main_finished=1',
                'HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed exact_values=146',
                'PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin','PS5HIP SHELL_EXIT request-return=0']
            positions=[log.index(e) for e in events]
            require(positions==sorted(positions),'normal callback order')
            require('QUARANTINED' not in log and 'VECTOR_ERROR' not in log,'normal failure marker')
        elif name=='shell-fail':
            require('HIP_TITLE FINAL ' in log and 'request-return=0xfffffff9' in log,'shell failure missing')
        else:
            require('PS5HIP SHELL_EXIT request=begin' not in log,'failure requested shell exit: '+name)
            if name!='final-sync':require('HIP_TITLE FINAL ' not in log,'false final proof: '+name)
        if name not in ('elevation','config','log-exists'):
            expected=1 if name in ('normal','shell-fail') else 0
            require(f'HIP_TITLE_MODEL STOP shell_requests={expected} ' in log,'missing model stop: '+name)
        if name in ('skip-unregister','reject-unregister','late-live'):
            require('reason=finalization' in log,'post-destructor rejection absent')
        if name=='final-sync':require('reason=final-log' in log,'final persistence rejection absent')
        if name=='log-exists':require(log=='previous-result\n','old result overwritten')
    # Production contract is checked after removing only explicitly model-only
    # instrumentation; the original raw transcript remains preserved.
    sys.path.insert(0,str(ROOT/'tools'))
    from check_canary_log import check_log
    normal=(output/cases['normal']['log']).read_text()
    model_lines=[line for line in normal.splitlines() if line.startswith(('VECTOR_MODEL_FINAL ','VECTOR_MODEL_CLEANUP ','HIP_TITLE_MODEL STOP '))]
    require(len(model_lines)==3,'unexpected model instrumentation')
    production='\n'.join(line for line in normal.splitlines() if line not in model_lines)+'\n'
    require(check_log(production.encode(),'gfx1013','hip_vector',shell_exit_probe=True)['application_result']=='application-pass','production transcript contract')
    for token in ('VECTOR_UNREGISTER after_main=true active=0','HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed exact_values=146','PS5HIP SHELL_EXIT request-return=0'):
        changed=production.replace(token+'\n','')
        require(check_log(changed.encode(),'gfx1013','hip_vector',shell_exit_probe=True)['application_result']!='application-pass','missing lifecycle evidence accepted')
    for old,new in [('removed=1','removed=0'),('terminal=0','terminal=1'),('allocations=closed','allocations=retained'),('state_destroyed=true','state_destroyed=false')]:
        require(check_log(production.replace(old,new).encode(),'gfx1013','hip_vector',shell_exit_probe=True)['application_result']!='application-pass','mutated lifecycle evidence accepted')
    require(check_log((production+'HIP_TITLE FINAL duplicate\n').encode(),'gfx1013','hip_vector',shell_exit_probe=True)['application_result']!='application-pass','duplicate lifecycle record accepted')
    return len(cases)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',type=Path)
    p.add_argument('--reference-root',type=Path);p.add_argument('--probe-dir',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.check:print('Checked',check(a.check),'actual title model transcripts');return
    require(a.reference_root and a.probe_dir and a.output,'build requires reference-root, probe-dir and output')
    ref,probe,out=a.reference_root.resolve(),a.probe_dir.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((probe/'receipt.json').read_text())
    for name in ('linux-combined.o','linux.ld'):
        require(sha(probe/name)==receipt['artifacts'][name],'frozen model input changed')
    evidence=json.loads((ROOT/'evidence/offline-compatible-hip-headers.json').read_text())
    fixtures=ref/'results/hip-host-abi-spike/rocm-6.2.4'
    inc=[];external_headers=[]
    for key,sub in [('HIP','include'),('clr','hipamd/include')]:
        base=fixtures/(key.lower()+'-'+evidence['input_sources'][key]['revision'])
        for name,digest in receipt['header_hashes'].items():
            if name.startswith(key+'/'):
                header=base/name.split('/',1)[1];require(sha(header)==digest,'header changed');external_headers.append(header)
        inc.append(base/sub)
    generated=fixtures/'package-build/include/hip/hip_version.h';require(sha(generated)==receipt['generated_version_sha256'],'generated header changed');inc.append(generated.parents[1])
    lock=json.loads((ROOT/'dependencies.json').read_text())
    platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
    elevation=ref/'.deps/elevation/client'
    common=['clang++-18','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_VECTOR_TITLE=1']
    for d in (ROOT/'include',*inc,platform,elevation):common+=['-I',str(d)]
    core=[ROOT/'src'/n for n in ('code_object.cpp','fat_binary.cpp','host_registration.cpp','host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp')]
    sources=core+[ROOT/'tools/hip_vector_control.cpp',ROOT/'tools/hip_vector_title.cpp',ROOT/'tools/native_shell_exit_probe.cpp',ROOT/'tools/native_runtime.h',ROOT/'tests/hip_vector_client.cpp',ROOT/'tests/hip_vector_model.cpp',ROOT/'tests/hip_vector_title_model.cpp',ROOT/'tools/check_canary_log.py',Path(__file__)]
    sources+=list((ROOT/'include/ps5hip').glob('*.h'))
    external_headers += [generated,elevation/'elevation.hpp',elevation/'protocol.hpp',platform/'ps5platform/kernel.h']
    external_before={str(h):sha(h) for h in external_headers}
    before={str(s.relative_to(ROOT)):sha(s) for s in sources};commands=[]
    def run(label,argv):
        result=subprocess.run([str(x) for x in argv],cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'))
        (out/(label+'.stdout')).write_text(result.stdout);(out/(label+'.stderr')).write_text(result.stderr)
        commands.append({'label':label,'argv':[str(x) for x in argv],'exit_code':result.returncode,'ASAN_OPTIONS':'protect_shadow_gap=0'})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(result.returncode==0,label+': '+result.stderr[-3000:])
    objects=[]
    for source in core+[ROOT/'tests/hip_vector_client.cpp',ROOT/'tests/hip_vector_model.cpp',ROOT/'tools/hip_vector_title.cpp',ROOT/'tests/hip_vector_title_model.cpp',ROOT/'tools/native_shell_exit_probe.cpp']:
        extra=[]
        if source.name=='hip_vector_client.cpp':extra=['-Dmain=ps5hip_vector_client_main']
        if source.name=='hip_vector_model.cpp':extra=['-DPS5HIP_VECTOR_MODEL=1']
        if source.name=='hip_vector_title.cpp':extra=['-DPS5HIP_VECTOR_MODEL=1','-Dmain=ps5hip_vector_title_main','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
        if source.name=='native_shell_exit_probe.cpp':extra=['-D_fini=ps5hip_title_model_fini','-DsceShellCoreUtilExitApp=ps5hip_title_model_shell_exit','-Dsleep=ps5hip_title_model_sleep','-Dfsync=ps5hip_title_model_fsync']
        if source.name in ('hip_vector_title.cpp','hip_vector_title_model.cpp'):extra+=['-std=c++20']
        obj=out/(source.stem+'.o');run('compile-'+source.stem,[*common,*extra,'-c',source,'-o',obj]);objects.append(obj)
    cases={}
    for partial in (False,True):
        control=out/('control-partial.o' if partial else 'control.o');run(control.stem,[*common,'-DPS5HIP_VECTOR_MODEL=1',*(['-DPS5HIP_VECTOR_PARTIAL=1'] if partial else []),'-c',ROOT/'tools/hip_vector_control.cpp','-o',control])
        binary=out/('title-partial' if partial else 'title');run(binary.name+'-link',[*common,'-pie','-fuse-ld=lld-18',*objects,control,probe/'linux-combined.o','-Wl,-T,'+str(probe/'linux.ld'),'-Wl,--wrap=__hipUnregisterFatBinary','-o',binary])
        for name in (['partial'] if partial else ['normal','submit','kick','timeout','unknown-submit','missing','corrupt','input','guard','data-allocate','data-close','init','elevation','config','log-exists','startup-sync','final-sync','skip-unregister','reject-unregister','late-live','shell-fail']):
            folder=out/'cases'/name;folder.mkdir(parents=True);config=folder/'results-dir.txt';config.write_text('relative-bad' if name=='config' else str(folder))
            log=folder/'ps5-hip-gfx1013-hip_vector.log'
            if name=='log-exists':log.write_text('previous-result\n')
            run('run-'+name,[binary,name,config])
            if not log.exists():log.write_bytes((out/('run-'+name+'.stdout')).read_bytes()+(out/('run-'+name+'.stderr')).read_bytes())
            cases[name]={'log':str(log.relative_to(out)),'sha256':sha(log)}
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    count=check(out);require(before=={str(s.relative_to(ROOT)):sha(s) for s in sources},'source changed during model')
    require(external_before=={str(h):sha(h) for h in external_headers},'external header changed during model')
    (out/'receipt.json').write_text(json.dumps({'schema':1,'source_hashes':before,'input_header_hashes':external_before,'cases':count,'genuine_compiler_ctor_dtor':True,'model_is_isa_emulation':False,'native_executed':False,'inputs':{'combined_sha256':sha(probe/'linux-combined.o'),'frozen_receipt_sha256':sha(probe/'receipt.json')},'artifacts':{str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()}},indent=2)+'\n')
    print('Passed',count,'actual title lifecycle cases')
if __name__=='__main__':main()
