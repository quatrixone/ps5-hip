#!/usr/bin/env python3
"""CPU-only TSAN of actual client coordination; synthetic HIP, no runtime/GPU claim."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from probe_hip_routes import prepared_inputs
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('main','init'),default='main')
    p.add_argument('--reference-root',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    _,headers,_,inputs=prepared_inputs(a.reference_root.resolve())
    files=['tools/hip_workers_client.cpp','tools/hip_workers_client.h','tests/hip_workers_client_tsan.cpp',
           'tests/hip_workers_client_tsan_test.py','tools/probe_hip_routes.py','tools/build_canaries.py',
           'tests/fat_binary_fixture_test.py','dependencies.json']
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'include/ps5hip').glob('*.h')]
    sources={name:sha(ROOT/name) for name in files}
    tools={name:shutil.which(name) for name in ('clang++-18','ld.lld-18')}
    assert all(tools.values())
    inputs.update({path:sha(path) for path in tools.values()})
    commands=[]
    def run(label,argv,timeout):
        q=subprocess.run(list(map(str,argv)),cwd=ROOT,env=dict(os.environ,TSAN_OPTIONS='halt_on_error=1'),capture_output=True,timeout=timeout)
        (out/(label+'.stdout')).write_bytes(q.stdout);(out/(label+'.stderr')).write_bytes(q.stderr)
        commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if q.returncode:raise RuntimeError(label+': '+q.stderr.decode(errors='replace')[-2000:])
        return q.stdout.decode()+q.stderr.decode()
    try:
        common=[tools['clang++-18'],'-std=c++17','-O1','-g','-pthread','-Wall','-Wextra','-Werror',
                '-fsanitize=thread','-fno-omit-frame-pointer','-fPIE','-I',ROOT/'include','-I',headers/'include']
        if a.variant=='init':common.append('-DPS5HIP_WORKERS_INIT_TITLE=1')
        run('compile-client',common+['-Dpthread_create=tsan_client_create','-Dpthread_join=tsan_client_join',
                                    '-c',ROOT/'tools/hip_workers_client.cpp','-o',out/'client.o'],180)
        run('link',common+['-pie','-fuse-ld=lld-18',ROOT/'tests/hip_workers_client_tsan.cpp',out/'client.o','-o',out/'model'],180)
        modes=['normal','partial-create','join-error']
        for mode in modes:
            text=run(mode,[out/'model',mode],35)
            expected='WORKERS_CLIENT_TSAN_PASS mode='+mode+' synthetic_hip=true'
            assert text.splitlines().count(expected)==1
            assert not any(word in text for word in ('ThreadSanitizer','runtime error:','assertion failed'))
        assert all(sha(ROOT/n)==value for n,value in sources.items()),'source drift'
        assert all(sha(n)==value for n,value in inputs.items()),'input drift'
        receipt={'schema':1,'classification':'actual-worker-client-synthetic-HIP-TSAN-pass',
                 'sources':sources,'inputs':inputs,'commands':commands,'modes':modes,'variant':a.variant,
                 'hardware_executed':False,'actual_hip_runtime_executed':False,
                 'limits':['Actual client coordination and pthreads only; HIP memory/launch/error APIs are synthetic.',
                           'Join-failure wrapper completes the real join before reporting failure; no termination of a stuck thread.'],
                 'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'sources':sources,'inputs':inputs,'commands':commands},indent=2)+'\n')
        raise

if __name__=='__main__':main()
