"""Run focused actual ROCclr lifecycle faults against a completed source-bound closure."""
from pathlib import Path
import argparse,hashlib,json,subprocess,shlex,os
p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];Q=a.proof.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=json.loads(Q.read_text());inputs={str(Q):h(Q)};commands=[]
assert q['classification']=='source-bound-real-clr-ps5-adapter-model'
assert q['sanitizer'] in ('address','thread','none')
for n,z in q['inputs'].items():assert h(n)==z,n;inputs[n]=z
for n,z in q['artifacts'].items():p=Q.parent/n;assert h(p)==z,n;inputs[str(p)]=z
source=R/'tests/clr_lifetime/thread_model.cpp'
for p in [source,source.parent/'README.md',Path(__file__)]:inputs[str(p.resolve())]=h(p)
compile_cmd=next(c['argv'] for c in q['commands'] if c['label']=='model').copy();flags=compile_cmd[:compile_cmd.index('-c')]
def run(label,cmd,env=None):
 try:r=subprocess.run(cmd,cwd=R,capture_output=True,text=True,timeout=30,env=env)
 except subprocess.TimeoutExpired as e:
  (O/(label+'.stdout')).write_bytes(e.stdout or b'');(O/(label+'.stderr')).write_bytes(e.stderr or b'');commands.append(dict(label=label,argv=cmd,timeout_seconds=30));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');raise
 (O/(label+'.stdout')).write_text(r.stdout);(O/(label+'.stderr')).write_text(r.stderr);commands.append(dict(label=label,argv=cmd,exit_code=r.returncode,sanitizer_environment={k:(env or os.environ).get(k) for k in ('ASAN_OPTIONS','UBSAN_OPTIONS','TSAN_OPTIONS')}));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert r.returncode==0,(label,r.stdout,r.stderr);return r
run('headers',flags+['-M',str(source),'-MF',str(O/'headers.d')])
for n in shlex.split((O/'headers.d').read_text().replace('\\\n',' ').split(':',1)[1]):
 z=h(n);assert n not in inputs or inputs[n]==z;inputs[n]=z
run('compile',flags+['-c',str(source),'-o',str(O/'thread-model.o')])
link=next(c['argv'] for c in q['commands'] if c['label']=='link').copy();model=str(Q.parent/'model.o');assert link.count(model)==1;link[link.index(model)]=str(O/'thread-model.o');link[link.index('-o')+1]=str(O/'thread-model');link+=['-Wl,--wrap=malloc,--wrap=pthread_create,--wrap=pthread_join,--wrap=_ZN3amd13AlignedMemory8allocateEmm,--wrap=_ZN3amd13AlignedMemory10deallocateEPv'];run('link',link)
modes=['allocation-1','allocation-2','allocation-3','allocation-4','thread-create-failure','thread-finished','thread-self-join','queue-create-failure','queue-normal','queue-join-failure','queue-self-release']
for mode in modes:
 env=dict(os.environ);env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';env['TSAN_OPTIONS']='halt_on_error=1';env['ASAN_OPTIONS']='detect_leaks='+('0' if mode in ['queue-join-failure','queue-self-release'] else '1')
 r=run(mode,[str(O/'thread-model'),mode],env);assert r.stdout.startswith('CLR_LIFETIME_PASS mode='+mode+' ') and r.stdout.count('\n')==1 and not r.stderr,(mode,r.stdout,r.stderr)
for n,z in inputs.items():assert h(n)==z,n
report=dict(classification='pass-focused-actual-clr-thread-lifecycle',sanitizer=q['sanitizer'],inputs=inputs,commands=commands,modes=modes,artifacts={p.name:h(p) for p in O.iterdir() if p.is_file()},limits=['Real patched upstream Thread and HostQueue; pthread failure return values injected at link wrappers, successful operations use real pthreads.','Join failure returns EINVAL without joining the already-finished worker; entire zero-ref queue and manual owners deliberately retained until test process exit.','Self-release test runs in genuine queue completion callback and retains live queue/owners until test process exit.','No native OS execution or GPU calls.'])
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(h(O/'receipt.json'))
