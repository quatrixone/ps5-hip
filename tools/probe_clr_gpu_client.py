if not __debug__: raise RuntimeError("Validation requires ordinary Python")
"""Actual native-client logic with explicitly synthetic transport, using qualified CLR objects."""
from pathlib import Path
import argparse,hashlib,json,subprocess,shlex,os
p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
C=Path(__file__).resolve().parents[1];Q=a.proof.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=json.loads(Q.read_text());inputs={str(Q):h(Q),str(Path(__file__)):h(__file__)};commands=[]
assert q['classification']=='source-bound-real-clr-ps5-adapter-model' and q['sanitizer'] in ('address','thread')
for n,z in q['inputs'].items():assert h(n)==z,n;inputs[n]=z
for n,z in q['artifacts'].items():p=Q.parent/n;assert h(p)==z,n;inputs[str(p)]=z
source=C/'tests/clr_gpu/client_model.cpp';code=O/'code.cpp'
blob_path=next(Path(n) for n in q['inputs'] if n.endswith('/ps5/threads.hsaco'))
blob=blob_path.read_bytes();assert h(blob_path)==q['genuine_code_object_sha256']
code.write_text('#include <cstddef>\nalignas(16) extern const unsigned char ps5hip_clr_code[] = {'+','.join(str(v) for v in blob)+'};\nextern const size_t ps5hip_clr_code_size = sizeof(ps5hip_clr_code);\n')
for p in (source,code,C/'tools/clr_gpu/client.cpp',C/'include/ps5hip/agc_backend.h'):inputs[str(p)]=h(p)
compile_cmd=next(c['argv'] for c in q['commands'] if c['label']=='model');flags=compile_cmd[:compile_cmd.index('-c')]+['-I'+str(C/'include')]
def run(label,cmd,env=None):
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=60,env=env);(O/(label+'.stdout')).write_text(z.stdout);(O/(label+'.stderr')).write_text(z.stderr);commands.append(dict(label=label,argv=cmd,exit_code=z.returncode,sanitizer_environment={k:(env or os.environ).get(k) for k in ('ASAN_OPTIONS','UBSAN_OPTIONS','TSAN_OPTIONS')}));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert z.returncode==0,(label,z.stdout,z.stderr);assert not any(v in z.stderr for v in ('runtime error:','ERROR: AddressSanitizer','WARNING: ThreadSanitizer','LeakSanitizer:'));return z
for name,p in [('client',source),('code',code)]:
 run(name+'-headers',flags+['-M',str(p),'-MF',str(O/(name+'.d'))])
 for n in shlex.split((O/(name+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]):inputs[n]=h(n)
 run(name+'-compile',flags+['-c',str(p),'-o',str(O/(name+'.o'))])
link=next(c['argv'] for c in q['commands'] if c['label']=='link').copy();link[link.index(str(Q.parent/'model.o'))]=str(O/'client.o');link.insert(link.index('-o'),str(O/'code.o'));link.insert(link.index('-o'),'-Wl,--wrap=pthread_join');link[link.index('-o')+1]=str(O/'client');run('link',link)
modes=['normal','init','noop','corrupt','submit','throw','kick','timeout','postflush','cleanup','join-fail']
for mode in modes:
 env=os.environ.copy();env['TSAN_OPTIONS']='halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';env['ASAN_OPTIONS']='protect_shadow_gap=0:detect_leaks='+('1' if mode=='normal' else '0')
 z=run(mode,[str(O/'client'),mode],env)
 assert ('CLR_GPU_CLIENT_MODEL_PASS' if mode=='normal' else 'CLR_GPU_CLIENT_MODEL_RETAIN mode='+mode) in z.stdout
for n,z in inputs.items():assert h(n)==z,n
r=dict(classification='actual-clr-native-client-synthetic-transport',sanitizer=q['sanitizer'],modes=modes,inputs=inputs,commands=commands,artifacts={p.name:h(p) for p in O.iterdir() if p.is_file()},gpu_executed=False)
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(h(O/'receipt.json'))
