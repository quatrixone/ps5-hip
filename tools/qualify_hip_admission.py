"""Fresh genuine upstream HIP/ROCclr sanitizer closure, with model-only transport."""
from pathlib import Path
import argparse,json,subprocess,hashlib,shlex,shutil,threading,concurrent.futures,os,re
if not __debug__:
 raise RuntimeError('Validation requires ordinary Python')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sanitizer',choices=['address','thread'],required=True);p.add_argument('--flags-only',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1];root=r.parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
inputs={};commands=[];lock=threading.Lock()
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p).resolve();v=h(p)
 with lock:
  assert str(p) not in inputs or inputs[str(p)]==v,str(p);inputs[str(p)]=v
 return p
old=root/'.local/g5-clr-ps5-adapter/results/asan-v7';proof=json.loads(bind(old/'receipt.json').read_text())
assert h(old/'receipt.json')=='61717102ab598b9aa7563eadf7b7dcd431e3785f7920b600605a5c67fb2dfd59'
command=next(c['argv'] for c in proof['commands'] if c['label']=='adapter')
flags=command[:command.index('-c')]
base=root/'.local/g5-clr-ps5-adapter';source=r/'results/source-v1'
flags=[x.replace(str(base/'results/clr-source-v7/source'),str(source)).replace(str(base/'include'),str(r/'include')).replace(str(base/'ports/clr'),str(r/'ports/clr')) for x in flags]
flags+=['-DPS5HIP_PUBLIC_HIP','-D__HIP_PLATFORM_AMD__','-DDISABLE_DIRECT_DISPATCH','-ffunction-sections','-fdata-sections','-I'+str(source/'hipamd/src'),'-I'+str(root/'results/hip-host-abi-spike/rocm-6.2.4/package-stage/include'),'-I'+str(root/'results/hip-host-abi-spike/rocm-6.2.4/package-build/include')]
flags=[x.replace('-fsanitize=address,undefined','-fsanitize=thread') if a.sanitizer=='thread' else x for x in flags]
flags[1:1]=['-I'+str(r/'ports/clr')]
if a.flags_only:
 (out/'flags.json').write_text(json.dumps(flags,indent=2)+'\n');print(h(out/'flags.json'));raise SystemExit(0)
bind(__file__);bind(shutil.which('clang++-18'));bind(shutil.which('ld.lld-18'))
sources=[]
for c in proof['commands']:
 if '-c' not in c['argv']:continue
 label=c['label'];s=c['argv'][c['argv'].index('-c')+1]
 s=s.replace(str(root/'.local/g5-clr-ps5-adapter/results/clr-source-v7/source'),str(r/'results/source-v1')).replace(str(root/'.local/g5-clr-ps5-adapter'),str(r))
 if label=='model':s=str(r/'tests/hip_admission_model.cpp')
 sources.append((label,Path(s)))
sources += [(s.stem,s) for s in sorted((r/'results/source-v1/hipamd/src').glob('*.cpp')) if s.name not in ('hip_runtime.cpp','fixme.cpp')]
for _,s in sources:bind(s)
def run(label,argv,env=None,expected=0):
 z=subprocess.run(argv,capture_output=True,text=True,timeout=120,env=env);(out/(label+'.stdout')).write_text(z.stdout);(out/(label+'.stderr')).write_text(z.stderr)
 with lock:
  commands.append(dict(label=label,argv=argv,exit_code=z.returncode,environment={k:(env or os.environ).get(k) for k in ('ASAN_OPTIONS','UBSAN_OPTIONS','TSAN_OPTIONS')}));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert z.returncode==expected,(label,z.returncode,z.stderr[-3000:]);return z

def compile(pair):
 label,s=pair;f=flags.copy()
 if label in ('hip_api_trace','hip_table_interface','hip_table_interface_c'):f += ['-include','hip/hip_runtime.h','-include','hip/hip_deprecated.h','-include','hip/amd_detail/amd_hip_gl_interop.h']
 dep=out/(label+'.d');run(label+'-headers',f+['-M',str(s),'-MF',str(dep)])
 for q in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):bind(q)
 run(label,f+['-c',str(s),'-o',str(out/(label+'.o'))]);print(label,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(compile,sources))
link=flags+['-fuse-ld=lld-18','-Wl,--gc-sections','-Wl,-z,defs','-Wl,--wrap=pthread_create','-Wl,--wrap=pthread_join']+[str(out/(label+'.o')) for label,_ in sources]+['-o',str(out/'model')]
driver=run('link-driver',link+['-###']).stderr
libraries=set()
for line in driver.splitlines():
 if line.lstrip().startswith('"'):
  for token in shlex.split(line):
   path=Path(token)
   if path.is_absolute() and path.is_file():bind(path)
   if token.startswith('-l') and len(token)>2:libraries.add(token[2:])
for library in sorted(libraries):
 selected=None
 for ext in ('so','a'):
  name='lib'+library+'.'+ext
  path=Path(run('resolve-'+name,[flags[0],'-print-file-name='+name]).stdout.strip())
  if path.is_absolute() and path.is_file():selected=path;break
 assert selected is not None,library
 bind(selected)
 if not selected.read_bytes().startswith(b'\x7fELF') and selected.suffix!='.a':
  for value in re.findall(r'/[^\s()]+',selected.read_text()):
   if Path(value).is_file():bind(value)
run('link',link)
modes=['normal','default','tls','init-fail','create-fail','join-fail','self-destroy','reset','reset-join-fail']
for mode in modes:
 env=dict(os.environ);env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';env['TSAN_OPTIONS']='halt_on_error=1';env['ASAN_OPTIONS']='halt_on_error=1:detect_leaks='+('0' if mode in ('join-fail','self-destroy','reset-join-fail') else '1')
 expected=66 if a.sanitizer=='thread' and mode in ('join-fail','reset-join-fail') else 0
 z=run(mode,[str(out/'model'),mode],env,expected);assert 'HIP_PUBLIC_PASS mode='+mode in z.stdout
 if expected:
  # An intentionally unconsumed join must remain observable, not suppressed.
  assert z.stderr.count('WARNING: ThreadSanitizer:')==1 and 'WARNING: ThreadSanitizer: thread leak' in z.stderr
  assert 'finished) created by main thread' in z.stderr and 'data race' not in z.stderr
 if mode!='init-fail':
  n=1 if mode=='create-fail' else 4 if mode in ('default','tls') else 3
  retained=mode in ('join-fail','self-destroy','reset-join-fail')
  assert f'HIP_PUBLIC_FINAL queues={n} retired={2 if mode=="self-destroy" else n} gate_destroyed={int(not retained)} retained={int(retained)}' in z.stdout
 print(mode,'PASS',flush=True)
assert all(h(p)==v for p,v in inputs.items()),'source/header/tool drift'
report=dict(classification='genuine-upstream-public-hip-admission-model',sanitizer=a.sanitizer,inputs=inputs,commands=commands,modes=modes,source_units=len(sources),gpu_executed=False,artifacts={str(p.relative_to(out)):h(p) for p in out.iterdir() if p.is_file()})
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(h(out/'receipt.json'))
