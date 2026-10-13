"""Source-bound actual ROCclr adapter model qualification; no native GPU claims."""
from pathlib import Path
import argparse,json,subprocess,hashlib,shlex,shutil,threading,concurrent.futures,os
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sanitizer',choices=['address','thread','none'],default='address');a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False);P=R/'results/clr-source-v7';old=B/'results/clr-native-combined-v6'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();inputs={};commands=[];lock=threading.Lock()
def bind(p,expected=None):
 p=Path(p).resolve();h=H(p)
 if expected is not None:assert h==expected,str(p)
 with lock:
  assert str(p) not in inputs or inputs[str(p)]==h,str(p);inputs[str(p)]=h
 return p
native=json.loads(bind(old/'receipt.json').read_text());linux=json.loads(bind(B/'results/clr-combined-workload-v1/receipt.json').read_text())
prepared=json.loads(bind(P/'preparation.json').read_text())
bind(R/'tools/patch_clr_lifetime.py',prepared['lifetime_generator_sha256'])
bind(R/'tools/patch_clr_lifetime_v6.py',prepared['lifetime_v6_sha256'])
bind(R/'tools/patch_clr_lifetime_v7.py',prepared['lifetime_v7_sha256'])
for path,digest in prepared['changed_sources'].items():bind(P/path,digest)
for q in [Path(__file__),R/'tools/prepare_clr_adapter.py',R/'dependencies.json',R/'ports/clr/ps5-device.cpp',R/'ports/clr/ps5-device.hpp',R/'ports/clr/ps5-device-unsupported.inc',R/'ports/clr/ps5-virtual-unsupported.inc',R/'tests/clr_adapter_model.cpp',P/'ps5-aot.patch']:bind(q)
compiler_root=B/'.local/g4-native-workers';proof_dir=compiler_root/'results/workers-compiler-v2'
proof=json.loads(bind(proof_dir/'receipt.json','f6c57047764720fbf6b61c7eb2d03939a914dc50a0942099d2b33a4a572f2d33').read_text())
for path,digest in proof['sources'].items():bind(compiler_root/path,digest)
for path,digest in proof['inputs'].items():bind(path,digest)
for value in proof['tools'].values():bind(value['path'],value['sha256'])
for path,digest in proof['artifacts'].items():bind(proof_dir/path,digest)
artifact=proof_dir/'ps5/threads.hsaco'
argv=next(c['argv'] for c in linux['commands'] if c['label']=='linux-compile-0');flags=argv[:argv.index('-c')]
flags=[x.replace(str(old/'source'),str(P/'source')) for x in flags]
sdk=B/'.deps'/('sdk-source-'+json.loads((R/'dependencies.json').read_text())['sdk']['revision'])/'platform/include'
flags+=['-DWITH_PS5_DEVICE','-DPS5HIP_CLR_MODEL','-I'+str(R/'include'),'-I'+str(R/'ports/clr'),'-I'+str(sdk)]
if a.sanitizer!='none':flags+=['-fsanitize='+('address,undefined' if a.sanitizer=='address' else 'thread'),'-fno-omit-frame-pointer']
for name in ('clang++-18','ld.lld-18'):bind(shutil.which(name))
def run(label,args,expected=0,env=None):
 result=subprocess.run(args,capture_output=True,text=True,timeout=90,env=env)
 (O/(label+'.stdout')).write_text(result.stdout);(O/(label+'.stderr')).write_text(result.stderr)
 with lock:
  commands.append(dict(label=label,argv=args,exit_code=result.returncode,expected_exit=expected,sanitizer_environment={k:(env or os.environ).get(k) for k in ('ASAN_OPTIONS','UBSAN_OPTIONS','TSAN_OPTIONS')}));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert (result.returncode==0 if expected==0 else result.returncode!=0),(label,result.stderr[-5000:])
 return result
sources=[('clr-'+str(i),Path(u['unit'].replace(str(old/'source'),str(P/'source')))) for i,u in enumerate(native['units'])]
sources[-1]=('clr-27',R/'ports/clr/ps5-clr-tls.cpp')
sources+=[('adapter',R/'ports/clr/ps5-device.cpp'),('model',R/'tests/clr_adapter_model.cpp')]
sources += [('core-'+n,R/'src'/(n+'.cpp')) for n in ['code_object','fat_binary','host_launch','launch','native_memory','native_data','native_launch','submission','hip_launch_profile']]
for _,source in sources:bind(source)
def compile_one(pair):
 label,source=pair;dep=O/(label+'.d')
 run(label+'-headers',flags+['-M',str(source),'-MF',str(dep)])
 for path in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):bind(path)
 run(label,flags+['-c',str(source),'-o',str(O/(label+'.o'))])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(compile_one,sources))
run('link',flags+['-fuse-ld=lld-18']+[str(O/(label+'.o')) for label,_ in sources]+['-Wl,-z,defs','-o',str(O/'model')])
modes=['normal','callback','badbinary','truncated','unsupported','destroy','submit','throw','kick','timeout','postflush','cleanup','noop','corrupt']
for mode in modes:
 env=dict(os.environ)
 # Deliberate quarantine is checked with exact ownership counts, not leak silence.
 if a.sanitizer=='address':
  env['ASAN_OPTIONS']='protect_shadow_gap=0:detect_leaks='+('1' if mode in ('normal','callback','badbinary','truncated','unsupported') else '0')
  env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
 result=run(mode,[str(O/'model'),str(artifact),mode],expected=1 if mode in ('noop','corrupt') else 0,env=env)
 if mode not in ('noop','corrupt'):assert 'CLR_PS5_ADAPTER_MODEL_PASS mode='+mode+' gpu_execution=false' in result.stdout
 else:assert 'CLR_OUTPUT_ORACLE_REJECT mode='+mode+' index=0' in result.stderr and 'CLR_ADAPTER_MODEL assertion' not in result.stderr
assert all(H(path)==digest for path,digest in inputs.items()),'input drift'
report=dict(classification='source-bound-real-clr-ps5-adapter-model',sanitizer=a.sanitizer,inputs=inputs,commands=commands,modes=modes,source_units=len(sources),genuine_code_object_sha256=H(artifact),artifacts={str(q.relative_to(O)):H(q) for q in O.iterdir() if q.is_file()},gpu_executed=False,native_linked=False)
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(H(O/'receipt.json'))
